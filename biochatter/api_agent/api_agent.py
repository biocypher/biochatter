### logic

# 1. User asks question req. API to find the answer
# 2. rag_agent receives the question and sends it to the api_agent // api_agent is manually selected
# 3. api_agent writes query for API specific to the question
# 3.1 question + API prompt template + BlastQuery are input into
# langchain.chains.openai_functions.create_structured_output_runnable -> returns a structured output which is our API call object
# 4. API call
# 4.1 API call is made using the structured output from 3.1
# 4.2 API returns a response which is saved
#
# 5. Read response from and uses it to answer question
# 6. answer is returned to rag_agent

from typing import Optional
import os

from .blast import BlastFetcher, BlastQueryBuilder, BlastQuery, BlastInterpreter
from .abc import BaseQueryBuilder


## Agent class
class APIAgent:
    """
    A class to interact with a tool's API for querying and fetching results.
    The query fields have to be defined in a Pydantic model (`BaseModel`) and
    used (i.e., parameterised by the LLM) in the query builder.

    Attributes:
        conversation_factory (callable): A function used to create a
            BioChatter conversation.

        result_path (str): The path to save results.

        query_builder (BlastQueryBuilder): An instance of the BlastQueryBuilder
            class.

        result_fetcher (BlastFetcher): An instance of the BlastFetcher class.

        result_interpreter (BlastInterpreter): An instance of the
            BlastInterpreter class.
    """

    def __init__(
        self,
        conversation_factory: callable,
        query_builder: "BaseQueryBuilder",
        result_fetcher: "BlastFetcher",
        result_interpreter: "BlastInterpreter",
    ):
        self.conversation_factory = conversation_factory
        self.result_path = ".blast"
        self.query_builder = query_builder
        self.result_fetcher = result_fetcher
        self.result_interpreter = result_interpreter
        self.final_answer = None
        self.error = None

        os.makedirs(self.result_path, exist_ok=True)

    def generate_query(self, question: str) -> Optional[BlastQuery]:
        try:
            conversation = self.conversation_factory()
            return self.query_builder.generate_query(question, conversation)
        except Exception as e:
            print(f"Error generating query: {e}")
            return None

    def submit_query(self, api_fields: BlastQuery) -> Optional[str]:
        try:
            return self.result_fetcher.submit_query(api_fields)
        except Exception as e:
            print(f"Error submitting query: {e}")
            return None

    def fetch_results(self, question_uuid: str, rid: str) -> Optional[str]:
        try:
            return self.result_fetcher.fetch_and_save_results(
                question_uuid, rid, self.result_path, 100
            )
        except Exception as e:
            print(f"Error fetching results: {e}")
            return None

    def extract_answer(self, question: str, file_name: str) -> Optional[str]:
        try:
            file_path = os.path.join(self.result_path, file_name)
            return self.result_interpreter.answer_extraction(
                question, file_path, 100
            )
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def execute(self, question: str):
        """
        Wrapper that uses class methods to execute the API agent logic. Consists
        of 1) query generation, 2) query submission, 3) results fetching, and
        4) answer extraction. The final answer is stored in the final_answer
        attribute.

        Args:
            question (str): The question to be answered.
        """
        # Generate query
        query = self.generate_query(question)
        if not query:
            print("Failed to generate query.")
            self.error = "Failed to generate query."
            return

        print(f"Generated query: {query}")

        # Submit query and get RID
        rid = self.submit_query(query)
        if not rid:
            print("Failed to submit query.")
            self.error = "Failed to submit query."
            return

        print(f"Received RID: {rid}")

        # Fetch results
        file_name = self.fetch_results(query.question_uuid, rid)
        if not file_name:
            print("Failed to fetch results.")
            self.error = "Failed to fetch results."
            return

        # Extract answer from results
        final_answer = self.extract_answer(question, file_name)
        if not final_answer:
            print("Failed to extract answer from results.")
            self.error = f"Failed to extract answer from results."
            return

        if final_answer:
            print(f"Final Answer: {final_answer}")
            self.final_answer = final_answer
            return
