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
from pydantic import BaseModel
from .abc import BaseQueryBuilder, BaseFetcher, BaseInterpreter


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

        query_builder (BaseQueryBuilder): An instance of a child of the
            BaseQueryBuilder class.

        result_fetcher (BaseFetcher): An instance of a child of the BaseFetcher
            class.

        result_interpreter (BaseInterpreter): An instance of a child of the
            BaseInterpreter class.
    """

    def __init__(
        self,
        conversation_factory: callable,
        query_builder: "BaseQueryBuilder",
        result_fetcher: "BaseFetcher",
        result_interpreter: "BaseInterpreter",
    ):
        self.conversation_factory = conversation_factory
        self.result_path = ".api_results/"
        self.query_builder = query_builder
        self.result_fetcher = result_fetcher
        self.result_interpreter = result_interpreter
        self.final_answer = None
        self.error = None

        os.makedirs(self.result_path, exist_ok=True)

    def generate_query(self, question: str) -> Optional[BaseModel]:
        """
        Use LLM to generate a query (a Pydantic model) based on the given
        question using a BioChatter conversation instance.
        """
        try:
            conversation = self.conversation_factory()
            return self.query_builder.generate_query(question, conversation)
        except Exception as e:
            print(f"Error generating query: {e}")
            return None

    def submit_query(self, api_fields: BaseModel) -> Optional[str]:
        """
        Submit the generated query to the API and return the RID.
        """
        try:
            return self.result_fetcher.submit_query(api_fields)
        except Exception as e:
            print(f"Error submitting query: {e}")
            return None

    def fetch_results(self, question_uuid: str, rid: str) -> Optional[str]:
        """
        Fetch the results of the query using the RID and save them. Implements
        retry logic to fetch results.
        """
        try:
            return self.result_fetcher.fetch_and_save_results(
                question_uuid, rid, self.result_path, 100
            )
        except Exception as e:
            print(f"Error fetching results: {e}")
            return None

    def summarise_results(self, question: str, file_name: str) -> Optional[str]:
        """
        Summarise the retrieved results to extract the answer to the question.
        """
        try:
            file_path = os.path.join(self.result_path, file_name)
            return self.result_interpreter.summarise_results(
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
        final_answer = self.summarise_results(question, file_name)
        if not final_answer:
            print("Failed to extract answer from results.")
            self.error = f"Failed to extract answer from results."
            return

        if final_answer:
            print(f"Final Answer: {final_answer}")
            self.final_answer = final_answer
            return
