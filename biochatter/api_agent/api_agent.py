from typing import Optional
from collections.abc import Callable
import os

from pydantic import BaseModel

from .abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder

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


## Agent class
class APIAgent:
    def __init__(
        self,
        conversation_factory: Callable,
        query_builder: "BaseQueryBuilder",
        result_fetcher: "BaseFetcher",
        result_interpreter: "BaseInterpreter",
    ):
        """

        API agent class to interact with a tool's API for querying and fetching
        results.  The query fields have to be defined in a Pydantic model
        (`BaseModel`) and used (i.e., parameterised by the LLM) in the query
        builder. Specific API agents are defined in submodules of this directory
        (`api_agent`). The agent's logic is implemented in the `execute` method.

        Attributes:
            conversation_factory (Callable): A function used to create a
                BioChatter conversation, providing LLM access.

            query_builder (BaseQueryBuilder): An instance of a child of the
                BaseQueryBuilder class.

            result_fetcher (BaseFetcher): An instance of a child of the
                BaseFetcher class.

            result_interpreter (BaseInterpreter): An instance of a child of the
                BaseInterpreter class.
        """
        self.conversation_factory = conversation_factory
        self.result_path = ".api_results/"
        self.query_builder = query_builder
        self.result_fetcher = result_fetcher
        self.result_interpreter = result_interpreter

        os.makedirs(self.result_path, exist_ok=True)

    def parameterise_query(self, question: str) -> Optional[BaseModel]:
        """
        Use LLM to parameterise a query (a Pydantic model) based on the given
        question using a BioChatter conversation instance.
        """
        try:
            conversation = self.conversation_factory()
            return self.query_builder.parameterise_query(question, conversation)
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

    def execute(self, question: str) -> Optional[str]:
        """
        Wrapper that uses class methods to execute the API agent logic. Consists
        of 1) query generation, 2) query submission, 3) results fetching, and
        4) answer extraction. The final answer is stored in the final_answer
        attribute.

        Args:
            question (str): The question to be answered.
        """
        # Generate query
        try:
            query = self.parameterise_query(question)
            if not query:
                raise ValueError("Failed to generate query.")
        except ValueError as e:
            print(e)

        # Submit query and get RID
        try:
            rid = self.submit_query(query)
            if not rid:
                raise ValueError("Failed to submit query.")
        except ValueError as e:
            print(e)

        print(f"Received RID: {rid}")

        # Fetch results
        try:
            file_name = self.fetch_results(query.question_uuid, rid)
            if not file_name:
                raise ValueError("Failed to fetch results.")
        except ValueError as e:
            print(e)

        # Extract answer from results
        try:
            final_answer = self.summarise_results(question, file_name)
            if not final_answer:
                raise ValueError("Failed to extract answer from results.")
        except ValueError as e:
            print(e)

        return final_answer
