"""Base API agent module."""

from collections.abc import Callable

from pydantic import BaseModel

from .agent_abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder

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
        fetcher: "BaseFetcher",
        interpreter: "BaseInterpreter",
    ):
        """API agent class to interact with a tool's API for querying and fetching
        results.  The query fields have to be defined in a Pydantic model
        (`BaseModel`) and used (i.e., parameterised by the LLM) in the query
        builder. Specific API agents are defined in submodules of this directory
        (`api_agent`). The agent's logic is implemented in the `execute` method.

        Attributes
        ----------
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
        self.query_builder = query_builder
        self.fetcher = fetcher
        self.interpreter = interpreter
        self.final_answer = None

    def parameterise_query(self, question: str) -> list[BaseModel] | None:
        """Use LLM to parameterise a query (a list of Pydantic models) based on the given
        question using a BioChatter conversation instance.
        """
        try:
            conversation = self.conversation_factory()
            return self.query_builder.parameterise_query(question, conversation)
        except Exception as e:
            print(f"Error generating query: {e}")
            return None

    def fetch_results(self, query_models: list[BaseModel]) -> str | None:
        """Fetch the results of the query using the individual API's implementation
        (either single-step or submit-retrieve).

        Args:
        ----
            query_models: list of parameterised query Pydantic models

        """
        try:
            return self.fetcher.fetch_results(query_models, 100)
        except Exception as e:
            print(f"Error fetching results: {e}")
            return None

    def summarise_results(
        self,
        question: str,
        response_text: str,
    ) -> str | None:
        """Summarise the retrieved results to extract the answer to the question."""
        try:
            return self.interpreter.summarise_results(
                question=question,
                conversation_factory=self.conversation_factory,
                response_text=response_text,
            )
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def execute(self, question: str) -> str | None:
        """Wrapper that uses class methods to execute the API agent logic. Consists
        of 1) query generation, 2) query submission, 3) results fetching, and
        4) answer extraction. The final answer is stored in the final_answer
        attribute.

        Args:
        ----
            question (str): The question to be answered.

        """
        # Generate query
        try:
            query_models = self.parameterise_query(question)
            if not query_models:
                raise ValueError("Failed to generate query.")
        except ValueError as e:
            print(e)

        # Fetch results
        try:
            response_text = self.fetch_results(
                query_models=query_models,
            )
            if not response_text:
                raise ValueError("Failed to fetch results.")
        except ValueError as e:
            print(e)

        # Extract answer from results
        try:
            final_answer = self.summarise_results(question, response_text)
            if not final_answer:
                raise ValueError("Failed to extract answer from results.")
        except ValueError as e:
            print(e)

        self.final_answer = final_answer
        return final_answer

    def get_description(self, tool_name: str, tool_desc: str):
        return f"This API agent interacts with {tool_name}'s API for querying and fetching results. {tool_desc}"
