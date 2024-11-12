from typing import Optional
from collections.abc import Callable
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.openai_functions import create_structured_output_runnable
import requests

from biochatter.llm_connect import Conversation
from .abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder

BRAPI_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use the web API of Breeding API (BrAPI) to answer questions about <content>.

You have to extract the appropriate information out of the examples:
1. To list information about the tools, use the endpoint <endpoint> with parameters like <param examples>.
2. <Other endpoints?>

Use these formats to generate queries based on the question provided. Below is more information about the BrAPI API:

Base URL

<base URL>

Endpoints and Parameters

<general API usage, examples, copy some parts of the BrAPI docs?>

"""


BRAPI_SUMMARY_PROMPT = """
You have to answer this question in a clear and concise manner: {question} Be factual!\n\
You are a world leading bioinformatician who knows everything about the Breeding API.\n\
Do not make up information, only use the provided information and mention how relevant the found information is based on your knowledge.\n\
Here is the information relevant to the question found on the BrAPI web API:\n\
{context}
"""


class BrAPIQueryParameters(BaseModel):
    base_url: str = Field(
        default="<base URL>",
        description="Base URL for the BrAPI API.",
    )
    endpoint: str = Field(
        ...,
        description="Specific API endpoint to hit. Example: 't/' for listing tools.",
    )
    # <any other field, exactly as it is spelled in the API, with an optional
    # default value and a useful description>
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )


class BrAPIQueryBuilder(BaseQueryBuilder):
    """A class for building an BrAPIQuery object."""

    def create_runnable(
        self,
        query_parameters: "BrAPIQueryParameters",
        conversation: "Conversation",
    ) -> Callable:
        """
        Creates a runnable object for executing queries using the LangChain
        `create_structured_output_runnable` method.

        Args:
            query_parameters: A Pydantic data model that specifies the fields of
                the API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
            A Callable object that can execute the query.
        """
        return create_structured_output_runnable(
            output_schema=query_parameters,
            llm=conversation.chat,
            prompt=self.structured_output_prompt,
        )

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> BrAPIQueryParameters:
        """

        Generates an BrAPIQuery object based on the given question, prompt,
        and BioChatter conversation. Uses a Pydantic model to define the API
        fields.  Creates a runnable that can be invoked on LLMs that are
        qualified to parameterise functions.

        Args:
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                BrAPIQuery.

        Returns:
            BrAPIQueryParameters: the parameterised query object (Pydantic model)
        """
        runnable = self.create_runnable(
            query_parameters=BrAPIQueryParameters,
            conversation=conversation,
        )
        oncokb_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n {BRAPI_QUERY_PROMPT}"}
        )
        oncokb_call_obj.question_uuid = str(uuid.uuid4())
        return oncokb_call_obj


class BrAPIFetcher(BaseFetcher):
    """
    A class for retrieving API results from BrAPI given a parameterized
    BrAPIQuery.
    """

    def __init__(self, api_token="demo"):
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        }
        self.base_url = "<base URL>"

    def fetch_results(
        self, request_data: BrAPIQueryParameters, retries: Optional[int] = 3
    ) -> str:
        """Function to submit the BrAPI query and fetch the results directly.
        No multi-step procedure, thus no wrapping of submission and retrieval in
        this case.

        Args:
            request_data: BrAPIQuery object (Pydantic model) containing the
                BrAPI query parameters.

        Returns:
            str: The results of the BrAPI query.
        """
        # Submit the query and get the URL
        params = request_data.dict(exclude_unset=True)
        endpoint = params.pop("endpoint")
        params.pop("question_uuid")
        full_url = f"{self.base_url}/{endpoint}"
        response = requests.get(full_url, headers=self.headers, params=params)
        response.raise_for_status()

        # Fetch the results from the URL
        results_response = requests.get(response.url, headers=self.headers)
        results_response.raise_for_status()

        return results_response.text


class BrAPIInterpreter(BaseInterpreter):
    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable,
        response_text: str,
    ) -> str:
        """
        Function to extract the answer from the BrAPI results.

        Args:
            question (str): The question to be answered.
            conversation_factory: A BioChatter conversation object.
            response_text (str): The response.text returned by BrAPI.

        Returns:
            str: The extracted answer from the BrAPI results.

        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class bioinformatician who knows "
                    "everything about the Breeding API (BrAPI) and its "
                    "contents. Your task is to interpret "
                    "results from BrAPI API calls and summarise "
                    "them for the user.",
                ),
                ("user", "{input}"),
            ]
        )
        summary_prompt = BRAPI_SUMMARY_PROMPT.format(
            question=question, context=response_text
        )
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        answer = chain.invoke({"input": {summary_prompt}})
        return answer
