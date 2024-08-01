from abc import ABC, abstractmethod
from typing import Any, Optional
from collections.abc import Callable

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from biochatter.llm_connect import Conversation


class BaseQueryBuilder(ABC):
    """
    An abstract base class for query builders.
    """

    @property
    def structured_output_prompt(self) -> ChatPromptTemplate:
        """
        Defines a structured output prompt template. This provides a default
        implementation for an API agent that can be overridden by subclasses to
        return a ChatPromptTemplate-compatible object.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class algorithm for extracting information in structured formats.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {input}",
                ),
                ("human", "Tip: Make sure to answer in the correct format"),
            ]
        )

    @abstractmethod
    def create_runnable(
        self,
        query_parameters: "BaseModel",
        conversation: "Conversation",
    ) -> Callable:
        """
        Creates a runnable object for executing queries. Must be implemented by
        subclasses. Should use the LangChain `create_structured_output_runnable`
        method to generate the Callable.

        Args:
            query_parameters: A Pydantic data model that specifies the fields of
                the API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
            A Callable object that can execute the query.
        """
        pass

    @abstractmethod
    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> BaseModel:
        """

        Parameterises a query object (a Pydantic model with the fields of the
        API) based on the given question using a BioChatter conversation
        instance. Must be implemented by subclasses.

        Args:
            question (str): The question to be answered.

            conversation: The BioChatter conversation object containing the LLM
                that should parameterise the query.

        Returns:
            A parameterised instance of the query object (Pydantic BaseModel)
        """
        pass


class BaseFetcher(ABC):
    """
    Abstract base class for fetchers. A fetcher is responsible for submitting
    queries (in systems where submission and fetching are separate) and fetching
    and saving results of queries. It has to implement a `fetch_results()`
    method, which can wrap a multi-step procedure to submit and retrieve. Should
    implement retry method to account for connectivity issues or processing
    times.
    """

    @abstractmethod
    def fetch_results(
        self,
        query_model,
        retries: Optional[int] = 3,
    ):
        """
        Fetches results by submitting a query. Can implement a multi-step
        procedure if submitting and fetching are distinct processes (e.g., in
        the case of long processing times as in the case of BLAST).

        Args:
            query_model: the Pydantic model describing the parameterised query
        """
        pass


class BaseInterpreter(ABC):
    """
    Abstract base class for result interpreters. The interpreter is aware of the
    nature and structure of the results and can extract and summarise
    information from them.
    """

    @abstractmethod
    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable,
        response_text: str,
    ) -> str:
        """
        Summarises an answer based on the given parameters.

        Args:
            question (str): The question that was asked.

            conversation_factory (Callable): A function that creates a
                BioChatter conversation.

            response_text (str): The response.text returned from the request.

        Returns:
            A summary of the answer.

        Todo:
            Genericise (remove file path and n_lines parameters, and use a
            generic way to get the results). The child classes should manage the
            specifics of the results.
        """
        pass
