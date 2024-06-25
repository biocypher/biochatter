from abc import ABC, abstractmethod
from typing import Any, Callable
from langchain_core.prompts import ChatPromptTemplate
from biochatter.llm_connect import Conversation
from pydantic import BaseModel


class AbstractQueryBuilder(ABC):
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
        api_fields: "BaseModel",
        conversation: "Conversation",
    ) -> Callable:
        """
        Creates a runnable object for executing queries. Must be implemented by
        subclasses. Should use the LangChain `create_structured_output_runnable`
        method to generate the callable.

        Args:
            api_fields: A Pydantic data model that specifies the fields of the
                API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
            A callable object that can execute the query.
        """
        pass

    @abstractmethod
    def generate_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> BaseModel:
        """
        Generates a query object (a parameterised Pydantic model with the fields
        of the API) based on the given question using a BioChatter conversation
        instance. Must be implemented by subclasses.

        Args:
            question (str): The question to be answered.

            conversation: The BioChatter conversation object containing the LLM
                that should parameterise the query.

        Returns:
            A parameterised instance of the query object (Pydantic BaseModel)
        """
        pass
