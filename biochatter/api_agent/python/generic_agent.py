"""Module for ingesting any Python module and generating a query builder."""

from collections.abc import Callable
from types import ModuleType

from langchain_core.output_parsers import PydanticToolsParser

from biochatter.api_agent.base.agent_abc import BaseAPIModel, BaseQueryBuilder
from biochatter.api_agent.python.autogenerate_model import generate_pydantic_classes
from biochatter.llm_connect import Conversation


class GenericQueryBuilder(BaseQueryBuilder):
    """A class for building a generic query using LLM tools.

    The query builder works by ingesting a Python module and generating a list
    of Pydantic classes for each callable in the module. It then uses these
    classes to parameterise a query using LLM tool binding.
    """

    def create_runnable(
        self,
        query_parameters: list["BaseAPIModel"],
        conversation: Conversation,
    ) -> Callable:
        """Create a runnable object for the query builder.

        Args:
        ----
            query_parameters: The list of Pydantic classes to be used for the
                query.

            conversation: The conversation object used for parameterising the
                query.

        Returns:
        -------
            The runnable object for the query builder.

        """
        runnable = conversation.chat.bind_tools(query_parameters, tool_choice="required")
        return runnable | PydanticToolsParser(tools=query_parameters)

    def parameterise_query(
        self,
        question: str,
        prompt: str,
        conversation: "Conversation",
        module: ModuleType,
        generated_classes: list[BaseAPIModel] | None = None,
    ) -> list[BaseAPIModel]:
        """Parameterise tool calls for any Python module.

        Generate a list of parameterised BaseModel instances based on the given
        question, prompt, and BioChatter conversation. Uses a Pydantic model
        to define the API fields.

        Using langchain's `bind_tools` method to allow the LLM to parameterise
        the function call, based on the functions available in the module.

        Relies on defined structure and annotation of the passed module.

        Args:
        ----
            question (str): The question to be answered.

            prompt (str): The prompt to be used for the query, instructing the
                LLM of its task and the module context.

            conversation: The conversation object used for parameterising the
                query.

            module: The Python module to be used for the query.

            generated_classes: The list of Pydantic classes to be used for the
                query. If not provided, the classes will be generated from the
                module. Allows for external injection of classes for testing
                purposes.

        Returns:
        -------
            list[BaseAPIModel]: the parameterised query object (Pydantic
                model)

        """
        if generated_classes is None:
            tools = generate_pydantic_classes(module)

        runnable = self.create_runnable(
            conversation=conversation,
            query_parameters=tools,
        )

        query = [
            ("system", prompt),
            ("human", question),
        ]

        return runnable.invoke(query)
