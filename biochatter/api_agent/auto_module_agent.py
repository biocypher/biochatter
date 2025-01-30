"""Module for automatically generating API agents from Python modules."""

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from .abc import BaseAPIModel, BaseQueryBuilder, BaseTools
from .collect_tool_info_from_module import collect_tool_info

class AutoModuleTools(BaseTools):
    """A class for automatically generating tools from a Python module."""

    def __init__(self, module: ModuleType, module_description: str = ""):
        """Initialize the tools by generating Pydantic models from the module.
        
        Args:
        ----
            module: The Python module to generate tools from
            module_description: Optional description of the module's functionality
        """
        self.module = module
        self.module_description = module_description
        self.tools_params, self.tools_descriptions = collect_tool_info(module)

class AutoModuleQueryBuilder(BaseQueryBuilder):
    """A class for building queries for automatically generated module tools."""

    def __init__(self, module: ModuleType):
        """Initialize the query builder with a module.
        
        Args:
        ----
            module: The Python module to generate tools from
        """
        self.module = module
        self.pydantic_tools = None

    def create_runnable(
        self,
        query_parameters: list[BaseAPIModel],
        conversation: "Conversation",
    ) -> Callable:
        """Create a runnable object for executing queries.

        Create runnable using the LangChain `create_structured_output_runnable`
        method.

        Args:
        ----
            query_parameters: A list of Pydantic models that specify the fields
                of the API that should be queried.
            conversation: A BioChatter conversation object.

        Returns:
        -------
            A Callable object that can execute the query.
        """
        runnable = conversation.chat.bind_tools(query_parameters)
        return runnable | PydanticToolsParser(tools=query_parameters)

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> list[BaseModel]:
        """Generate query parameters based on the question.

        Uses the automatically generated tools to parameterize the query based on
        the given question and BioChatter conversation.

        Args:
        ----
            question: The question to be answered.
            conversation: The conversation object used for parameterising the query.

        Returns:
        -------
            list[BaseModel]: The parameterised query objects (Pydantic models)
        """

        tool_maker = AutoModuleTools(self.module)
        if self.pydantic_tools is None:
            self.pydantic_tools = tool_maker.make_pydantic_tools()
        
        runnable = self.create_runnable(
            conversation=conversation,
            query_parameters=self.pydantic_tools
        )
        return runnable.invoke(question)
