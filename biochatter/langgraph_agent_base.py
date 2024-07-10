
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Union
import logging
from enum import Enum

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_core.pydantic_v1 import (
    ValidationError
)
from langsmith import (
    traceable
)
from langgraph.graph import MessageGraph, END
from langgraph.graph.graph import CompiledGraph

logger = logging.getLogger(__name__)
class ResponderWithRetries:
    """
    Raise request to LLM with 3 retries
    """
    def __init__(self, runnable, validator):
        """
        Args:
        runnable: LLM agent
        validator: used to validate response
        """
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        """
        Invoke LLM agent, this function will be called by LangGraph
        Args:
        state List[BaseMessage]: message history
        """
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response 

DRAFT_NODE = "draft"
EXECUTE_TOOL_NODE = "execute_tool"
REVISE_NODE = "revise"
END_NODE = END

class ReflexionAgent(ABC):
    """
    LLM agent reflexion framework:

    start -> draft -> execute tool -> revise -> evaluation -> end
                        /|\                        |
                         ---------------------------
    """
    RECURSION_LIMIT = 30
    def __init__(
        self,
        conversation_factory: Callable,
        max_steps: Optional[int]=20,
    ):
        """
        Args:
          conversation_factory Callable: the callback to create Conversation
          max_steps int: max steps for reflextion loop
        """
        if max_steps <= 0:
            max_steps = ReflexionAgent.RECURSION_LIMIT
        recursion_limit = ReflexionAgent.RECURSION_LIMIT
        if recursion_limit < max_steps:
            recursion_limit = max_steps
        self.initial_responder = None
        self.revise_responder = None
        self.max_steps = max_steps
        self.recursion_limit = recursion_limit
        self._logs: str = ""
        self.conversation = conversation_factory()
    
    def _should_continue(self, state: List[BaseMessage]):
        """
        Determine if we need to continue reflexion
        Args:
          state List[BaseMessage]: message history
        """
        num_steps = ReflexionAgent._get_num_iterations(state)
        if num_steps > self.max_steps:
            return END_NODE
        return EXECUTE_TOOL_NODE

    @abstractmethod
    def _tool_function(self, state: List[BaseMessage]):
        """
        tool function, execute tool based on initial draft or revised answer
        Args:
          state List[BaseMessage]: message history
        """
        pass

    @abstractmethod
    def _create_initial_responder(self, prompt: Optional[str]=None) -> ResponderWithRetries:
        """
        draft responder, draft initial answer
        Args:
          prompt str: prompt for LLM to draft initial answer
        """
        pass

    @abstractmethod
    def _create_revise_responder(self, prompt: Optional[str]=None) -> ResponderWithRetries:
        """
        revise responder, revise answer according to tool function result 
        Args:
          prompt str: prompt for LLM to draft initial answer
        """
        pass

    @abstractmethod
    def _log_step_message(self, step: int, node: str, output: BaseMessage) -> None:
        """
        log step message
        Args:
          step int: step index
          output BaseMessage: step message
        """
        pass

    @abstractmethod
    def _log_final_result(self, output: BaseMessage) -> None:
        """
        log final result
        Args:
          output BaseMessage: last step message
        """
        pass

    @abstractmethod
    def _parse_final_result(self, output: BaseMessage) -> str | None:
        """
        parse the result of the last step
        Args:
          output BaseMessage: last step message
        Returns:
          str | None: the parsed reuslt of the last step
        """
        pass

    def _log_message(self, msg: str = "", level: Optional[Literal["info", "error", "warn"]]="info"):
        """
        Save logs message
        """
        logger_func = logger.info if level == "info" else (logger.error if level == "error" else logger.warn)
        logger_func(msg)
        self._logs = self._logs + f"[{level}]" + f"{datetime.now().isoformat()} - {msg}\n"

    @property
    def logs(self):
        return self._logs
    
    @staticmethod
    def _get_num_iterations(state: List[BaseMessage]):
        """
        Calculate iteration number
        Args:
          state List[BaseMessage]: message history

        Returns:
          int: the iterations number 
        """
        i = 0
        for m in state[::-1]:
            if not isinstance(m, (ToolMessage, AIMessage)):
                break
            i += 1
        return i

    def _build_graph(self, prompt: Optional[str]=None):
        """
        Build Langgraph graph
        Args:
          prompt str: prompt for LLM
        Returns:
          CompiledGraph | None: a built Langgraph graph. If there
                                are errors ocurred, it returns None
        """
        try:
            self.initial_responder = self._create_initial_responder(prompt)
            self.revise_responder = self._create_revise_responder(prompt)
            builder = MessageGraph()
            builder.add_node(DRAFT_NODE, self.initial_responder.respond)
            builder.add_node(EXECUTE_TOOL_NODE, self._tool_function)
            builder.add_node(REVISE_NODE, self.revise_responder.respond)
            builder.add_edge(DRAFT_NODE, EXECUTE_TOOL_NODE)
            builder.add_edge(EXECUTE_TOOL_NODE, REVISE_NODE)
    
            builder.add_conditional_edges(REVISE_NODE, self._should_continue)
            builder.set_entry_point(DRAFT_NODE)
            graph = builder.compile()
            return graph
        except Exception as e:
            logger.error(e)
            return None
        
    def _extract_result_from_final_step(self, step: Dict[str, List[BaseMessage]] | BaseMessage):
        """ 
        extract result from last step
        This function is used to facilitate test.
        """
        return step[END][-1]

    def _execute_graph(self, graph: Optional[CompiledGraph]=None, question: Optional[str]="") -> str | None:
        """
        execute Langgraph graph
        Args:
          graph CompiledGraph: Langgraph graph
          question str: user question

        Returns: 
          answer str | None: string answer parsed from Langgraph graph execution
        """
        if graph is None:
            return None
        if len(question) == 0:
            return None
        
        events = graph.stream(
            [HumanMessage(content=question)], {
                "recursion_limit": self.recursion_limit
            }
        )

        for i, step in enumerate(events):
            node, output = next(iter(step.items()))
            self._log_step_message(i+1, node, output)

        final_result = self._extract_result_from_final_step(step)
        self._log_final_result(final_result)
        return self._parse_final_result(final_result)
        
    def execute(self, question: str, prompt: Optional[str]=None) -> (str | None):
        """
        Execute ReflexionAgent
        Args:
          question str: user question
          prompt str: user prompt
        Returns:
          answer str | None: If it executes successfully, an answer to the q
                            uestion will be returned, otherwise, it returns None
        """
        if len(question) == 0:
            return None
        graph = self._build_graph(prompt)
        return self._execute_graph(graph, question)
        