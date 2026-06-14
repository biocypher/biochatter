import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import ValidationError
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable

logger = logging.getLogger(__name__)


class ReflexionAgentLogger:
    def __init__(self) -> None:
        self._logs: str = ""

    def log_step_message(
        self,
        step: int,
        node_name: str,
        output: BaseMessage,
    ):
        """Log step message
        Args:
          step int: step index
          output BaseMessage: step message
        """

    def log_final_result(self, final_result: dict[str, Any]) -> None:
        """Log final result
        Args:
          output BaseMessage: last step message
        """

    def _log_message(
        self,
        msg: str = "",
        level: Literal["info", "error", "warn"] | None = "info",
    ):
        """Save log message

        Args:
        ----
            msg: the message to be logged

            level: the log level to write

        """
        logger_func = logger.info if level == "info" else (logger.error if level == "error" else logger.warning)
        logger_func(msg)
        self._logs = self._logs + f"[{level}]" + f"{datetime.now().isoformat()} - {msg}\n"

    @property
    def logs(self):
        return self._logs


class ResponderWithRetries:
    """Raise request to LLM with 3 retries"""

    def __init__(self, runnable, validator):
        """Args:
        ----
        runnable: LLM agent
        validator: used to validate response

        """
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: list[BaseMessage]):
        """Invoke LLM agent, this function will be called by LangGraph
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


class ReflexionAgentResult:
    def __init__(self, answer: str | None, tool_result: list[Any] | None):
        self.answer = answer
        self.tool_result = tool_result


class ReflexionAgent(ABC):
    r"""LLM agent reflexion framework:

    start -> draft -> execute tool -> revise -> evaluation -> end
                        /|\                        |
                         ---------------------------
    """

    RECURSION_LIMIT = 30

    def __init__(
        self,
        conversation_factory: Callable,
        max_steps: int | None = 20,
        agent_logger: ReflexionAgentLogger | None = ReflexionAgentLogger(),
    ):
        """Args:
        ----
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
        self.conversation = conversation_factory()
        self.agent_logger = agent_logger

    def _should_continue(self, state: list[BaseMessage]):
        """Determine if we need to continue reflexion
        Args:
          state List[BaseMessage]: message history
        """
        num_steps = ReflexionAgent._get_num_iterations(state)
        if num_steps > self.max_steps:
            return END
        return EXECUTE_TOOL_NODE

    @abstractmethod
    def _tool_function(self, state: list[BaseMessage]) -> ToolMessage:
        """Tool function, execute tool based on initial draft or revised answer
        Args:
          state List[BaseMessage]: message history
        Returns:
          ToolMessage
        """

    @abstractmethod
    def _create_initial_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        """Draft responder, draft initial answer
        Args:
          prompt str: prompt for LLM to draft initial answer
        """

    @abstractmethod
    def _create_revise_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        """Revise responder, revise answer according to tool function result
        Args:
          prompt str: prompt for LLM to draft initial answer
        """

    @abstractmethod
    def _parse_final_result(
        self,
        messages: list[BaseMessage],
    ) -> ReflexionAgentResult:
        """Parse the result of the last step
        Args:
          output BaseMessage: last step message
        Returns:
          ReflexionAgentResult: the parsed reuslt of the last step
        """

    def get_logs(self):
        return self.agent_logger.logs

    @staticmethod
    def _get_num_iterations(state: list[BaseMessage]):
        """Calculate iteration number
        Args:
          state List[BaseMessage]: message history

        Returns
        -------
          int: the iterations number

        """
        i = 0
        for m in state[::-1]:
            if not isinstance(m, (ToolMessage, AIMessage)):
                break
            i += 1
        return i

    @staticmethod
    def _get_user_question(state: list[BaseMessage]):
        """Get user's question from messages array"""
        for m in state:
            if not isinstance(m, HumanMessage):
                continue
            return m.content
        return None

    @staticmethod
    def _get_last_tool_result(messages: list[BaseMessage]):
        """Get result of the last tool node"""
        for m in messages[::-1]:
            if not isinstance(m, ToolMessage):
                continue
            content = json.loads(m.content)
            return content["result"]
        return None

    def _build_graph(self, prompt: str | None = None):
        """Build Langgraph graph for execution of chained LLM processes.

        Args:
        ----
          prompt str: prompt for LLM

        Returns:
        -------
          CompiledStateGraph | None: a Langgraph graph or None in case of errors

        """
        try:
            self.initial_responder = self._create_initial_responder(prompt)
            self.revise_responder = self._create_revise_responder(prompt)
            builder = StateGraph(MessagesState)

            def draft_node(state: MessagesState):
                return {"messages": [self.initial_responder.respond(state["messages"])]}

            def execute_tool_node(state: MessagesState):
                return {"messages": [self._tool_function(state["messages"])]}

            def revise_node(state: MessagesState):
                return {"messages": [self.revise_responder.respond(state["messages"])]}

            builder.add_node(DRAFT_NODE, draft_node)
            builder.add_node(EXECUTE_TOOL_NODE, execute_tool_node)
            builder.add_node(REVISE_NODE, revise_node)
            builder.add_edge(START, DRAFT_NODE)
            builder.add_edge(DRAFT_NODE, EXECUTE_TOOL_NODE)
            builder.add_edge(EXECUTE_TOOL_NODE, REVISE_NODE)
            builder.add_conditional_edges(
                REVISE_NODE,
                lambda state: self._should_continue(state["messages"]),
            )
            return builder.compile()
        except Exception as e:
            logger.error(e)
            return None

    def _execute_graph(
        self,
        graph: CompiledStateGraph | None = None,
        question: str | None = "",
    ) -> ReflexionAgentResult:
        """Execute Langgraph graph
        Args:
          graph CompiledStateGraph: Langgraph graph
          question str: user question

        Returns
        -------
          answer str | None: string answer parsed from Langgraph graph execution

        """
        if graph is None:
            return None
        if len(question) == 0:
            return None

        events = graph.stream(
            {"messages": [HumanMessage(content=question)]},
            {
                "recursion_limit": self.recursion_limit,
            },
        )
        messages = [HumanMessage(content=question)]
        for i, step in enumerate(events):
            node, output = next(iter(step.items()))
            if isinstance(output, dict) and "messages" in output:
                for msg in output["messages"]:
                    self.agent_logger.log_step_message(i + 1, node, msg)
                    messages.append(msg)
            else:
                self.agent_logger.log_step_message(i + 1, node, output)
                messages.append(output)

        final_result = self._parse_final_result(messages)
        self.agent_logger.log_final_result(final_result)
        return final_result

    def execute(
        self,
        question: str,
        prompt: str | None = None,
    ) -> ReflexionAgentResult:
        """Execute ReflexionAgent. Wrapper for building a graph and executing it,
        returning the final answer.

        Args:
        ----
          question str: user question
          prompt str: user prompt

        Returns:
        -------
          answer str | None: If it executes successfully, an answer to the
            question will be returned, otherwise, it returns None

        """
        if len(question) == 0:
            return None
        graph = self._build_graph(prompt)
        return self._execute_graph(graph, question)
