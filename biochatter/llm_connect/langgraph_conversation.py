"""LangGraph-based agentic conversation implementation.

This module provides the LangGraphConversation class that implements an agentic
conversation framework using LangGraph and LangChain, with a router/planner
architecture for handling both direct responses and multi-step tool execution.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State model for the LangGraph conversation.

    Attributes
    ----------
        messages: List of conversation messages
        plan: Optional plan string for multi-step execution
        intermediate_steps: List of intermediate tool execution results

    """

    messages: list[BaseMessage]
    plan: str | None
    intermediate_steps: list[tuple[str, Any]]


class LangGraphConversation:
    """LangGraph-based agentic conversation implementation.

    This class provides an agentic conversation framework that can either
    respond directly to user queries or execute multi-step plans using
    available tools. The conversation uses a router/planner architecture to
    decide between direct responses and planned tool execution.
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        *,
        tools: Sequence[BaseTool] | None = None,
        config: Mapping | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        llm_kwargs: Mapping | None = None,
    ) -> None:
        """Initialize the LangGraphConversation.

        Args:
        ----
            model_name: Name of the model to use
            model_provider: Provider of the model (e.g., 'openai', 'anthropic')
            tools: Optional sequence of tools to make available
            config: Optional configuration mapping
            checkpointer: Optional checkpoint manager for persistence
            llm_kwargs: Optional additional keyword arguments for LLM
                initialization

        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools = list(tools) if tools else []
        self.config = config or {}
        self.checkpointer = checkpointer

        # Initialize the LLM using langchain's init_chat_model
        self.llm = init_chat_model(model=model_name, model_provider=model_provider, **(llm_kwargs or {}))

        # Build the graph immediately
        self.graph = self._build_graph()

    def bind_tools(self, tools: Sequence[BaseTool]) -> None:
        """Extend the persistent tool list with additional tools.

        Args:
        ----
            tools: Sequence of tools to add to the persistent tool list

        """
        self.tools.extend(tools)
        # Rebuild the graph to incorporate new tools
        self.graph = self._build_graph()

    def invoke(self, query: str, *, tools: Sequence[BaseTool] | None = None) -> str:
        """Invoke the conversation with a query and optional ad-hoc tools.

        Args:
        ----
            query: User query string
            tools: Optional sequence of ad-hoc tools for this specific query

        Returns:
        -------
            Final assistant reply as a string

        """
        # Store current tools for this invocation
        original_tools = self.tools.copy()

        # Temporarily add ad-hoc tools
        if tools:
            self.tools.extend(tools)

        try:
            # Initialize conversation state
            initial_state: ConversationState = {
                "messages": [HumanMessage(content=query)],
                "plan": None,
                "intermediate_steps": [],
            }

            # Create runtime config
            runtime_config = {}
            if self.config:
                runtime_config.update(self.config)

            # Run the graph
            result = self.graph.invoke(initial_state, runtime_config)

            # Extract the final assistant message
            if result["messages"]:
                final_message = result["messages"][-1]
                if hasattr(final_message, "content"):
                    return final_message.content
                return str(final_message)

            return "No response generated."

        finally:
            # Restore original tools
            self.tools = original_tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns
        -------
            Compiled LangGraph state graph

        """
        # Define the state graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("router", self._route_or_plan)
        workflow.add_node("execute", self._maybe_execute)

        # Add edges
        workflow.add_edge(START, "router")
        workflow.add_edge("router", "execute")
        workflow.add_edge("execute", END)

        # Compile with optional checkpointer
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    def _route_or_plan(self, state: ConversationState) -> dict[str, Any]:
        """Router/planner node that decides between direct response and planning.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with plan decision

        """
        if not state["messages"]:
            return {"plan": None}

        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, "content") else str(last_message)

        # Create routing prompt
        routing_prompt = f"""Decide if the last user message can be answered directly or requires multiple tool calls.

User message: {user_query}

Output exactly "DIRECT" if it can be answered directly, or a numbered plan if multiple tool calls are needed.

For example:
- If asking "What is the capital of France?" → output "DIRECT"
- If asking "Compare protein sequences A and B, then analyze their structure" → output:
  1. Retrieve protein sequence A
  2. Retrieve protein sequence B
  3. Compare sequences
  4. Analyze structural implications
"""

        # Get routing decision from LLM
        routing_message = HumanMessage(content=routing_prompt)
        response = self.llm.invoke([routing_message])

        response_content = response.content if hasattr(response, "content") else str(response)

        # Determine if direct response or planning is needed
        if response_content.strip().upper() == "DIRECT":
            return {"plan": None}
        return {"plan": response_content.strip()}

    def _maybe_execute(self, state: ConversationState) -> dict[str, Any]:
        """Handle either direct response or planned execution.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with final response

        """
        # Use the tools stored on the instance
        available_tools = self.tools

        new_messages = list(state["messages"])
        new_intermediate_steps = list(state["intermediate_steps"])

        if state["plan"] is None:
            # Direct response path - bind tools and respond directly
            if available_tools:
                llm_with_tools = self.llm.bind_tools(available_tools)
                response = llm_with_tools.invoke(state["messages"])
            else:
                response = self.llm.invoke(state["messages"])

            new_messages.append(response)
        else:
            # Planned execution path - execute each step in the plan
            plan_steps = self._parse_plan(state["plan"])

            for step in plan_steps:
                # Find appropriate tool for this step
                tool_name, tool_args = self._resolve_tool_for_step(step, available_tools)

                if tool_name and tool_args is not None:
                    # Execute the tool
                    tool = next((t for t in available_tools if t.name == tool_name), None)
                    if tool:
                        try:
                            result = tool.invoke(tool_args)
                            new_intermediate_steps.append((tool_name, result))
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Tool execution failed for %s: %s", tool_name, e)
                            new_intermediate_steps.append((tool_name, f"Error: {e}"))

            # Synthesize final answer using all intermediate results
            synthesis_prompt = self._create_synthesis_prompt(
                original_query=state["messages"][0].content if state["messages"] else "",
                intermediate_steps=new_intermediate_steps,
            )

            synthesis_message = HumanMessage(content=synthesis_prompt)
            final_response = self.llm.invoke([synthesis_message])
            new_messages.append(final_response)

        return {"messages": new_messages, "intermediate_steps": new_intermediate_steps}

    def _parse_plan(self, plan: str) -> list[str]:
        """Parse a numbered plan into individual steps.

        Args:
        ----
            plan: Plan string with numbered steps

        Returns:
        -------
            List of individual step strings

        """
        if not plan:
            return []

        # Split by numbered items (1., 2., etc.)
        steps = re.split(r"\d+\.", plan)
        # Filter out empty strings and strip whitespace
        return [step.strip() for step in steps if step.strip()]

    def _resolve_tool_for_step(
        self, step: str, available_tools: Sequence[BaseTool]
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Resolve which tool and arguments to use for a given step.

        Args:
        ----
            step: Description of the step to execute
            available_tools: Available tools to choose from

        Returns:
        -------
            Tuple of (tool_name, tool_arguments) or (None, None) if no tool
            found

        """
        if not available_tools:
            return None, None

        # Simple heuristic: use first tool that might be relevant
        # In a real implementation, this could be more sophisticated
        for tool in available_tools:
            tool_name = tool.name.lower()
            step_lower = step.lower()

            # Basic keyword matching - check if tool name is in the step
            if tool_name in step_lower:
                # Return basic arguments - more sophisticated in practice
                return tool.name, {"input": step}

        return None, None

    def _create_synthesis_prompt(self, original_query: str, intermediate_steps: list[tuple[str, Any]]) -> str:
        """Create a prompt for synthesizing the final answer from intermediate
        results.

        Args:
        ----
            original_query: The original user query
            intermediate_steps: List of (tool_name, result) tuples

        Returns:
        -------
            Synthesis prompt string

        """
        steps_text = "\n".join([f"- {tool_name}: {result}" for tool_name, result in intermediate_steps])

        return f"""Based on the following intermediate results, provide a comprehensive answer to the original question.

Original question: {original_query}

Intermediate results:
{steps_text}

Please synthesize this information into a clear, helpful response that directly addresses the original question."""
