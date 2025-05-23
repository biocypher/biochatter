"""LangGraph-based agentic conversation implementation.

This module provides the LangGraphConversation class that implements an agentic
conversation framework using LangGraph and LangChain, with a router/planner
architecture for handling both direct responses and multi-step tool execution.
"""

from __future__ import annotations

import logging
import operator
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool


logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State model for the LangGraph conversation.

    Attributes
    ----------
        messages: List of conversation messages
        plan: Optional plan string for multi-step execution
        intermediate_steps: List of intermediate tool execution results

    """

    messages: Annotated[list[BaseMessage], operator.add]
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
        save_history: bool = True,
        thread_id: int = 0,
        config: dict | None = None,
        checkpointer_type: Literal["memory", "sqlite"] = "memory",
        llm_kwargs: Mapping | None = None,
    ) -> None:
        """Initialize the LangGraphConversation.

        Args:
        ----
            model_name: Name of the model to use
            model_provider: Provider of the model (e.g., 'openai', 'anthropic')
            tools: Optional sequence of tools to make available
            save_history: Whether to save history
            thread_id: Default thread ID to use if not provided
            config: Optional configuration mapping
            checkpointer_type: Type of checkpointer to use
            llm_kwargs: Optional additional keyword arguments for LLM
                initialization

        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools = list(tools) if tools else []

        # case in which no config but save_history is True
        if save_history and config is None:
            config = {"configurable": {"thread_id": thread_id}}
            self.memory = MemorySaver()
        elif save_history and config is not None:
            # check if thread_id is in config
            assert "thread_id" in config["configurable"], "thread_id must be in config if save_history is True"
            self.memory = MemorySaver()
        elif not save_history:
            config = {}
            self.memory = None
        else:
            raise ValueError("Invalid configuration")

        self.config = config or {}

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
            # Rebuild graph with new tools
            self.graph = self._build_graph()

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
            # Rebuild graph with restored tools if we had ad-hoc tools
            if tools:
                self.graph = self._build_graph()

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
        workflow.add_node("direct_response", self._direct_response)
        workflow.add_node("llm_with_tools", self._llm_with_tools)

        # Create ToolNode with current tools
        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))
        else:
            # Create empty ToolNode if no tools available
            workflow.add_node("tools", ToolNode([]))

        workflow.add_node("reason_about_tools", self._reason_about_tools)

        # Add edges
        workflow.add_edge(START, "router")

        # Conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "direct": "direct_response",
                "tool": "llm_with_tools",
                "plan": "llm_with_tools",  # Handle plan similarly to tool
            },
        )

        workflow.add_edge("direct_response", END)

        # Conditional edges from llm_with_tools
        workflow.add_conditional_edges("llm_with_tools", self._should_call_tools, {"tools": "tools", "end": END})

        workflow.add_edge("tools", "reason_about_tools")
        workflow.add_edge("reason_about_tools", END)

        # Compile with optional checkpointer
        if self.memory:
            return workflow.compile(checkpointer=self.memory)
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
        routing_prompt = f"""Decide how to handle the user message based on these three options:

User message: {user_query}

The tools available are:
{self._get_tools_description(self.tools)}

Output exactly one of:
- "DIRECT" if it can be answered directly without any tools
- "TOOL" if it can be answered with a single tool call (prefer using tools over direct response when possible)
- A numbered plan if multiple tool calls are needed

For example:
- "What is the capital of France?" → output "DIRECT"
- "Search for information about protein folding" → output "TOOL"
- "Compare protein sequences A and B, then analyze their structure" → output:
  1. Retrieve protein sequence A
  2. Retrieve protein sequence B
  3. Compare sequences
  4. Analyze structural implications
"""

        # Get routing decision from LLM
        routing_message = HumanMessage(content=routing_prompt)
        response = self.llm.invoke([routing_message])

        response_content = response.content if hasattr(response, "content") else str(response)

        # Determine the execution path
        response_clean = response_content.strip().upper()
        if response_clean == "DIRECT":
            return {"plan": None}
        if response_clean == "TOOL":
            return {"plan": "TOOL"}
        return {"plan": response_content.strip()}

    def _route_decision(self, state: ConversationState) -> str:
        """Determine the routing decision based on the plan.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Route decision: "direct", "tool", or "plan"

        """
        if state["plan"] is None:
            return "direct"
        if state["plan"] == "TOOL":
            return "tool"
        return "plan"

    def _direct_response(self, state: ConversationState) -> dict[str, Any]:
        """Handle direct response without tools.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with direct response

        """
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def _llm_with_tools(self, state: ConversationState) -> dict[str, Any]:
        """Generate response with tools available for calling.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with LLM response (potentially with tool calls)

        """
        if state["plan"] and state["plan"] != "TOOL":
            # Handle planned execution
            plan_steps = self._parse_plan(state["plan"])
            if plan_steps:
                # Create a message asking the LLM to execute the first step
                plan_message = HumanMessage(
                    content=f"Please execute this step: {plan_steps[0]}. Use the available tools if needed."
                )
                messages = state["messages"] + [plan_message]
            else:
                messages = state["messages"]
        else:
            # Handle direct tool usage
            messages = state["messages"]

        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = self.llm.invoke(messages)

        return {"messages": [response]}

    def _should_call_tools(self, state: ConversationState) -> str:
        """Determine if tools should be called based on the last message.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            "tools" if tools should be called, "end" otherwise

        """
        if not state["messages"]:
            return "end"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    def _reason_about_tools(self, state: ConversationState) -> dict[str, Any]:
        """Generate reasoning about tool results.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with reasoning response

        """
        # The ToolNode will have added ToolMessage(s) to the messages
        # Now we ask the LLM to reason about the results
        reasoning_prompt = HumanMessage(
            content="Based on the tool results above, please provide a comprehensive response to the original question."
        )

        messages_with_prompt = state["messages"] + [reasoning_prompt]
        response = self.llm.invoke(messages_with_prompt)

        return {"messages": [response]}

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

    def _get_tools_description(self, tools: list[BaseTool]) -> str:
        """Get a formatted description of available tools.

        Args:
        ----
            tools: List of available tools

        Returns:
        -------
            Formatted string with tool names and descriptions

        """
        if not tools:
            return "No tools available."

        tool_descriptions = []
        for tool in tools:
            tool_name = tool.name
            tool_desc = getattr(tool, "description", "No description available.")
            tool_descriptions.append(f"- {tool_name}: {tool_desc}")

        return "\n".join(tool_descriptions)

    def reset(self) -> None:
        """Reset the conversation memory by creating a new checkpointer.

        This method deletes the current checkpointer and creates a fresh one,
        then rebuilds the graph with the new checkpointer. This ensures a
        completely clean slate for the conversation.
        Only works when save_history is enabled.

        Raises
        ------
            ValueError: If no checkpointer is configured

        """
        if not hasattr(self, "memory") or self.memory is None:
            raise ValueError("No checkpointer configured. Cannot reset conversation memory.")

        # Create a new MemorySaver instance
        self.memory = MemorySaver()

        # Rebuild the graph with the new checkpointer
        self.graph = self._build_graph()
