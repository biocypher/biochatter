"""LangGraph-based agentic conversation implementation.

This module provides the LangGraphConversation class that implements an agentic
conversation framework using LangGraph and LangChain, with a router/planner
architecture for handling both direct responses and multi-step tool execution.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from biochatter.llm_connect.sequential_agent import SequentialAgent

from .sequential_agent import AgentState as ConversationState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool


logger = logging.getLogger(__name__)


class LangGraphConversation:
    """LangGraph-based agentic conversation implementation.

    This class provides an agentic conversation framework that can either
    respond directly to user queries or execute multi-step plans using
    available tools. The conversation uses a router/planner architecture to
    decide between direct responses and planned tool execution.
    """

    # =============================================================================
    # CORE LIFECYCLE METHODS
    # =============================================================================

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
        sqlite_db_path: str | None = None,
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
            checkpointer_type: Type of checkpointer to use ('memory' or 'sqlite')
            sqlite_db_path: Path to SQLite database file (only used when checkpointer_type='sqlite')
            llm_kwargs: Optional additional keyword arguments for LLM
                initialization

        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools = list(tools) if tools else []
        self.checkpointer_type = checkpointer_type
        self.sqlite_db_path = sqlite_db_path

        # Setup memory and config using auxiliary method
        self.config, self.memory = self._setup_memory_management(
            save_history=save_history,
            thread_id=thread_id,
            config=config,
            checkpointer_type=checkpointer_type,
            sqlite_db_path=sqlite_db_path,
        )

        # Initialize the LLM using langchain's init_chat_model
        self.llm = init_chat_model(model=model_name, model_provider=model_provider, **(llm_kwargs or {}))

        # Initialize the SequentialAgent for planned execution
        self.sequential_agent = SequentialAgent(
            model_name=model_name, model_provider=model_provider, tools=self.tools, llm_kwargs=llm_kwargs
        )

        # Build the graph immediately
        self.graph = self._build_graph()

    def _setup_memory_management(
        self,
        save_history: bool,
        thread_id: int,
        config: dict | None,
        checkpointer_type: Literal["memory", "sqlite"],
        sqlite_db_path: str | None,
    ) -> tuple[dict, MemorySaver | SqliteSaver | None]:
        """Setup memory management configuration and checkpointer.

        Args:
        ----
            save_history: Whether to save conversation history
            thread_id: Default thread ID to use if not provided
            config: Optional configuration mapping
            checkpointer_type: Type of checkpointer to use
            sqlite_db_path: Path to SQLite database file

        Returns:
        -------
            Tuple of (config_dict, memory_checkpointer)

        Raises:
        ------
            ValueError: If configuration is invalid

        """
        if not save_history:
            return {}, None

        # Setup config with thread_id
        if config is None:
            config = {"configurable": {"thread_id": thread_id}}
        # Ensure thread_id is in config
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": thread_id}
        elif "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = thread_id

        # Create appropriate checkpointer based on type
        if checkpointer_type == "memory":
            memory = MemorySaver()
        elif checkpointer_type == "sqlite":
            if sqlite_db_path is None:
                # Use default path if none provided
                sqlite_db_path = "conversation_checkpoints.db"

            # For SQLite, we need to enter the context manager and store the active saver
            sqlite_context = SqliteSaver.from_conn_string(sqlite_db_path)
            memory = sqlite_context.__enter__()
            # Store the context manager so we can properly clean it up if needed
            self._sqlite_context = sqlite_context
            # Update the instance variable with the actual path used
            self.sqlite_db_path = sqlite_db_path
        else:
            raise ValueError(f"Unsupported checkpointer_type: {checkpointer_type}")

        return config, memory

    def __del__(self) -> None:
        """Cleanup when the conversation is destroyed."""
        self._cleanup_sqlite_context()

    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================

    def bind_tools(self, tools: Sequence[BaseTool]) -> None:
        """Extend the persistent tool list with additional tools.

        Args:
        ----
            tools: Sequence of tools to add to the persistent tool list

        """
        self.tools.extend(tools)
        # Update the sequential agent with new tools
        self.sequential_agent.bind_tools(tools)
        # Rebuild the graph to incorporate new tools
        self.graph = self._build_graph()

    def invoke(self, query: str, *, tools: Sequence[BaseTool] | None = None, config: dict | None = None) -> str:
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
            # Update sequential agent with new tools
            self.sequential_agent.bind_tools(tools)
            # Rebuild graph with new tools
            self.graph = self._build_graph()

        try:
            # Initialize conversation state
            initial_state: ConversationState = {
                "messages": [HumanMessage(content=query)],
                "plan": None,
                "current_query": query,
            }

            # Create runtime config
            if config:
                self.config = config
            elif self.config and not config:
                pass
            else:
                self.config = {}

            # Run the graph
            result = self.graph.invoke(initial_state, config=self.config)

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
            # Restore sequential agent tools if we had ad-hoc tools
            if tools:
                # Reset sequential agent to original tools
                self.sequential_agent = SequentialAgent(
                    model_name=self.model_name, model_provider=self.model_provider, tools=self.tools, llm_kwargs={}
                )
                # Rebuild graph with restored tools
                self.graph = self._build_graph()

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

        # Clean up existing SQLite context if it exists
        self._cleanup_sqlite_context()

        # Extract current thread_id from config for reuse
        current_thread_id = self.config.get("configurable", {}).get("thread_id", 0)

        # Create a new checkpointer using the auxiliary method
        self.config, self.memory = self._setup_memory_management(
            save_history=True,  # We know save_history is True if we have memory
            thread_id=current_thread_id,
            config=None,  # Let the method create new config
            checkpointer_type=self.checkpointer_type,
            sqlite_db_path=self.sqlite_db_path,
        )

        # Rebuild the graph with the new checkpointer
        self.graph = self._build_graph()

    # =============================================================================
    # GRAPH BUILDING METHODS
    # =============================================================================

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
        workflow.add_node("sequential_execution", self._sequential_execution)

        # Create ToolNode with current tools
        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))
        else:
            # Create empty ToolNode if no tools available
            workflow.add_node("tools", ToolNode([]))

        workflow.add_node("reason_about_tools", self._reason_about_tools)

        # Add edges - router now uses Command objects for direct routing
        workflow.add_edge(START, "router")
        # No conditional edges needed from router - Command objects handle the routing

        workflow.add_edge("direct_response", END)
        workflow.add_edge("sequential_execution", END)

        # Conditional edges from llm_with_tools
        workflow.add_conditional_edges("llm_with_tools", self._should_call_tools, {"tools": "tools", "end": END})

        workflow.add_edge("tools", "reason_about_tools")
        workflow.add_edge("reason_about_tools", END)

        # Compile with optional checkpointer
        if self.memory:
            return workflow.compile(checkpointer=self.memory)
        return workflow.compile()

    # =============================================================================
    # CORE WORKFLOW METHODS
    # =============================================================================

    def _route_or_plan(
        self, state: ConversationState
    ) -> Command[Literal["direct_response", "llm_with_tools", "sequential_execution"]]:
        """Router/planner node that decides between direct response and planning.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Command object with routing decision and state update

        """
        if not state["messages"]:
            return Command(goto="direct_response")

        # Create routing prompt using dedicated method
        routing_prompt = self._generate_routing_prompt(state["current_query"], self.tools)

        # Get routing decision from LLM
        routing_message = HumanMessage(content=routing_prompt)
        response = self.llm.invoke([routing_message])

        response_content = response.content if hasattr(response, "content") else str(response)

        # Determine the execution path and route using Command
        response_clean = response_content.strip().upper()
        if response_clean == "DIRECT":
            return Command(goto="direct_response")
        if response_clean == "TOOL":
            return Command(goto="llm_with_tools")
        if response_clean == "PLAN":
            return Command(goto="sequential_execution")
        # Fallback: treat as plan with the actual response content
        return Command(goto="sequential_execution")

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
        reasoning_prompt = self._generate_reasoning_prompt()

        messages_with_prompt = state["messages"] + [HumanMessage(content=reasoning_prompt)]
        response = self.llm.invoke(messages_with_prompt)

        return {"messages": [response]}

    def _sequential_execution(self, state: ConversationState) -> dict[str, Any]:
        """Execute sequential planning using the SequentialAgent.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Updated state with sequential execution results

        """
        if not state["messages"]:
            return {"messages": []}

        # Use the sequential agent to process the query
        try:
            agent_result = self.sequential_agent.invoke(state=state, config=self.config)

            # Extract the final response from the agent result
            if agent_result["messages"]:
                # Get all messages except the original human message (since it's already in state)
                new_messages = agent_result["messages"][1:]  # Skip the original HumanMessage
                return {"messages": new_messages}
            # Fallback response
            fallback_message = BaseMessage(content="I was unable to complete the planned execution.", type="ai")
            return {"messages": [fallback_message]}

        except Exception as e:
            logger.error("Sequential execution failed: %s", e)
            error_message = BaseMessage(content=f"I encountered an error during planned execution: {e}", type="ai")
            return {"messages": [error_message]}

    # =============================================================================
    # PROMPT GENERATION METHODS
    # =============================================================================

    def _generate_routing_prompt(self, current_query: str, tools: list[BaseTool]) -> str:
        """Generate the routing prompt for deciding between execution strategies.

        Args:
        ----
            current_query: The user's current query
            tools: List of available tools

        Returns:
        -------
            Formatted routing prompt string

        """
        tools_description = self._get_tools_description(tools)

        return f"""<instructions>
Decide how to handle the user message based on these three options:
</instructions>

<query>
{current_query}
</query>

<tools>
The tools available are:
{tools_description}
</tools>

<options>
Output exactly one of:
- "DIRECT" if it can be answered directly without any tools
- "TOOL" if it can be answered with a single tool call (prefer using tools over direct response when possible)
- "PLAN" if it can be answered with a multi-step plan
</options>

<examples>
- "What is the capital of France?" → output "DIRECT"
- "Search for information about protein folding" → output "TOOL"
- "Compare protein sequences A and B, then analyze their structure" → output "PLAN"
</examples>"""

    def _generate_reasoning_prompt(self) -> str:
        """Generate the reasoning prompt for analyzing tool results.

        Returns
        -------
            Formatted reasoning prompt string

        """
        return """<instructions>
Based on the tool results above, please provide a comprehensive response to the original question.
</instructions>

<task>
Analyze the tool execution results and synthesize them into a coherent, helpful response that directly addresses the user's original query.
</task>"""

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

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

    # =============================================================================
    # CLEANUP METHODS
    # =============================================================================

    def _cleanup_sqlite_context(self) -> None:
        """Clean up the SQLite context manager if it exists."""
        if hasattr(self, "_sqlite_context") and self._sqlite_context is not None:
            try:
                self._sqlite_context.__exit__(None, None, None)
            except Exception:
                # Ignore cleanup errors
                pass
            finally:
                self._sqlite_context = None
