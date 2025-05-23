"""Sequential Agent implementation using LangGraph for explicit, revisable
sequential thinking.

This module provides the SequentialAgent class that implements an agentic
conversation framework using LangGraph with a planner-executor-controller
architecture for step-by-step execution of tasks.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class Step(TypedDict):
    """Single step in the execution plan.

    Attributes
    ----------
        task: Natural-language instruction for the step
        expected: What the model expects to obtain from executing this step
        tool: Name of bound tool to call, if any
        status: Current status of the step
        revisions: List of revisions to the various steps

    """

    task: str
    expected: str
    tool: str | None
    status: Literal["pending", "done", "skipped"]
    revisions: list[str] | None


class AgentState(TypedDict):
    """State schema for the sequential agent.

    Attributes
    ----------
        messages: Running transcript of the conversation and tool traces
        plan: Mutable list of steps to execute

    """

    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[Step] | None
    current_query: str | None


class SequentialAgent:
    """Sequential Agent implementation using LangGraph.

    This class provides an agentic conversation framework that performs
    explicit, revisable sequential thinking by planning steps and executing
    them one by one with the ability to revise the plan based on execution
    results.
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        *,
        tools: Sequence[BaseTool] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SequentialAgent.

        Args:
        ----
            model_name: Name of the model to use
            model_provider: Provider of the model (e.g., 'openai', 'anthropic')
            tools: Optional sequence of tools to make available
            llm_kwargs: Optional additional keyword arguments for LLM
                initialization

        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools = list(tools) if tools else []

        # Initialize the LLM using langchain's init_chat_model
        self.llm = init_chat_model(model=model_name, model_provider=model_provider, **(llm_kwargs or {}))

        # Create tool node if tools are available
        self.tool_node = ToolNode(self.tools) if self.tools else None

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        """Build the LangGraph state graph."""
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", self._planner)
        workflow.add_node("executor", self._executor)
        workflow.add_node("controller", self._controller)

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "controller")
        workflow.add_edge("executor", "planner")  # Allow dynamic re-planning

        # Add conditional edge from controller
        workflow.add_conditional_edges("controller", self._route_from_controller, {"executor": "executor", "END": END})

        # Compile the graph
        return workflow.compile()

    def _planner(self, state: AgentState) -> dict[str, Any]:
        """Plan the next steps or update existing plan.

        If state.plan is empty or every step is done, produce a fresh ordered
        plan. If some steps remain, optionally update the remaining portion
        of the plan.
        """
        # Check if we need a fresh plan
        if not state["plan"] or all(step["status"] == "done" for step in state["plan"]):
            # Create a fresh plan for the user query
            messages = state["messages"]

            # Get the user query from the current query or the last human message
            user_query = (
                state.get("current_query")
                if state.get("current_query")
                else next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")
            )

            available_tools = [tool.name for tool in self.tools] if self.tools else "None"

            plan_prompt = f"""
You are a helpful planning assistant. Given the following user query, create a 
detailed step-by-step plan to accomplish the task.

User Query: {user_query}

Available Tools: {available_tools}

Create a plan with steps that are:
1. Atomic and specific
2. Executable with available tools or direct reasoning
3. Ordered logically

For each step, specify:
- task: A clear description of what to do
- expected: What you expect to get from this step
- tool: The name of the tool to use, or null for direct reasoning
- status: Always set to "pending"

Return your plan as a JSON list of steps in the following format:
[
    {{
        "task": "description of task",
        "expected": "what you expect to achieve",
        "tool": "tool_name_or_null",
        "status": "pending"
    }}
]

Only return the JSON, no additional text.
"""

            response = self.llm.invoke([HumanMessage(content=plan_prompt)])

            try:
                plan_text = response.content.strip()
                # Extract JSON from response if it's wrapped in other text
                if plan_text.startswith("```"):
                    plan_text = plan_text.split("```")[1]
                    plan_text = plan_text.removeprefix("json")
                plan_text = plan_text.strip()
                new_plan = json.loads(plan_text)

                # Validate plan structure
                validated_plan = []
                for step_data in new_plan:
                    step: Step = {
                        "task": str(step_data.get("task", "")),
                        "expected": str(step_data.get("expected", "")),
                        "tool": step_data.get("tool") if step_data.get("tool") != "null" else None,
                        "status": "pending",
                    }
                    validated_plan.append(step)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse plan: %s", e)
                # Fallback to a simple single-step plan
                validated_plan = [
                    {
                        "task": f"Address the user query: {user_query}",
                        "expected": "A comprehensive response to the user's question",
                        "tool": None,
                        "status": "pending",
                    }
                ]

            return {"plan": validated_plan}

        # Optionally update remaining steps based on new information
        # For now, just keep the existing plan
        return {}

    def _executor(self, state: AgentState) -> dict[str, Any]:
        """Execute the next pending step.

        Pop the next pending step and either call tool_node or call LLM
        directly. Mark the step done and append result messages.
        """
        # Find the next pending step
        plan = state["plan"][:]  # Copy to avoid mutation
        next_step_index = None

        for i, step in enumerate(plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        if next_step_index is None:
            # No pending steps
            return {}

        current_step = plan[next_step_index]

        # Execute the step with comprehensive prompt
        # Collect information about completed steps for context
        completed_steps = [step for step in plan if step["status"] == "done"]
        completed_steps_summary = ""
        if completed_steps:
            completed_steps_summary = "\n".join(
                [f"- {step['task']} (Expected: {step['expected']})" for step in completed_steps]
            )

        # Get original query for context
        original_query = state.get("current_query", "")
        if not original_query:
            # Fallback to first human message if current_query not available
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break

        # Include suggested tool information in the prompt
        tool_info = ""
        if current_step["tool"]:
            tool_info = f"\nSUGGESTED TOOL: {current_step['tool']} - Use this tool if appropriate for the task."

        task_prompt = f"""
You are executing a specific step in a larger plan to address the original question.

ORIGINAL QUESTION: {original_query}

COMPLETED STEPS SO FAR:
{completed_steps_summary if completed_steps_summary else "None yet"}

CURRENT STEP TO EXECUTE:
Task: {current_step["task"]}
Expected Outcome: {current_step["expected"]}{tool_info}

Based on the original question, progress made so far, and the conversation context, provide a focused response for this specific step that builds upon previous work.
"""

        # Create LLM with tools bound if available
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        # Get response from LLM (may include tool calls)
        response = llm_with_tools.invoke(state["messages"] + [HumanMessage(content=task_prompt)])

        # Handle tool execution based on suggested tool
        if current_step["tool"] and self.tool_node:
            # Use suggested tool directly - check if response contains tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Process tool calls through ToolNode
                try:
                    tool_result = self.tool_node.invoke({"messages": state["messages"] + [response]})
                    # Handle tool results - ensure we get ToolMessage objects
                    if isinstance(tool_result, dict) and "messages" in tool_result:
                        # Use the messages directly if they're already ToolMessage objects, or convert them
                        result_messages = []
                        for i, msg in enumerate(tool_result["messages"]):
                            if isinstance(msg, ToolMessage):
                                result_messages.append(msg)
                            else:
                                # Convert to ToolMessage if not already
                                result_messages.append(
                                    ToolMessage(
                                        content=str(msg.content) if hasattr(msg, "content") else str(msg),
                                        tool_call_id="step_" + str(next_step_index) + "_" + str(i),
                                    )
                                )
                    else:
                        # Single result - convert to ToolMessage
                        result_messages = [
                            ToolMessage(content=str(tool_result), tool_call_id="step_" + str(next_step_index))
                        ]
                except Exception:
                    logger.exception("Tool execution failed")
                    # Return tool error message for compatibility
                    result_messages = [
                        ToolMessage(
                            content="Tool execution failed - see logs for details",
                            tool_call_id="step_" + str(next_step_index),
                        )
                    ]
            else:
                # LLM didn't make tool calls even though tool was suggested - return LLM response
                result_messages = [response]
        else:
            # Direct response without tool suggestion
            result_messages = [response]

        # Mark step as done
        plan[next_step_index] = {**current_step, "status": "done"}

        return {"messages": result_messages, "plan": plan}

    def _controller(self, state: AgentState) -> dict[str, Any]:
        """Control the flow based on remaining pending steps.

        Args:
        ----
            state: Current conversation state

        Returns:
        -------
            Empty dict (routing decision is handled by _route_from_controller)

        """
        # This method doesn't update state, just returns empty dict
        # Actual routing decision is made by _route_from_controller
        return {}

    def _route_from_controller(self, state: AgentState) -> str:
        """Determine the next node from controller.

        Returns
        -------
            "executor" if there are pending steps, "END" otherwise

        """
        # Check if any steps are still pending
        has_pending = any(step["status"] == "pending" for step in state["plan"])

        if has_pending:
            return "executor"
        return "END"

    def invoke(
        self,
        query: str | None = None,
        state: AgentState | None = None,
        config: dict[str, Any] | None = None,
    ) -> AgentState:
        """Invoke the sequential agent with a query.

        Args:
        ----
            query: User query string
            state: Optional initial state (coming from another graph)
            config: Optional runtime configuration

        Returns:
        -------
            Final updated AgentState with full conversation and tool traces

        """
        if (query is None) and (state is None):
            raise ValueError("query or state is required")

        if state is None:
            # Initialize the state
            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "plan": [],
                "current_query": query,
            }
        else:
            initial_state = state

        # Run the graph
        return self.graph.invoke(initial_state, config=config)

    def bind_tools(self, tools: Sequence[BaseTool]) -> None:
        """Bind additional tools to the agent.

        Args:
        ----
            tools: Sequence of tools to add

        """
        self.tools.extend(tools)
        self.tool_node = ToolNode(self.tools) if self.tools else None
        # Rebuild the graph with new tools
        self.graph = self._build_graph()
