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
from langgraph.types import Command

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
        needs_replanning: Flag indicating if replanning was requested by a revision

    """

    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[Step] | None
    current_query: str | None
    needs_replanning: bool | None


class SequentialAgent:
    """Sequential Agent implementation using LangGraph.

    This class provides an agentic conversation framework that performs
    explicit, revisable sequential thinking by planning steps and executing
    them one by one with the ability to revise the plan based on execution
    results.
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

    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================

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
                "needs_replanning": False,
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

    # =============================================================================
    # GRAPH BUILDING METHODS
    # =============================================================================

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

    # =============================================================================
    # CORE WORKFLOW METHODS
    # =============================================================================

    def _executor(self, state: AgentState) -> dict[str, Any] | Command[Literal["planner"]]:
        """Execute the next pending step.

        Pop the next pending step and either call tool_node or call LLM
        directly. Mark the step done and append result messages.
        """
        messages_to_add = []

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

        # Generate task execution prompt using dedicated method
        task_prompt = self._generate_task_execution_prompt(
            original_query, completed_steps_summary, current_step, next_step_index, state["plan"]
        )

        # Create LLM with tools bound if available
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        messages_to_add.append(HumanMessage(content=task_prompt))

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
                        messages_to_add.extend(result_messages)
                    else:
                        # Single result - convert to ToolMessage
                        result_messages = [
                            ToolMessage(content=str(tool_result), tool_call_id="step_" + str(next_step_index))
                        ]
                        messages_to_add.extend(result_messages)
                except Exception:
                    logger.exception("Tool execution failed")
                    # Return tool error message for compatibility
                    result_messages = [
                        ToolMessage(
                            content="Tool execution failed - see logs for details",
                            tool_call_id="step_" + str(next_step_index),
                        )
                    ]
                    messages_to_add.extend(result_messages)
            else:
                # LLM didn't make tool calls even though tool was suggested - return LLM response
                result_messages = [response]
                messages_to_add.extend(result_messages)
        else:
            # Direct response without tool suggestion
            result_messages = [response]
            messages_to_add.extend(result_messages)

        # Evaluate results against expectations and generate revisions
        actual_results = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in result_messages])

        revision_prompt = self._generate_revision_prompt(original_query, plan, current_step, actual_results)

        try:
            # messages_to_add.append(HumanMessage(content=revision_prompt))
            revision_response = self.llm.invoke([HumanMessage(content=revision_prompt)])
            revision_text = revision_response.content.strip()

            # Extract JSON from response if it's wrapped in other text
            if revision_text.startswith("```"):
                revision_text = revision_text.split("```")[1]
                revision_text = revision_text.removeprefix("json")
            revision_text = revision_text.strip()

            revisions = json.loads(revision_text) if revision_text else {}

            # Add revision message only if we have valid revision data
            if isinstance(revisions, dict) and "revisions" in revisions and "change_plan" in revisions:
                messages_to_add.append(
                    HumanMessage(
                        content=f"Revision: {revisions['revisions']}\nNeed to change plan: {revisions['change_plan']}"
                    )
                )

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse revision recommendations: %s", e)
            revisions = {}

        # Mark step as done and add revisions
        # Append new revisions to any existing ones
        existing_revisions = current_step.get("revisions", [])
        if isinstance(revisions, dict) and "revisions" in revisions:
            # Extract the revision text from the dict
            revision_text = revisions["revisions"]
            if isinstance(existing_revisions, list):
                combined_revisions = existing_revisions + [revision_text]
            else:
                combined_revisions = [revision_text]
        else:
            # Keep existing revisions if no new ones
            combined_revisions = existing_revisions if isinstance(existing_revisions, list) else []
        plan[next_step_index] = {**current_step, "status": "done", "revisions": combined_revisions}

        # Check if we need to rethink the plan based on revision recommendations
        if isinstance(revisions, dict) and revisions.get("change_plan", False):
            # Use Command to redirect to planner for replanning
            return Command(goto="planner", update={"messages": messages_to_add, "plan": plan, "needs_replanning": True})

        return {"messages": messages_to_add, "plan": plan}

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

    # =============================================================================
    # PLANNING METHODS
    # =============================================================================

    def _planner(self, state: AgentState) -> dict[str, Any]:
        """Plan the next steps or update existing plan.

        Handles three scenarios:
        1. Initial planning: No plan exists
        2. Replanning: Some steps are done, need to replan remaining work
        3. Continue existing: Plan exists with only pending steps
        """
        messages = state["messages"]
        user_query = (
            state.get("current_query")
            if state.get("current_query")
            else next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")
        )
        available_tools = [tool.name for tool in self.tools] if self.tools else "None"

        # Case 1: Initial planning (no plan exists)
        if not state["plan"]:
            return self._create_initial_plan(user_query, available_tools)

        # Case 1.5: All steps completed - do nothing, let controller end the process
        if all(step["status"] == "done" for step in state["plan"]):
            return {}

        # Case 2: Replanning scenario (only if explicitly requested by a revision or if there are completed steps with insights)
        completed_steps = [step for step in state["plan"] if step["status"] == "done"]
        pending_steps = [step for step in state["plan"] if step["status"] == "pending"]

        # Check if replanning is needed based on:
        # 1. Explicit request from revision (needs_replanning flag)
        # 2. Completed steps with revisions (insights gained)
        should_replan = state.get("needs_replanning", False)

        if should_replan and completed_steps and pending_steps:
            result = self._replan_remaining_steps(user_query, available_tools, completed_steps, pending_steps)
            # Reset the replanning flag after replanning
            result["needs_replanning"] = False
            return result

        # Case 3: Continue existing plan (only pending steps remain)
        return {}

    def _create_initial_plan(self, user_query: str, available_tools: str) -> dict[str, Any]:
        """Create a fresh initial plan for the user query."""
        plan_prompt = self._generate_initial_plan_prompt(user_query, available_tools)

        response = self.llm.invoke([HumanMessage(content=plan_prompt)])
        return self._parse_and_validate_plan(response, user_query)

    def _replan_remaining_steps(
        self, user_query: str, available_tools: str, completed_steps: list[Step], pending_steps: list[Step]
    ) -> dict[str, Any]:
        """Replan the remaining steps based on completed work and revisions."""
        # Summarize completed work and key insights
        completed_summary = []
        key_revisions = []

        for step in completed_steps:
            completed_summary.append(f"✓ {step['task']} (Expected: {step['expected']})")
            if step.get("revisions") and isinstance(step["revisions"], list):
                key_revisions.extend(step["revisions"])

        completed_work = "\n".join(completed_summary) if completed_summary else "None completed yet"
        insights = (
            "\n".join([f"- {revision}" for revision in key_revisions]) if key_revisions else "No specific insights yet"
        )

        # Current pending steps for context
        remaining_work = "\n".join([f"• {step['task']} (Expected: {step['expected']})" for step in pending_steps])

        replan_prompt = self._generate_replan_prompt(
            user_query, available_tools, completed_work, insights, remaining_work
        )

        response = self.llm.invoke([HumanMessage(content=replan_prompt)])
        replanned_steps = self._parse_and_validate_plan(response, user_query, is_replan=True)

        # Combine completed steps with new replanned steps
        if "plan" in replanned_steps:
            full_plan = completed_steps + replanned_steps["plan"]
            return {"plan": full_plan}

        # Fallback: keep existing plan if replanning fails
        return {}

    # =============================================================================
    # PROMPT GENERATION METHODS
    # =============================================================================

    def _generate_initial_plan_prompt(self, user_query: str, available_tools: str) -> str:
        """Generate the initial planning prompt.

        Args:
        ----
            user_query: The user's query to plan for
            available_tools: String description of available tools

        Returns:
        -------
            Formatted initial planning prompt string

        """
        return f"""<role>
You are a helpful planning assistant. Given the following user query, create a 
detailed step-by-step plan to accomplish the task.
</role>

<query>
{user_query}
</query>

<tools>
Available Tools: {available_tools}
</tools>

<requirements>
Create a plan with steps that are:
1. Atomic and specific
2. Executable with available tools or direct reasoning
3. Ordered logically
</requirements>

<step_format>
For each step, specify:
- task: A clear description of what to do
- expected: What you expect to get from this step
- tool: The name of the tool to use, or null for direct reasoning
- status: Always set to "pending"
</step_format>

<output_format>
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
</output_format>"""

    def _generate_replan_prompt(
        self, user_query: str, available_tools: str, completed_work: str, insights: str, remaining_work: str
    ) -> str:
        """Generate the replanning prompt.

        Args:
        ----
            user_query: The original user query
            available_tools: String description of available tools
            completed_work: Summary of completed steps
            insights: Key insights from completed work
            remaining_work: Summary of remaining planned work

        Returns:
        -------
            Formatted replanning prompt string

        """
        return f"""<role>
You are a replanning assistant. Based on completed work and insights gained, 
create a revised plan for the remaining tasks.
</role>

<original_query>
{user_query}
</original_query>

<completed_work>
{completed_work}
</completed_work>

<insights>
Key insights and revisions from completed work:
{insights}
</insights>

<remaining_work>
Currently planned remaining work:
{remaining_work}
</remaining_work>

<tools>
Available Tools: {available_tools}
</tools>

<requirements>
Based on the insights from completed work, create a NEW plan for the remaining tasks.
The new plan should:
1. Take into account lessons learned from completed steps
2. Address any issues identified in the revisions
3. Be more effective than the original remaining plan
4. Complete the original user query successfully
</requirements>

<output_format>
Return your revised plan as a JSON list of steps in the following format:
[
    {{
        "task": "description of task",
        "expected": "what you expect to achieve", 
        "tool": "tool_name_or_null",
        "status": "pending"
    }}
]

Only return the JSON, no additional text.
</output_format>"""

    def _generate_task_execution_prompt(
        self,
        original_query: str,
        completed_steps_summary: str,
        current_step: Step,
        next_step_index: int,
        plan: list[Step],
    ) -> str:
        """Generate the task execution prompt.

        Args:
        ----
            original_query: The original user query
            completed_steps_summary: Summary of completed steps
            current_step: The current step to execute
            next_step_index: Index of the current step
            plan: The full plan with all steps

        Returns:
        -------
            Formatted task execution prompt string

        """
        # Include suggested tool information
        tool_info = ""
        if current_step["tool"]:
            tool_info = f"SUGGESTED TOOL: {current_step['tool']} - Use this tool if appropriate for the task."

        # Include previous revisions if they exist for the current step
        revisions_info = ""
        if current_step.get("revisions") and isinstance(current_step["revisions"], list) and current_step["revisions"]:
            revisions_content = "\n".join([f"- {revision}" for revision in current_step["revisions"]])
            revisions_info = f"Previous revisions for this step:\n{revisions_content}"

        return f"""<role>
You are executing a specific step in a larger plan to address the original question.
</role>

<original_question>
{original_query}
</original_question>

<completed_steps>
{completed_steps_summary if completed_steps_summary else "None yet"}
</completed_steps>

<current_step>
Task: {current_step["task"]}
Expected Outcome: {current_step["expected"]}
</current_step>

<tool_suggestion>
{tool_info if tool_info else "No specific tool suggested for this step."}
</tool_suggestion>

<revisions>
{revisions_info if revisions_info else "No previous revisions for this step."}
</revisions>

<instructions>
Based on the original question, progress made so far, and the conversation context, provide a focused response for this specific step that builds upon previous work.
</instructions>"""

    def _generate_revision_prompt(
        self, original_query: str, plan: list[Step], current_step: Step, actual_results: str
    ) -> str:
        """Generate the revision evaluation prompt.

        Args:
        ----
            original_query: The original user query
            plan: The full plan with all steps
            current_step: The step that was just executed
            actual_results: The actual results from step execution

        Returns:
        -------
            Formatted revision evaluation prompt string

        """
        remaining_steps = "\n".join(
            [
                f"- {step['task']} (Expected: {step['expected']}, Tool: {step['tool']}, Status: {step['status']})"
                for step in plan
                if step.get("status") != "done"
            ]
        )

        return f"""<role>
You are evaluating whether a step in a plan was executed successfully. Compare the expected outcome with the actual results and provide revision recommendations if needed.
</role>

<original_question>
{original_query}
</original_question>

<remaining_steps>
Steps to execute:
{remaining_steps}
</remaining_steps>

<executed_step>
Task: {current_step["task"]}
Expected Outcome: {current_step["expected"]}
</executed_step>

<actual_results>
{actual_results}
</actual_results>

<evaluation_criteria>
Please evaluate:
1. Does the actual result meet the expected outcome?
2. What could be improved or done differently?
3. Are there any gaps or missing elements?
</evaluation_criteria>

<output_format>
Return a JSON dict with the following keys:
- "revisions": A succinct mono line revising the obtained result
- "change_plan": either True or False, indicating if the plan should be revised
Only return the JSON, no additional text.
</output_format>"""

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _parse_and_validate_plan(self, response, user_query: str, is_replan: bool = False) -> dict[str, Any]:
        """Parse and validate the LLM's plan response."""
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
                    "revisions": [],
                }
                validated_plan.append(step)

            return {"plan": validated_plan}

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse %s: %s", "replan" if is_replan else "plan", e)
            # Fallback to a simple single-step plan
            action = "replan and address" if is_replan else "address"
            validated_plan = [
                {
                    "task": f"Re-evaluate and {action} the user query: {user_query}",
                    "expected": "A comprehensive response addressing the user's question",
                    "tool": None,
                    "status": "pending",
                    "revisions": [],
                }
            ]
            return {"plan": validated_plan}
