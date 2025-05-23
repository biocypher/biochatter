"""Tests for the Sequential Agent module."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from biochatter.llm_connect.sequential_agent import (
    AgentState,
    SequentialAgent,
    Step,
)


# Mock tools for testing
@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers together."""
    return first_int + second_int


@tool
def echo(text: str) -> str:
    """Echo the input text back."""
    return f"Echo: {text}"


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"


class TestStep:
    """Test the Step TypedDict."""

    def test_step_creation(self):
        """Test creating a Step with all required fields."""
        step: Step = {"task": "Test task", "expected": "Test expectation", "tool": "test_tool", "status": "pending"}
        assert step["task"] == "Test task"
        assert step["expected"] == "Test expectation"
        assert step["tool"] == "test_tool"
        assert step["status"] == "pending"

    def test_step_with_none_tool(self):
        """Test creating a Step with None tool."""
        step: Step = {"task": "Direct reasoning task", "expected": "Direct response", "tool": None, "status": "pending"}
        assert step["tool"] is None


class TestAgentState:
    """Test the AgentState TypedDict."""

    def test_agent_state_initialization(self):
        """Test that AgentState initializes correctly."""
        state: AgentState = {"messages": [], "plan": []}
        assert state["messages"] == []
        assert state["plan"] == []

    def test_agent_state_with_data(self):
        """Test AgentState with initial data."""
        message = HumanMessage(content="Hello")
        step: Step = {"task": "Test task", "expected": "Test result", "tool": None, "status": "pending"}
        state: AgentState = {"messages": [message], "plan": [step]}
        assert len(state["messages"]) == 1
        assert len(state["plan"]) == 1
        assert state["plan"][0]["task"] == "Test task"


class TestSequentialAgent:
    """Test the SequentialAgent class."""

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_initialization(self, mock_init):
        """Test basic initialization of SequentialAgent."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        assert agent.model_name == "gpt-4"
        assert agent.model_provider == "openai"
        assert agent.tools == []
        assert agent.llm == mock_llm
        assert agent.tool_node is None
        assert agent.graph is not None

        mock_init.assert_called_once_with(model="gpt-4", model_provider="openai")

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_initialization_with_tools(self, mock_init):
        """Test initialization with tools."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        tools = [add, echo]
        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=tools)

        assert len(agent.tools) == 2
        assert agent.tools[0] == add
        assert agent.tools[1] == echo
        assert agent.tool_node is not None

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_initialization_with_llm_kwargs(self, mock_init):
        """Test initialization with LLM kwargs."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        llm_kwargs = {"temperature": 0.5, "max_tokens": 100}
        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", llm_kwargs=llm_kwargs)

        mock_init.assert_called_once_with(model="gpt-4", model_provider="openai", temperature=0.5, max_tokens=100)

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_bind_tools(self, mock_init):
        """Test binding additional tools."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[add])

        assert len(agent.tools) == 1

        # Bind additional tools
        agent.bind_tools([echo, search])

        assert len(agent.tools) == 3
        assert add in agent.tools
        assert echo in agent.tools
        assert search in agent.tools

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_creates_fresh_plan(self, mock_init):
        """Test planner creates a fresh plan when state plan is empty."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock the planning response
        plan_json = json.dumps(
            [
                {
                    "task": "Understand the query",
                    "expected": "Clear understanding of what user wants",
                    "tool": None,
                    "status": "pending",
                },
                {
                    "task": "Search for information",
                    "expected": "Relevant search results",
                    "tool": "search",
                    "status": "pending",
                },
            ]
        )

        planning_response = AIMessage(content=plan_json)
        mock_llm.invoke.return_value = planning_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[search])

        state: AgentState = {"messages": [HumanMessage(content="What is machine learning?")], "plan": []}

        result = agent._planner(state)

        assert "plan" in result
        assert len(result["plan"]) == 2
        assert result["plan"][0]["task"] == "Understand the query"
        assert result["plan"][1]["tool"] == "search"
        assert all(step["status"] == "pending" for step in result["plan"])

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_handles_malformed_json(self, mock_init):
        """Test planner handles malformed JSON gracefully."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock response with malformed JSON
        planning_response = AIMessage(content="This is not valid JSON")
        mock_llm.invoke.return_value = planning_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {"messages": [HumanMessage(content="What is machine learning?")], "plan": []}

        result = agent._planner(state)

        assert "plan" in result
        assert len(result["plan"]) == 1
        assert result["plan"][0]["task"].startswith("Re-evaluate and address the user query")
        assert result["plan"][0]["tool"] is None
        assert result["plan"][0]["status"] == "pending"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_skips_when_all_done(self, mock_init):
        """Test planner returns empty dict when all steps are done (fixes infinite loop)."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [{"task": "Task 1", "expected": "Result 1", "tool": None, "status": "done"}],
        }

        result = agent._planner(state)

        # Should return empty dict to allow controller to end the process
        assert result == {}
        # Verify LLM is not called since no new plan should be created
        mock_llm.invoke.assert_not_called()

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_direct_llm_execution(self, mock_init):
        """Test executor with direct LLM execution (no tools)."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        execution_response = AIMessage(content="I understand machine learning is...")
        revision_response = AIMessage(content='{"revisions": "No revisions needed", "change_plan": false}')
        mock_llm.invoke.side_effect = [execution_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [
                {"task": "Explain machine learning", "expected": "Clear explanation", "tool": None, "status": "pending"}
            ],
        }

        result = agent._executor(state)

        assert "messages" in result
        assert "plan" in result
        assert len(result["messages"]) == 3
        # First message is the task prompt
        assert "You are executing a specific step" in result["messages"][0].content
        # Second message is the AI response
        assert result["messages"][1].content == "I understand machine learning is..."
        # Third message is the revision
        assert "Revision: No revisions needed" in result["messages"][2].content
        assert result["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_with_tool_execution(self, mock_init):
        """Test executor with tool execution."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock LLM response with tool calls
        execution_response = MagicMock()
        execution_response.tool_calls = [{"name": "search", "args": {"query": "machine learning"}}]
        mock_llm_with_tools.invoke.return_value = execution_response

        # Mock revision evaluation response
        revision_response = AIMessage(content='{"revisions": "No revisions needed", "change_plan": false}')
        mock_llm.invoke.return_value = revision_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[search])

        # Mock tool node invocation
        mock_tool_result = "Search results for machine learning"
        agent.tool_node.invoke = MagicMock(return_value=mock_tool_result)

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [
                {
                    "task": "Search for machine learning information",
                    "expected": "Search results",
                    "tool": "search",
                    "status": "pending",
                }
            ],
        }

        result = agent._executor(state)

        assert "messages" in result
        assert "plan" in result
        assert len(result["messages"]) == 3
        # First message is the task prompt
        assert "You are executing a specific step" in result["messages"][0].content
        # Second message is the tool result
        assert isinstance(result["messages"][1], ToolMessage)
        # Third message is the revision
        assert "Revision: No revisions needed" in result["messages"][2].content
        assert result["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_no_pending_steps(self, mock_init):
        """Test executor when no pending steps exist."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [{"task": "Complete task", "expected": "Done", "tool": None, "status": "done"}],
        }

        result = agent._executor(state)

        # Should return empty dict when no pending steps
        assert result == {}

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_route_from_controller_with_pending_steps(self, mock_init):
        """Test controller routing with pending steps."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "plan": [{"task": "Pending task", "expected": "Result", "tool": None, "status": "pending"}],
        }

        result = agent._route_from_controller(state)
        assert result == "executor"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_route_from_controller_no_pending_steps(self, mock_init):
        """Test controller routing with no pending steps."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "plan": [{"task": "Done task", "expected": "Result", "tool": None, "status": "done"}],
        }

        result = agent._route_from_controller(state)
        assert result == "END"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_route_from_controller_empty_plan(self, mock_init):
        """Test controller routing with empty plan."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {"messages": [HumanMessage(content="Test")], "plan": []}

        result = agent._route_from_controller(state)
        assert result == "END"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_invoke_complete_workflow(self, mock_init):
        """Test complete invoke workflow with mocked graph."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # Mock the graph invoke method
        mock_final_state: AgentState = {
            "messages": [
                HumanMessage(content="What is machine learning?"),
                AIMessage(content="Machine learning is a subset of AI..."),
            ],
            "plan": [
                {"task": "Explain machine learning", "expected": "Clear explanation", "tool": None, "status": "done"}
            ],
        }

        agent.graph.invoke = MagicMock(return_value=mock_final_state)

        result = agent.invoke("What is machine learning?")

        assert len(result["messages"]) == 2
        assert len(result["plan"]) == 1
        assert result["plan"][0]["status"] == "done"

        # Verify the graph was called with correct initial state
        agent.graph.invoke.assert_called_once()
        call_args = agent.graph.invoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0].content == "What is machine learning?"
        assert call_args["plan"] == []

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_tool_execution_error_handling(self, mock_init):
        """Test error handling during tool execution."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock LLM response with tool calls
        execution_response = MagicMock()
        execution_response.tool_calls = [{"name": "search", "args": {"query": "machine learning"}}]
        mock_llm_with_tools.invoke.return_value = execution_response

        # Mock revision evaluation response
        revision_response = AIMessage(content='{"revisions": "No revisions needed", "change_plan": false}')
        mock_llm.invoke.return_value = revision_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[search])

        # Mock tool node to raise an exception
        agent.tool_node.invoke = MagicMock(side_effect=Exception("Tool error"))

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [
                {"task": "Search for information", "expected": "Search results", "tool": "search", "status": "pending"}
            ],
        }

        result = agent._executor(state)

        assert "messages" in result
        assert "plan" in result
        assert len(result["messages"]) == 3
        # First message is the task prompt
        assert "You are executing a specific step" in result["messages"][0].content
        # Second message is the tool error
        assert isinstance(result["messages"][1], ToolMessage)
        assert "Tool execution failed" in result["messages"][1].content
        # Third message is the revision
        assert "Revision: No revisions needed" in result["messages"][2].content
        assert result["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_with_code_block_json(self, mock_init):
        """Test planner handles JSON wrapped in code blocks."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        plan_data = [{"task": "Analyze query", "expected": "Understanding", "tool": None, "status": "pending"}]

        # JSON wrapped in code block
        planning_response = AIMessage(content=f"```json\n{json.dumps(plan_data)}\n```")
        mock_llm.invoke.return_value = planning_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {"messages": [HumanMessage(content="Test query")], "plan": []}

        result = agent._planner(state)

        assert "plan" in result
        assert len(result["plan"]) == 1
        assert result["plan"][0]["task"] == "Analyze query"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_command_on_change_plan(self, mock_init):
        """Test executor returns Command when revision recommends changing plan."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock task execution response
        task_response = AIMessage(content="Task completed with some issues")

        # Mock revision response that recommends changing the plan
        revision_json = json.dumps(
            {"revisions": "The current approach is insufficient, need to rethink strategy", "change_plan": True}
        )
        revision_response = AIMessage(content=revision_json)

        mock_llm.invoke.side_effect = [task_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # Create state with a pending step
        step: Step = {
            "task": "Test task",
            "expected": "Expected result",
            "tool": None,
            "status": "pending",
            "revisions": [],
        }
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "plan": [step],
            "current_query": "Test query",
        }

        result = agent._executor(state)

        # Should return a Command object since change_plan is True
        from langgraph.types import Command

        assert isinstance(result, Command)
        assert result.goto == "planner"
        assert "messages" in result.update
        assert "plan" in result.update
        assert result.update["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_dict_on_no_change_plan(self, mock_init):
        """Test executor returns dict when revision does not recommend changing plan."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock task execution response
        task_response = AIMessage(content="Task completed successfully")

        # Mock revision response that does not recommend changing the plan
        revision_json = json.dumps({"revisions": "Task was completed successfully", "change_plan": False})
        revision_response = AIMessage(content=revision_json)

        mock_llm.invoke.side_effect = [task_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # Create state with a pending step
        step: Step = {
            "task": "Test task",
            "expected": "Expected result",
            "tool": None,
            "status": "pending",
            "revisions": [],
        }
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "plan": [step],
            "current_query": "Test query",
        }

        result = agent._executor(state)

        # Should return a dict since change_plan is False
        assert isinstance(result, dict)
        assert "messages" in result
        assert "plan" in result
        assert result["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_replanning_scenario(self, mock_init):
        """Test planner handles replanning when needs_replanning flag is set."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock the replanning response
        replan_json = json.dumps(
            [
                {
                    "task": "Clean and preprocess data based on previous insights",
                    "expected": "Clean dataset ready for analysis",
                    "tool": "preprocess_tool",
                    "status": "pending",
                },
                {
                    "task": "Perform analysis with enhanced methodology",
                    "expected": "Comprehensive analysis results",
                    "tool": "analyze_data",
                    "status": "pending",
                },
            ]
        )

        replanning_response = AIMessage(content=replan_json)
        mock_llm.invoke.return_value = replanning_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[search])

        # State with mixed completed and pending steps (replanning scenario)
        # IMPORTANT: needs_replanning flag must be set for replanning to occur
        state: AgentState = {
            "messages": [HumanMessage(content="Analyze this complex dataset")],
            "plan": [
                {
                    "task": "Initial data exploration",
                    "expected": "Understanding of data structure",
                    "tool": None,
                    "status": "done",
                    "revisions": ["Data quality issues detected", "Need preprocessing step"],
                },
                {
                    "task": "Direct analysis of raw data",
                    "expected": "Analysis results",
                    "tool": "analyze_data",
                    "status": "pending",
                    "revisions": [],
                },
                {
                    "task": "Generate final report",
                    "expected": "Complete report",
                    "tool": None,
                    "status": "pending",
                    "revisions": [],
                },
            ],
            "current_query": "Analyze this complex dataset",
            "needs_replanning": True,  # This flag is now required for replanning
        }

        result = agent._planner(state)

        assert "plan" in result
        # Should have 1 completed step + 2 new replanned steps
        assert len(result["plan"]) == 3

        # First step should remain completed
        assert result["plan"][0]["status"] == "done"
        assert result["plan"][0]["task"] == "Initial data exploration"
        assert "Data quality issues detected" in result["plan"][0]["revisions"]

        # New steps should be pending and reflect insights from completed work
        assert result["plan"][1]["status"] == "pending"
        assert "preprocess" in result["plan"][1]["task"].lower()
        assert result["plan"][2]["status"] == "pending"
        assert "enhanced" in result["plan"][2]["task"].lower()

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_initial_vs_replanning_distinction(self, mock_init):
        """Test that planner correctly distinguishes initial planning from replanning."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # Test Case 1: Initial planning (empty plan)
        initial_plan_json = json.dumps([{"task": "Step 1", "expected": "Result 1", "tool": None, "status": "pending"}])
        mock_llm.invoke.return_value = AIMessage(content=initial_plan_json)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="New query")],
            "plan": [],
            "current_query": "New query",
        }

        result = agent._planner(initial_state)
        assert "plan" in result
        assert len(result["plan"]) == 1

        # Verify that initial planning prompt was used (check call was made)
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0][0].content
        assert "helpful planning assistant" in call_args
        assert "detailed step-by-step plan" in call_args

        # Reset mock for next test
        mock_llm.reset_mock()

        # Test Case 2: Replanning scenario
        replan_json = json.dumps(
            [{"task": "Revised step", "expected": "Better result", "tool": None, "status": "pending"}]
        )
        mock_llm.invoke.return_value = AIMessage(content=replan_json)

        replanning_state: AgentState = {
            "messages": [HumanMessage(content="Same query")],
            "plan": [
                {
                    "task": "Completed step",
                    "expected": "Done",
                    "tool": None,
                    "status": "done",
                    "revisions": ["Insight gained"],
                },
                {"task": "Pending step", "expected": "To do", "tool": None, "status": "pending", "revisions": []},
            ],
            "current_query": "Same query",
            "needs_replanning": True,  # This flag is now required for replanning
        }

        result = agent._planner(replanning_state)
        assert "plan" in result
        assert len(result["plan"]) == 2  # 1 completed + 1 replanned

        # Verify that replanning prompt was used
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0][0].content
        assert "replanning assistant" in call_args
        assert "<completed_work>" in call_args
        assert "<insights>" in call_args

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_no_replanning_without_flag(self, mock_init):
        """Test that planner does NOT replan when needs_replanning flag is False, even with revisions."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # State with completed steps that have revisions but no needs_replanning flag
        no_replan_state: AgentState = {
            "messages": [HumanMessage(content="Query")],
            "plan": [
                {
                    "task": "Completed step",
                    "expected": "Done",
                    "tool": None,
                    "status": "done",
                    "revisions": ["Some insight gained", "Another revision"],
                },
                {"task": "Pending step", "expected": "To do", "tool": None, "status": "pending", "revisions": []},
            ],
            "current_query": "Query",
            "needs_replanning": False,  # Explicitly set to False
        }

        result = agent._planner(no_replan_state)

        # Should return empty dict (no replanning triggered)
        assert result == {}
        # LLM should not be called since no replanning needed
        assert not mock_llm.invoke.called

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_continue_existing_plan(self, mock_init):
        """Test planner continues with existing plan when only pending steps remain."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # State with only pending steps (should continue existing plan)
        continue_state: AgentState = {
            "messages": [HumanMessage(content="Query")],
            "plan": [
                {"task": "Step 1", "expected": "Result 1", "tool": None, "status": "pending", "revisions": []},
                {"task": "Step 2", "expected": "Result 2", "tool": None, "status": "pending", "revisions": []},
            ],
            "current_query": "Query",
        }

        result = agent._planner(continue_state)

        # Should return empty dict (no replanning needed)
        assert result == {}
        # LLM should not be called since no planning needed
        assert not mock_llm.invoke.called

    def test_index_scanning_logic_robustness(self):
        """Test the core index-scanning logic handles replanning correctly."""
        # Test the core algorithm: scan from index 0 to find first pending step

        # Scenario 1: Original plan before execution
        original_plan = [
            {"task": "Step 0", "expected": "Result 0", "tool": None, "status": "done", "revisions": []},
            {"task": "Step 1", "expected": "Result 1", "tool": None, "status": "pending", "revisions": []},
            {"task": "Step 2", "expected": "Result 2", "tool": None, "status": "pending", "revisions": []},
        ]

        # Find next step index using the same logic as executor
        next_step_index = None
        for i, step in enumerate(original_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index == 1  # Should find step 1

        # Scenario 2: After step 1 completes and triggers replanning
        # Step 1 gets marked as done, plan gets restructured with new pending steps

        replanned_plan = [
            {"task": "Step 0", "expected": "Result 0", "tool": None, "status": "done", "revisions": []},
            {
                "task": "Step 1",
                "expected": "Result 1",
                "tool": None,
                "status": "done",
                "revisions": ["Completed but triggered replan"],
            },
            {"task": "New step A", "expected": "New result A", "tool": None, "status": "pending", "revisions": []},
            {"task": "New step B", "expected": "New result B", "tool": None, "status": "pending", "revisions": []},
        ]

        # Find next step index after replanning using same logic
        next_step_index = None
        for i, step in enumerate(replanned_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index == 2  # Should find "New step A" at index 2
        assert replanned_plan[next_step_index]["task"] == "New step A"

        # Scenario 3: After executing "New step A"
        post_execution_plan = [
            {"task": "Step 0", "expected": "Result 0", "tool": None, "status": "done", "revisions": []},
            {
                "task": "Step 1",
                "expected": "Result 1",
                "tool": None,
                "status": "done",
                "revisions": ["Completed but triggered replan"],
            },
            {"task": "New step A", "expected": "New result A", "tool": None, "status": "done", "revisions": []},
            {"task": "New step B", "expected": "New result B", "tool": None, "status": "pending", "revisions": []},
        ]

        # Find next step index after first replanned step completes
        next_step_index = None
        for i, step in enumerate(post_execution_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index == 3  # Should find "New step B" at index 3
        assert post_execution_plan[next_step_index]["task"] == "New step B"

        print("✅ Index-scanning logic is robust to plan restructuring!")

    def test_index_scanning_handles_edge_cases(self):
        """Test edge cases for the index scanning approach."""
        # Edge case 1: No pending steps
        all_done_plan = [
            {"task": "Step 0", "expected": "Result 0", "tool": None, "status": "done", "revisions": []},
            {"task": "Step 1", "expected": "Result 1", "tool": None, "status": "done", "revisions": []},
        ]

        next_step_index = None
        for i, step in enumerate(all_done_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index is None  # Should find no pending steps

        # Edge case 2: All pending steps (fresh plan)
        all_pending_plan = [
            {"task": "Step 0", "expected": "Result 0", "tool": None, "status": "pending", "revisions": []},
            {"task": "Step 1", "expected": "Result 1", "tool": None, "status": "pending", "revisions": []},
        ]

        next_step_index = None
        for i, step in enumerate(all_pending_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index == 0  # Should find first step

        # Edge case 3: Multiple replanning cycles (complex plan evolution)
        complex_evolved_plan = [
            {"task": "Original step 0", "expected": "Done", "tool": None, "status": "done", "revisions": []},
            {
                "task": "Original step 1",
                "expected": "Done",
                "tool": None,
                "status": "done",
                "revisions": ["Triggered first replan"],
            },
            {
                "task": "First replan step A",
                "expected": "Done",
                "tool": None,
                "status": "done",
                "revisions": ["Triggered second replan"],
            },
            {"task": "Second replan step X", "expected": "Current", "tool": None, "status": "pending", "revisions": []},
            {"task": "Second replan step Y", "expected": "Future", "tool": None, "status": "pending", "revisions": []},
        ]

        next_step_index = None
        for i, step in enumerate(complex_evolved_plan):
            if step["status"] == "pending":
                next_step_index = i
                break

        assert next_step_index == 3  # Should find first pending regardless of complex history
        assert complex_evolved_plan[next_step_index]["task"] == "Second replan step X"

        print("✅ Index-scanning handles edge cases correctly!")
