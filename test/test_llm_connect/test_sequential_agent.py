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
        assert result["plan"][0]["task"].startswith("Address the user query")
        assert result["plan"][0]["tool"] is None
        assert result["plan"][0]["status"] == "pending"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_skips_when_all_done(self, mock_init):
        """Test planner skips when all steps are done."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "plan": [{"task": "Task 1", "expected": "Result 1", "tool": None, "status": "done"}],
        }

        # Should trigger fresh planning since all steps are done
        plan_json = json.dumps([{"task": "New task", "expected": "New result", "tool": None, "status": "pending"}])

        planning_response = AIMessage(content=plan_json)
        mock_llm.invoke.return_value = planning_response

        result = agent._planner(state)

        assert "plan" in result
        assert len(result["plan"]) == 1
        assert result["plan"][0]["task"] == "New task"

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
