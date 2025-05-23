"""Test for the sequential agent infinite loop fix."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from biochatter.llm_connect.sequential_agent import SequentialAgent, AgentState


@tool
def mock_add_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestSequentialAgentFix:
    """Test suite for the sequential agent infinite loop fix."""

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_does_not_recreate_plan_when_all_done(self, mock_init_chat_model):
        """Test that planner doesn't create a new plan when all steps are completed."""
        # Create a mock LLM
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm

        # Create agent
        agent = SequentialAgent(model_name="test-model", model_provider="test-provider", tools=[mock_add_tool])

        # Create state with all steps completed
        completed_plan = [
            {"task": "Add 5 and 3", "expected": "8", "tool": "mock_add_tool", "status": "done", "revisions": []},
            {
                "task": "Multiply result by 4",
                "expected": "32",
                "tool": "mock_multiply_tool",
                "status": "done",
                "revisions": [],
            },
        ]

        state: AgentState = {
            "messages": [HumanMessage(content="test query")],
            "plan": completed_plan,
            "current_query": "test query",
            "needs_replanning": False,
        }

        # Call planner
        result = agent._planner(state)

        # Verify that no new plan is created (returns empty dict)
        assert result == {}

        # Verify LLM is not called to create a new plan
        mock_llm.invoke.assert_not_called()

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_planner_creates_initial_plan_when_no_plan_exists(self, mock_init_chat_model):
        """Test that planner creates initial plan when no plan exists."""
        # Create a mock LLM that returns a valid plan
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """[
            {
                "task": "Test task",
                "expected": "Test expected",
                "tool": null,
                "status": "pending"
            }
        ]"""
        mock_llm.invoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_llm

        # Create agent
        agent = SequentialAgent(model_name="test-model", model_provider="test-provider", tools=[mock_add_tool])

        # Create state with no plan
        state: AgentState = {
            "messages": [HumanMessage(content="test query")],
            "plan": None,
            "current_query": "test query",
            "needs_replanning": False,
        }

        # Call planner
        result = agent._planner(state)

        # Verify that a new plan is created
        assert "plan" in result
        assert len(result["plan"]) == 1
        assert result["plan"][0]["task"] == "Test task"

        # Verify LLM was called to create the plan
        mock_llm.invoke.assert_called_once()

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_route_from_controller_ends_when_all_done(self, mock_init_chat_model):
        """Test that controller routes to END when all steps are done."""
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm
        agent = SequentialAgent(model_name="test-model", model_provider="test-provider", tools=[mock_add_tool])

        # Create state with all steps completed
        completed_plan = [
            {"task": "Test task 1", "expected": "Result 1", "tool": None, "status": "done", "revisions": []},
            {"task": "Test task 2", "expected": "Result 2", "tool": None, "status": "done", "revisions": []},
        ]

        state: AgentState = {
            "messages": [HumanMessage(content="test")],
            "plan": completed_plan,
            "current_query": "test",
            "needs_replanning": False,
        }

        # Call route_from_controller
        route = agent._route_from_controller(state)

        # Verify it routes to END
        assert route == "END"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_route_from_controller_continues_when_pending_steps(self, mock_init_chat_model):
        """Test that controller routes to executor when there are pending steps."""
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm
        agent = SequentialAgent(model_name="test-model", model_provider="test-provider", tools=[mock_add_tool])

        # Create state with pending steps
        mixed_plan = [
            {"task": "Test task 1", "expected": "Result 1", "tool": None, "status": "done", "revisions": []},
            {"task": "Test task 2", "expected": "Result 2", "tool": None, "status": "pending", "revisions": []},
        ]

        state: AgentState = {
            "messages": [HumanMessage(content="test")],
            "plan": mixed_plan,
            "current_query": "test",
            "needs_replanning": False,
        }

        # Call route_from_controller
        route = agent._route_from_controller(state)

        # Verify it routes to executor
        assert route == "executor"


if __name__ == "__main__":
    pytest.main([__file__])
