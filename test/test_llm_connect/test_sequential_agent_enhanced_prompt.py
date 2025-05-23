"""Test enhanced prompt functionality in sequential agent."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from biochatter.llm_connect.sequential_agent import AgentState, SequentialAgent


@tool
def test_tool(query: str) -> str:
    """Test tool for testing purposes."""
    return f"Test result for: {query}"


class TestSequentialAgentEnhancedPrompt:
    """Test enhanced prompt functionality that includes original question and completed steps."""

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_includes_original_question_and_completed_steps(self, mock_init):
        """Test that executor prompt includes original question and completed steps."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()

        # Mock the LLM setup
        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock execution response
        execution_response = AIMessage(content="Based on the original question and completed work...")
        mock_llm_with_tools.invoke.return_value = execution_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[test_tool])

        # Set up state with original query and some completed steps
        state: AgentState = {
            "messages": [HumanMessage(content="What is the best approach for analyzing gene expression data?")],
            "current_query": "What is the best approach for analyzing gene expression data?",
            "plan": [
                {
                    "task": "Identify data preprocessing requirements",
                    "expected": "List of preprocessing steps",
                    "tool": None,
                    "status": "done",
                },
                {
                    "task": "Research analysis methods",
                    "expected": "Comparison of different methods",
                    "tool": "test_tool",
                    "status": "done",
                },
                {
                    "task": "Provide final recommendation",
                    "expected": "Specific recommendation with reasoning",
                    "tool": None,
                    "status": "pending",
                },
            ],
        }

        result = agent._executor(state)

        # Verify that the LLM was called
        mock_llm_with_tools.invoke.assert_called_once()

        # Get the call arguments to verify the prompt content
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        human_message = call_args[-1]  # Last message should be the task prompt
        prompt_content = human_message.content

        # Verify that the prompt includes the original question
        assert "ORIGINAL QUESTION: What is the best approach for analyzing gene expression data?" in prompt_content

        # Verify that completed steps are included
        assert "COMPLETED STEPS SO FAR:" in prompt_content
        assert "Identify data preprocessing requirements" in prompt_content
        assert "Research analysis methods" in prompt_content

        # Verify that current task is included
        assert "CURRENT STEP TO EXECUTE:" in prompt_content
        assert "Provide final recommendation" in prompt_content

        # Verify the contextualized instruction
        assert "Based on the original question, progress made so far" in prompt_content

        # Verify response handling
        assert "messages" in result
        assert "plan" in result
        assert result["plan"][2]["status"] == "done"  # Third step should now be done

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_with_suggested_tool_in_prompt(self, mock_init):
        """Test that suggested tool information is included in the prompt."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()

        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        execution_response = AIMessage(content="I'll use the suggested tool...")
        mock_llm_with_tools.invoke.return_value = execution_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[test_tool])

        state: AgentState = {
            "messages": [HumanMessage(content="Search for recent papers on CRISPR")],
            "current_query": "Search for recent papers on CRISPR",
            "plan": [
                {
                    "task": "Search academic databases",
                    "expected": "List of recent CRISPR papers",
                    "tool": "test_tool",
                    "status": "pending",
                }
            ],
        }

        result = agent._executor(state)

        # Get the prompt content
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        human_message = call_args[-1]
        prompt_content = human_message.content

        # Verify that suggested tool information is included
        assert "SUGGESTED TOOL: test_tool - Use this tool if appropriate for the task." in prompt_content
        assert "ORIGINAL QUESTION: Search for recent papers on CRISPR" in prompt_content

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_without_current_query_falls_back_to_human_message(self, mock_init):
        """Test that executor falls back to human message when current_query is not available."""
        mock_llm = MagicMock()

        mock_init.return_value = mock_llm

        execution_response = AIMessage(content="Using fallback to human message...")
        mock_llm.invoke.return_value = execution_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # State without current_query
        state: AgentState = {
            "messages": [
                HumanMessage(content="Analyze this dataset"),
                AIMessage(content="I can help with that"),
                HumanMessage(content="Please start the analysis"),
            ],
            "plan": [
                {"task": "Begin analysis", "expected": "Initial analysis results", "tool": None, "status": "pending"}
            ],
        }

        result = agent._executor(state)

        # Get the prompt content - when no tools, uses mock_llm directly
        call_args = mock_llm.invoke.call_args[0][0]
        human_message = call_args[-1]
        prompt_content = human_message.content

        # Verify that it uses the first human message as the original question
        assert "ORIGINAL QUESTION: Analyze this dataset" in prompt_content

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_handles_multiple_tool_messages(self, mock_init):
        """Test that executor properly handles multiple ToolMessage objects returned from tool execution."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()

        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock LLM response with tool calls
        execution_response = MagicMock()
        execution_response.tool_calls = [{"name": "test_tool", "args": {"query": "test"}}]
        mock_llm_with_tools.invoke.return_value = execution_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[test_tool])

        # Mock tool node to return multiple messages
        mock_tool_messages = [
            ToolMessage(content="First tool result", tool_call_id="call_1"),
            ToolMessage(content="Second tool result", tool_call_id="call_2"),
        ]
        agent.tool_node.invoke = MagicMock(return_value={"messages": mock_tool_messages})

        state: AgentState = {
            "messages": [HumanMessage(content="Search for information")],
            "current_query": "Search for information",
            "plan": [
                {
                    "task": "Search databases",
                    "expected": "Multiple search results",
                    "tool": "test_tool",
                    "status": "pending",
                }
            ],
        }

        result = agent._executor(state)

        # Verify that multiple ToolMessages are preserved
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert all(isinstance(msg, ToolMessage) for msg in result["messages"])
        assert result["messages"][0].content == "First tool result"
        assert result["messages"][1].content == "Second tool result"
        assert result["plan"][0]["status"] == "done"
