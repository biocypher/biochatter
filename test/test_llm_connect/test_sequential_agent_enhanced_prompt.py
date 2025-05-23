"""Test enhanced prompt functionality in sequential agent."""

import json
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

        # Mock revision evaluation response
        revision_response = AIMessage(content='["Could be more specific", "Add more details"]')
        mock_llm.invoke.return_value = revision_response

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
        assert "<original_question>" in prompt_content
        assert "What is the best approach for analyzing gene expression data?" in prompt_content

        # Verify that completed steps are included
        assert "<completed_steps>" in prompt_content
        assert "Identify data preprocessing requirements" in prompt_content
        assert "Research analysis methods" in prompt_content

        # Verify that current task is included
        assert "<current_step>" in prompt_content
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

        # Mock revision evaluation response
        revision_response = AIMessage(content='{"revisions": "No revisions needed", "change_plan": false}')
        mock_llm.invoke.return_value = revision_response

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
        assert "<original_question>" in prompt_content
        assert "Search for recent papers on CRISPR" in prompt_content

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_without_current_query_falls_back_to_human_message(self, mock_init):
        """Test that executor falls back to human message when current_query is not available."""
        mock_llm = MagicMock()

        mock_init.return_value = mock_llm

        execution_response = AIMessage(content="Using fallback to human message...")
        # Mock revision evaluation response as well
        revision_response = AIMessage(content='{"revisions": "No revisions needed", "change_plan": false}')
        mock_llm.invoke.side_effect = [execution_response, revision_response]

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
        call_args = mock_llm.invoke.call_args_list[0][0][0]  # First call is execution
        human_message = call_args[-1]
        prompt_content = human_message.content

        # Verify that it uses the first human message as the original question
        assert "<original_question>" in prompt_content
        assert "Analyze this dataset" in prompt_content

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

        # Mock revision evaluation response
        revision_response = AIMessage(
            content='{"revisions": "Tool results could be more comprehensive", "change_plan": false}'
        )
        mock_llm.invoke.return_value = revision_response

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
        assert len(result["messages"]) == 4
        # First message is the task prompt
        assert "You are executing a specific step" in result["messages"][0].content
        # Next two messages are the tool results
        assert isinstance(result["messages"][1], ToolMessage)
        assert isinstance(result["messages"][2], ToolMessage)
        assert result["messages"][1].content == "First tool result"
        assert result["messages"][2].content == "Second tool result"
        # Fourth message is the revision
        assert "Revision: Tool results could be more comprehensive" in result["messages"][3].content
        assert result["plan"][0]["status"] == "done"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_evaluates_results_and_populates_revisions(self, mock_init):
        """Test that executor evaluates results against expectations and populates revisions field."""
        mock_llm = MagicMock()

        mock_init.return_value = mock_llm

        # Mock execution response
        execution_response = AIMessage(content="Analysis complete. Found 3 genes with significant expression changes.")

        # Mock revision evaluation response with specific recommendations
        revision_text = "Should include statistical significance values"
        revision_response = AIMessage(content=json.dumps({"revisions": revision_text, "change_plan": False}))

        # Set up side_effect to return different responses for each call
        mock_llm.invoke.side_effect = [execution_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="Analyze gene expression data")],
            "current_query": "Analyze gene expression data",
            "plan": [
                {
                    "task": "Perform statistical analysis of gene expression",
                    "expected": "Detailed report with statistical significance and methodology",
                    "tool": None,
                    "status": "pending",
                }
            ],
        }

        result = agent._executor(state)

        # Verify that revision evaluation was called
        assert mock_llm.invoke.call_count == 2

        # Get the revision evaluation prompt (second call)
        revision_call_args = mock_llm.invoke.call_args_list[1][0][0]
        revision_prompt = revision_call_args[-1].content

        # Verify revision prompt contains expected elements
        assert "<original_question>" in revision_prompt
        assert "Analyze gene expression data" in revision_prompt
        assert "Task: Perform statistical analysis of gene expression" in revision_prompt
        assert "Expected Outcome: Detailed report with statistical significance and methodology" in revision_prompt
        assert "<actual_results>" in revision_prompt
        assert "Analysis complete. Found 3 genes with significant expression changes." in revision_prompt

        # Verify that revisions were populated in the step
        assert "plan" in result
        completed_step = result["plan"][0]
        assert completed_step["status"] == "done"
        assert completed_step["revisions"] == [revision_text]

        # Verify that the original step fields are preserved
        assert completed_step["task"] == "Perform statistical analysis of gene expression"
        assert completed_step["expected"] == "Detailed report with statistical significance and methodology"

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_handles_revision_parsing_errors_gracefully(self, mock_init):
        """Test that executor handles malformed revision JSON gracefully."""
        mock_llm = MagicMock()

        mock_init.return_value = mock_llm

        # Mock execution response
        execution_response = AIMessage(content="Task completed successfully.")

        # Mock revision evaluation response with malformed JSON
        revision_response = AIMessage(content="This is not valid JSON format")

        mock_llm.invoke.side_effect = [execution_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        state: AgentState = {
            "messages": [HumanMessage(content="Complete the task")],
            "current_query": "Complete the task",
            "plan": [
                {
                    "task": "Execute task",
                    "expected": "Successful completion",
                    "tool": None,
                    "status": "pending",
                }
            ],
        }

        result = agent._executor(state)

        # Verify that revisions field is empty list when parsing fails
        assert "plan" in result
        completed_step = result["plan"][0]
        assert completed_step["status"] == "done"
        assert completed_step["revisions"] == []  # Should be empty list on parsing error

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_includes_revisions_in_task_prompt(self, mock_init):
        """Test that existing revisions are included in the task prompt."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()

        mock_init.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock execution response
        execution_response = AIMessage(content="Executing task with revisions in mind...")
        mock_llm_with_tools.invoke.return_value = execution_response

        # Mock revision evaluation response
        revision_response = AIMessage(content='{"revisions": "New revision for this attempt", "change_plan": false}')
        mock_llm.invoke.return_value = revision_response

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai", tools=[test_tool])

        # State with a step that has existing revisions
        state: AgentState = {
            "messages": [HumanMessage(content="Analyze protein structure data")],
            "current_query": "Analyze protein structure data",
            "plan": [
                {
                    "task": "Generate structural analysis report",
                    "expected": "Comprehensive protein structure analysis",
                    "tool": "test_tool",
                    "status": "pending",
                    "revisions": [
                        "Previous attempt lacked detailed bond analysis",
                        "Need to include secondary structure information",
                    ],
                }
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
        assert "<original_question>" in prompt_content
        assert "Analyze protein structure data" in prompt_content

        # Verify that current task is included
        assert "<current_step>" in prompt_content
        assert "Generate structural analysis report" in prompt_content

        # Verify that existing revisions are included in the prompt
        assert "<revisions>" in prompt_content
        assert "Previous revisions for this step:" in prompt_content
        assert "- Previous attempt lacked detailed bond analysis" in prompt_content
        assert "- Need to include secondary structure information" in prompt_content

        # Verify that suggested tool information is included
        assert "SUGGESTED TOOL: test_tool - Use this tool if appropriate for the task." in prompt_content

        # Verify response handling and that new revision was added
        assert "messages" in result
        assert "plan" in result
        completed_step = result["plan"][0]
        assert completed_step["status"] == "done"

        # Check that the new revision was appended to existing ones
        expected_revisions = [
            "Previous attempt lacked detailed bond analysis",
            "Need to include secondary structure information",
            "New revision for this attempt",
        ]
        assert completed_step["revisions"] == expected_revisions

    @patch("biochatter.llm_connect.sequential_agent.init_chat_model")
    def test_executor_handles_no_revisions_gracefully(self, mock_init):
        """Test that task prompt works correctly when no revisions exist."""
        mock_llm = MagicMock()

        mock_init.return_value = mock_llm

        # Mock execution response
        execution_response = AIMessage(content="Executing task without previous revisions...")
        # Mock revision evaluation response
        revision_response = AIMessage(content='{"revisions": "First revision for this step", "change_plan": false}')
        mock_llm.invoke.side_effect = [execution_response, revision_response]

        agent = SequentialAgent(model_name="gpt-4", model_provider="openai")

        # State with a step that has no existing revisions
        state: AgentState = {
            "messages": [HumanMessage(content="Process dataset")],
            "current_query": "Process dataset",
            "plan": [
                {
                    "task": "Clean and preprocess data",
                    "expected": "Cleaned dataset ready for analysis",
                    "tool": None,
                    "status": "pending",
                    # No revisions field
                }
            ],
        }

        result = agent._executor(state)

        # Get the prompt content
        call_args = mock_llm.invoke.call_args_list[0][0][0]  # First call is execution
        human_message = call_args[-1]
        prompt_content = human_message.content

        # Verify that the prompt does NOT include revisions section when no revisions exist
        assert "Previous revisions for this step:" not in prompt_content

        # Verify that other expected sections are still present
        assert "<original_question>" in prompt_content
        assert "Process dataset" in prompt_content
        assert "<current_step>" in prompt_content
        assert "Clean and preprocess data" in prompt_content

        # Verify that the first revision was added correctly
        completed_step = result["plan"][0]
        assert completed_step["revisions"] == ["First revision for this step"]
