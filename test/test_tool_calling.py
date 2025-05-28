import pytest
from unittest.mock import MagicMock
from biochatter.llm_connect.conversation import Conversation
from langchain_core.messages import AIMessage


class MockTool:
    def __init__(self, name, result=None, raise_exc=False):
        self.name = name
        self.result = result
        self.raise_exc = raise_exc
        self.invoke = MagicMock(side_effect=self._invoke)
        self.ainvoke = MagicMock(side_effect=self._ainvoke)

    def _invoke(self, args):
        if self.raise_exc:
            raise Exception("Tool error")
        return self.result

    async def _ainvoke(self, args):
        if self.raise_exc:
            raise Exception("Tool error")
        return self.result


class DummyConversation(Conversation):
    def __init__(self, tool_call_mode="auto", mcp=False):
        super().__init__(
            model_name="dummy",
            prompts={
                "primary_model_prompts": [],
                "correcting_agent_prompts": [],
                "rag_agent_prompts": [],
                "tool_prompts": {},
            },
        )
        self.tool_call_mode = tool_call_mode
        self.mcp = mcp
        self.messages = []
        self.last_human_prompt = "What is the answer?"
        self.general_instructions_tool_interpretation = "General instructions."
        self.additional_instructions_tool_interpretation = "Additional instructions."
        self.chat = MagicMock()

    def _primary_query(self, text):
        pass

    def _correct_response(self, msg):
        pass

    def append_ai_message(self, message):
        self.messages.append(AIMessage(content=message))

    def set_api_key(self, api_key, user=None):
        pass


class ToolForFormat:
    def __init__(self, name, description, tool_call_schema, args):
        self.name = name
        self.description = description
        self.tool_call_schema = tool_call_schema
        self.args = args


@pytest.fixture
def mock_tools():
    tool1 = MockTool("tool1", result="result1")
    tool2 = MockTool("tool2", result="result2")
    tool3 = MockTool("tool3", raise_exc=True)
    return [tool1, tool2, tool3]


@pytest.fixture
def dummy_conv():
    return DummyConversation()


def test_auto_mode_success(dummy_conv, mock_tools):
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=False)
    assert "Tool call (tool1) result: result1" in msg
    assert "Tool call (tool2) result: result2" in msg
    # ToolMessage should be appended for each call
    assert any("result1" in str(m) for m in dummy_conv.messages)
    assert any("result2" in str(m) for m in dummy_conv.messages)


def test_auto_mode_exception(dummy_conv, mock_tools):
    tool_calls = [
        {"name": "tool3", "args": {"z": 3}, "id": "id3"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=False)
    assert "Error executing tool tool3" in msg
    assert any("Error executing tool tool3" in str(m) for m in dummy_conv.messages)


def test_text_mode(dummy_conv, mock_tools):
    dummy_conv.tool_call_mode = "text"
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=False)
    assert "Tool: tool1 - Arguments: {" in msg
    assert "Tool: tool2 - Arguments: {" in msg
    # Should append as AI message
    assert any(isinstance(m, AIMessage) for m in dummy_conv.messages)


def test_empty_tool_calls(dummy_conv, mock_tools):
    msg = dummy_conv._process_tool_calls([], mock_tools, "fallback", explain_tool_result=False)
    assert msg == "fallback"


def test_invalid_mode(dummy_conv, mock_tools):
    dummy_conv.tool_call_mode = "invalid_mode"
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=False)
    assert msg == "fallback"


def test_auto_mode_explain_tool_result_success(dummy_conv, mock_tools):
    # Patch chat.invoke to return a mock with .content
    dummy_conv.chat.invoke.return_value = MagicMock(content="This is an interpretation.")
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)
    assert "Tool call (tool1) result: result1" in msg
    assert "Tool result interpretation: This is an interpretation." in msg
    # Should append the interpretation as an AI message
    assert any(isinstance(m, AIMessage) and "This is an interpretation." in m.content for m in dummy_conv.messages)


def test_auto_mode_explain_tool_result_exception(dummy_conv, mock_tools):
    # Should not call interpretation if tool fails
    dummy_conv.chat.invoke.reset_mock()
    tool_calls = [
        {"name": "tool3", "args": {"z": 3}, "id": "id3"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)
    assert "Error executing tool tool3" in msg
    # chat.invoke should not be called
    dummy_conv.chat.invoke.assert_not_called()


def test_tool_formatter_non_mcp(dummy_conv):
    tools = [
        ToolForFormat(
            name="toolA",
            description="descA",
            tool_call_schema="schemaA",
            args="argsA",
        ),
        ToolForFormat(
            name="toolB",
            description="descB",
            tool_call_schema="schemaB",
            args="argsB",
        ),
    ]
    formatted = dummy_conv._tool_formatter(tools, mcp=False)
    assert "<tool_0>" in formatted
    assert "Tool name: toolA" in formatted
    assert "Tool description: descA" in formatted
    assert "Tool call schema:\n argsA" in formatted
    assert "<tool_1>" in formatted
    assert "Tool name: toolB" in formatted
    assert "Tool description: descB" in formatted
    assert "Tool call schema:\n argsB" in formatted


def test_tool_formatter_mcp(dummy_conv):
    tools = [
        ToolForFormat(
            name="toolA",
            description="descA",
            tool_call_schema="schemaA",
            args="argsA",
        )
    ]
    formatted = dummy_conv._tool_formatter(tools, mcp=True)
    assert "Tool call schema:\n schemaA" in formatted
    assert "argsA" not in formatted  # Should use tool_call_schema, not args


def test_create_tool_prompt(dummy_conv):
    # Add a user message to messages, as _create_tool_prompt expects it
    dummy_conv.messages.append(MagicMock(content="What is the capital of France?"))
    tools = [
        ToolForFormat(
            name="toolA",
            description="descA",
            tool_call_schema="schemaA",
            args="argsA",
        )
    ]
    prompt_msg = dummy_conv._create_tool_prompt(tools, additional_tools_instructions="Extra instructions", mcp=False)
    # prompt_msg is a SystemMessage or similar with .content
    content = prompt_msg.content if hasattr(prompt_msg, "content") else str(prompt_msg)
    assert "What is the capital of France?" in content
    assert "Tool name: toolA" in content
    assert "Tool description: descA" in content
    assert "Tool call schema:" in content
    assert "Extra instructions" in content


def test_porcess_manual_tool_call_non_mcp(dummy_conv):
    tool = MockTool("tool1", result="sync_result")
    tool.description = "desc"
    tool.tool_call_schema = "schema"
    tool.args = "args"
    tool_call = {"tool_name": "tool1", "param": 42}
    msg = dummy_conv._process_manual_tool_call(tool_call.copy(), [tool], explain_tool_result=False)
    assert "Tool: tool1" in msg
    assert "Arguments: {'param': 42}" in msg
    assert "Tool result: sync_result" in msg
    tool.invoke.assert_called_once_with({"param": 42})
    tool.ainvoke.assert_not_called()


def test_porcess_manual_tool_call_mcp(dummy_conv, monkeypatch):
    dummy_conv.mcp = True
    tool = MockTool("tool1", result="async_result")
    tool.description = "desc"
    tool.tool_call_schema = "schema"
    tool.args = "args"
    tool_call = {"tool_name": "tool1", "param": 99}

    # Patch asyncio.get_running_loop and run_until_complete
    class DummyLoop:
        def run_until_complete(self, coro):
            # Actually run the coroutine
            import asyncio

            return asyncio.get_event_loop().run_until_complete(coro)

    monkeypatch.setattr("asyncio.get_running_loop", lambda: DummyLoop())
    msg = dummy_conv._process_manual_tool_call(tool_call.copy(), [tool], explain_tool_result=False)
    assert "Tool: tool1" in msg
    assert "Arguments: {'param': 99}" in msg
    assert "Tool result: async_result" in msg
    tool.ainvoke.assert_called_once_with({"param": 99})
    tool.invoke.assert_not_called()


def test_porcess_manual_tool_call_explain_tool_result(dummy_conv):
    tool = MockTool("tool1", result="sync_result")
    tool.description = "desc"
    tool.tool_call_schema = "schema"
    tool.args = "args"
    tool_call = {"tool_name": "tool1", "param": 42}
    dummy_conv.chat.invoke.return_value = MagicMock(content="Interpreted result.")
    msg = dummy_conv._process_manual_tool_call(tool_call.copy(), [tool], explain_tool_result=True)
    assert "Tool result interpretation: Interpreted result." in msg
    assert any("Interpreted result." in m.content for m in dummy_conv.messages if hasattr(m, "content"))


def test_auto_mode_explain_tool_result_single_tool(dummy_conv, mock_tools):
    """Test that single tool call with explain_tool_result maintains current behavior."""
    dummy_conv.chat.invoke.return_value = MagicMock(content="Single tool interpretation.")
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)

    # Should contain tool result
    assert "Tool call (tool1) result: result1" in msg
    # Should contain interpretation (singular)
    assert "Tool result interpretation: Single tool interpretation." in msg
    # Should invoke chat.invoke once
    dummy_conv.chat.invoke.assert_called_once()
    # Should append the interpretation as an AI message
    assert any(isinstance(m, AIMessage) and "Single tool interpretation." in m.content for m in dummy_conv.messages)


def test_auto_mode_explain_tool_result_multiple_tools(dummy_conv, mock_tools):
    """Test that multiple tool calls with explain_tool_result are explained together."""
    dummy_conv.chat.invoke.return_value = MagicMock(content="Combined tools interpretation.")
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)

    # Should contain both tool results
    assert "Tool call (tool1) result: result1" in msg
    assert "Tool call (tool2) result: result2" in msg
    # Should contain interpretation (plural)
    assert "Tool results interpretation: Combined tools interpretation." in msg

    # Should invoke chat.invoke once with combined results
    dummy_conv.chat.invoke.assert_called_once()
    call_args = dummy_conv.chat.invoke.call_args[0][0]

    # The combined tool results should be in the prompt
    assert "Tool: tool1" in call_args
    assert "Arguments: {'x': 1}" in call_args
    assert "Result: result1" in call_args
    assert "Tool: tool2" in call_args
    assert "Arguments: {'y': 2}" in call_args
    assert "Result: result2" in call_args

    # Should append the interpretation as an AI message
    assert any(isinstance(m, AIMessage) and "Combined tools interpretation." in m.content for m in dummy_conv.messages)


def test_auto_mode_explain_tool_result_multiple_tools_with_error(dummy_conv, mock_tools):
    """Test that multiple tool calls with explain_tool_result handle errors correctly."""
    dummy_conv.chat.invoke.return_value = MagicMock(content="Partial success interpretation.")
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool3", "args": {"z": 3}, "id": "id3"},  # This will raise an exception
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)

    # Should contain successful tool result
    assert "Tool call (tool1) result: result1" in msg
    # Should contain error message for failed tool
    assert "Error executing tool tool3" in msg

    # Only successful tools should be included in explanation
    # Since only 1 tool succeeded, it should use single tool interpretation (not plural)
    dummy_conv.chat.invoke.assert_called_once()
    call_args = dummy_conv.chat.invoke.call_args[0][0]

    # Should only contain the successful tool result (not the full format since it's single tool)
    assert "result1" in call_args
    # Should not contain the failed tool
    assert "tool3" not in call_args
    # Should use singular interpretation since only 1 tool succeeded
    assert "Tool result interpretation: Partial success interpretation." in msg

    # Should append the interpretation as an AI message
    assert any(isinstance(m, AIMessage) and "Partial success interpretation." in m.content for m in dummy_conv.messages)


def test_auto_mode_explain_tool_result_all_tools_fail(dummy_conv, mock_tools):
    """Test that when all tools fail, no explanation is attempted."""
    dummy_conv.chat.invoke.reset_mock()
    tool_calls = [
        {"name": "tool3", "args": {"z": 3}, "id": "id3"},  # This will raise an exception
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)

    # Should contain error message
    assert "Error executing tool tool3" in msg
    # Should not invoke chat.invoke since no successful tools
    dummy_conv.chat.invoke.assert_not_called()


def test_auto_mode_explain_tool_result_three_tools(dummy_conv, mock_tools):
    """Test that three successful tool calls are explained together."""
    # Add another successful tool
    tool4 = MockTool("tool4", result="result4")
    all_tools = mock_tools + [tool4]

    dummy_conv.chat.invoke.return_value = MagicMock(content="Three tools interpretation.")
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
        {"name": "tool4", "args": {"w": 4}, "id": "id4"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, all_tools, "fallback", explain_tool_result=True)

    # Should contain all tool results
    assert "Tool call (tool1) result: result1" in msg
    assert "Tool call (tool2) result: result2" in msg
    assert "Tool call (tool4) result: result4" in msg
    # Should contain interpretation (plural)
    assert "Tool results interpretation: Three tools interpretation." in msg

    # Should invoke chat.invoke once with all combined results
    dummy_conv.chat.invoke.assert_called_once()
    call_args = dummy_conv.chat.invoke.call_args[0][0]

    # All three tools should be in the prompt
    assert "Tool: tool1" in call_args
    assert "Tool: tool2" in call_args
    assert "Tool: tool4" in call_args
    assert call_args.count("Tool:") == 3  # Ensure exactly 3 tools are mentioned


# ===== NEW TESTS FOR TOOL CALL TRACKING FUNCTIONALITY =====


def test_tool_call_tracking_enabled(dummy_conv, mock_tools):
    """Test that tool calls are tracked when track_tool_calls=True."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    # Initially, tool_calls deque should be empty
    assert len(dummy_conv.tool_calls) == 0

    # Process tool calls with tracking enabled
    msg = dummy_conv._process_tool_calls(
        tool_calls, mock_tools, "fallback", explain_tool_result=False, track_tool_calls=True
    )

    # Verify tool calls were tracked
    assert len(dummy_conv.tool_calls) == 2

    # Check first tracked tool call
    tracked_call_1 = dummy_conv.tool_calls[0]
    assert tracked_call_1["name"] == "tool1"
    assert tracked_call_1["args"] == {"x": 1}
    assert tracked_call_1["id"] == "id1"

    # Check second tracked tool call
    tracked_call_2 = dummy_conv.tool_calls[1]
    assert tracked_call_2["name"] == "tool2"
    assert tracked_call_2["args"] == {"y": 2}
    assert tracked_call_2["id"] == "id2"


def test_tool_call_tracking_disabled_by_default(dummy_conv, mock_tools):
    """Test that tool calls are not tracked by default (track_tool_calls=False)."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    # Initially, tool_calls deque should be empty
    assert len(dummy_conv.tool_calls) == 0

    # Process tool calls without tracking (default behavior)
    msg = dummy_conv._process_tool_calls(
        tool_calls, mock_tools, "fallback", explain_tool_result=False, track_tool_calls=False
    )

    # Verify tool calls were not tracked
    assert len(dummy_conv.tool_calls) == 0


def test_tool_call_tracking_disabled_explicit(dummy_conv, mock_tools):
    """Test that tool calls are not tracked when explicitly disabled."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]

    # Process tool calls with tracking explicitly disabled
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", track_tool_calls=False)

    # Verify tool calls were not tracked
    assert len(dummy_conv.tool_calls) == 0


def test_tool_call_tracking_with_tool_errors(dummy_conv, mock_tools):
    """Test that tool calls are tracked even when tools fail."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool3", "args": {"z": 3}, "id": "id3"},  # This tool will raise an exception
    ]

    # Process tool calls with tracking enabled
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", track_tool_calls=True)

    # Verify both tool calls were tracked (even the failed one)
    assert len(dummy_conv.tool_calls) == 2

    # Check successful tool call was tracked
    tracked_call_1 = dummy_conv.tool_calls[0]
    assert tracked_call_1["name"] == "tool1"
    assert tracked_call_1["args"] == {"x": 1}
    assert tracked_call_1["id"] == "id1"

    # Check failed tool call was also tracked
    tracked_call_2 = dummy_conv.tool_calls[1]
    assert tracked_call_2["name"] == "tool3"
    assert tracked_call_2["args"] == {"z": 3}
    assert tracked_call_2["id"] == "id3"


def test_tool_call_tracking_deque_behavior(dummy_conv, mock_tools):
    """Test that tool_calls behaves as a deque and can accumulate across multiple calls."""
    from collections import deque

    # Verify it's actually a deque
    assert isinstance(dummy_conv.tool_calls, deque)

    # First batch of tool calls
    tool_calls_1 = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]
    dummy_conv._process_tool_calls(tool_calls_1, mock_tools, "fallback", track_tool_calls=True)
    assert len(dummy_conv.tool_calls) == 1

    # Second batch of tool calls
    tool_calls_2 = [
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
        {"name": "tool1", "args": {"z": 3}, "id": "id3"},
    ]
    dummy_conv._process_tool_calls(tool_calls_2, mock_tools, "fallback", track_tool_calls=True)

    # Should now have 3 total tracked calls
    assert len(dummy_conv.tool_calls) == 3

    # Verify order is maintained (FIFO - first in, first out)
    assert dummy_conv.tool_calls[0]["id"] == "id1"
    assert dummy_conv.tool_calls[1]["id"] == "id2"
    assert dummy_conv.tool_calls[2]["id"] == "id3"


def test_tool_call_tracking_deque_access_methods(dummy_conv, mock_tools):
    """Test that deque access methods work correctly."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", track_tool_calls=True)

    # Test deque methods
    assert len(dummy_conv.tool_calls) == 2

    # Test popleft (remove from left)
    first_call = dummy_conv.tool_calls.popleft()
    assert first_call["name"] == "tool1"
    assert len(dummy_conv.tool_calls) == 1

    # Test appendleft (add to left)
    new_call = {"name": "new_tool", "args": {"a": 1}, "id": "new_id"}
    dummy_conv.tool_calls.appendleft(new_call)
    assert len(dummy_conv.tool_calls) == 2
    assert dummy_conv.tool_calls[0]["name"] == "new_tool"


def test_reset_clears_tool_calls(dummy_conv, mock_tools):
    """Test that the reset method clears the tool_calls deque."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    # Track some tool calls
    dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", track_tool_calls=True)
    assert len(dummy_conv.tool_calls) == 2

    # Reset the conversation
    dummy_conv.reset()

    # Verify tool_calls is cleared but still a deque
    assert len(dummy_conv.tool_calls) == 0
    from collections import deque

    assert isinstance(dummy_conv.tool_calls, deque)


def test_tool_call_tracking_text_mode_not_supported(dummy_conv, mock_tools):
    """Test that tool call tracking doesn't interfere with text mode."""
    dummy_conv.tool_call_mode = "text"
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    # In text mode, tools are not executed, so tracking doesn't happen
    msg = dummy_conv._process_tool_calls(
        tool_calls,
        mock_tools,
        "fallback",
        track_tool_calls=True,  # This should be ignored in text mode
    )

    # Verify the text formatting still works
    assert "Tool: tool1 - Arguments: {" in msg
    assert "Tool: tool2 - Arguments: {" in msg

    # Tool calls should not be tracked in text mode since tools aren't executed
    assert len(dummy_conv.tool_calls) == 0


def test_tool_call_tracking_empty_tool_calls(dummy_conv, mock_tools):
    """Test that tracking works correctly with empty tool calls."""
    # Process empty tool calls
    msg = dummy_conv._process_tool_calls([], mock_tools, "fallback", track_tool_calls=True)

    # Should return fallback and not track anything
    assert msg == "fallback"
    assert len(dummy_conv.tool_calls) == 0


def test_tool_call_tracking_with_missing_tool(dummy_conv, mock_tools):
    """Test tool call tracking when a tool is not found in available tools."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "nonexistent_tool", "args": {"y": 2}, "id": "id2"},
    ]

    # Process tool calls with a nonexistent tool
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", track_tool_calls=True)

    # The first tool should be tracked and executed
    assert len(dummy_conv.tool_calls) == 2
    assert dummy_conv.tool_calls[0]["name"] == "tool1"

    # The second tool should also be tracked even though it wasn't found/executed
    assert dummy_conv.tool_calls[1]["name"] == "nonexistent_tool"

    # But the message should only contain the result from the first tool
    assert "Tool call (tool1) result: result1" in msg
    # The nonexistent tool should not appear in results
    assert "nonexistent_tool" not in msg


def test_tool_call_tracking_preserves_original_functionality(dummy_conv, mock_tools):
    """Test that adding tool call tracking doesn't break existing functionality."""
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
        {"name": "tool2", "args": {"y": 2}, "id": "id2"},
    ]

    # Test with tracking enabled
    msg_with_tracking = dummy_conv._process_tool_calls(
        tool_calls, mock_tools, "fallback", explain_tool_result=False, track_tool_calls=True
    )

    # Reset for comparison
    dummy_conv.reset()

    # Test with tracking disabled
    msg_without_tracking = dummy_conv._process_tool_calls(
        tool_calls, mock_tools, "fallback", explain_tool_result=False, track_tool_calls=False
    )

    # The output messages should be identical
    assert msg_with_tracking == msg_without_tracking

    # Only difference should be in tracking
    assert len(dummy_conv.tool_calls) == 0  # No tracking in second call


def test_langchain_conversation_track_tool_calls_parameter():
    """Test that LangChainConversation properly passes through track_tool_calls parameter."""
    from biochatter.llm_connect.langchain import LangChainConversation
    from unittest.mock import patch, MagicMock

    # Create a mock conversation instance
    with patch.object(LangChainConversation, "set_api_key", return_value=True):
        conv = LangChainConversation(
            model_name="test-model",
            model_provider="test-provider",
            prompts={
                "primary_model_prompts": [],
                "correcting_agent_prompts": [],
                "rag_agent_prompts": [],
                "tool_prompts": {},
            },
        )

        # Mock the _process_tool_calls method to verify it receives the parameter
        conv._process_tool_calls = MagicMock(return_value="tool result")

        # Mock the chat and other necessary attributes
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "test_tool", "args": {}, "id": "test_id"}]
        mock_response.content = "test content"
        conv.chat = MagicMock()
        conv.chat.invoke = MagicMock(return_value=mock_response)
        conv.messages = []

        # Call _primary_query with track_tool_calls=True
        conv._primary_query(track_tool_calls=True)

        # Verify that _process_tool_calls was called with track_tool_calls=True
        conv._process_tool_calls.assert_called_once()
        call_args = conv._process_tool_calls.call_args
        assert call_args[1]["track_tool_calls"] is True
