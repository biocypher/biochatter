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
    dummy_conv.chat.invoke.return_value = MagicMock(content='This is an interpretation.')
    tool_calls = [
        {"name": "tool1", "args": {"x": 1}, "id": "id1"},
    ]
    msg = dummy_conv._process_tool_calls(tool_calls, mock_tools, "fallback", explain_tool_result=True)
    assert "Tool call (tool1) result: result1" in msg
    assert "Tool result interpretation: This is an interpretation." in msg
    # Should append the interpretation as an AI message
    assert any(
        isinstance(m, AIMessage) and "This is an interpretation." in m.content for m in dummy_conv.messages
    )


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
