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
