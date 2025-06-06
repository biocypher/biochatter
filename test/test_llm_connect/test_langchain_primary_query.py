import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from biochatter.llm_connect.langchain import LangChainConversation

# Import TOOL_CALLING_MODELS and STRUCTURED_OUTPUT_MODELS from where they are referenced in langchain.py
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# Mock Pydantic model for structured output tests
class MockOutputModel(BaseModel):
    param1: str
    param2: int


# Mock tool function
def mock_tool_one(arg1: str) -> str:
    """A mock tool function one."""
    return f"Result from tool one: {arg1}"


mock_tool_one.__name__ = "mock_tool_one"
mock_tool_one.__doc__ = "A mock tool function one that takes arg1 and returns a string."


def mock_tool_two(arg_x: int) -> str:
    """A mock tool function two."""
    return f"Result from tool two: {arg_x}"


mock_tool_two.__name__ = "mock_tool_two"
mock_tool_two.__doc__ = "A mock tool function two that takes arg_x and returns a string."


@pytest.fixture
def mock_chat_object():
    chat_mock = MagicMock()

    # Default invoke returns a simple AIMessage like response
    mock_response = AIMessage(content="Default AI response", id="ai_msg_1")
    mock_response.usage_metadata = {"total_tokens": 50}
    # tool_calls should be an attribute, even if empty list initially
    mock_response.tool_calls = []

    chat_mock.invoke = MagicMock(return_value=mock_response)
    # bind_tools and with_structured_output should return a chat-like object (can be self)
    chat_mock.bind_tools = MagicMock(return_value=chat_mock)
    chat_mock.with_structured_output = MagicMock(return_value=chat_mock)

    return chat_mock


@pytest.fixture
def conversation_instance(mock_chat_object):
    prompts = {"system_prompt": "You are a helpful assistant."}
    instance = LangChainConversation(
        model_name="test-model",  # This will be overridden in tests or patched constants
        model_provider="test-provider",
        prompts=prompts,
    )
    instance.chat = mock_chat_object
    instance.messages = [HumanMessage(content="User query")]  # A starting message
    instance.tools = []  # Instance tools, can be populated by tests
    instance.tools_prompt = None
    instance.additional_tools_instructions = None

    # Mock methods that _primary_query calls internally
    instance.append_ai_message = MagicMock()
    instance._process_tool_calls = MagicMock(return_value="Processed tool output string")
    instance._process_manual_tool_call = MagicMock(return_value="Processed manual tool output string")
    # _create_tool_prompt returns a message object, e.g. SystemMessage
    instance._create_tool_prompt = MagicMock(return_value=SystemMessage(content="Generated tool prompt"))
    return instance


def test_primary_query_no_tools_no_structured_basic_model(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "basic-model"
    initial_messages = list(conversation_instance.messages)

    mock_response_content = "AI response without tools"
    mock_response = AIMessage(content=mock_response_content, id="ai_msg_2")
    mock_response.usage_metadata = {"total_tokens": 10}
    mock_response.tool_calls = []
    mock_chat_object.invoke = MagicMock(return_value=mock_response)

    with (
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
    ):
        msg, token_usage = conversation_instance._primary_query(tools=None, structured_model=None)

    assert msg == mock_response_content
    assert token_usage == 10
    # The mock records the final state of self.messages, which includes the appended response
    expected_final_messages = initial_messages + [mock_response]
    mock_chat_object.invoke.assert_called_once_with(expected_final_messages)
    conversation_instance.append_ai_message.assert_not_called()  # Changed: now uses messages.append(response)
    mock_chat_object.bind_tools.assert_not_called()
    mock_chat_object.with_structured_output.assert_not_called()


def test_primary_query_with_tools_model_supports_tool_calling_tool_used(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "gemini-2.0-flash"  # Assumed to be in TOOL_CALLING_MODELS
    conversation_instance.tools = [mock_tool_one]  # Instance tool
    query_tool = mock_tool_two  # Tool passed in query
    all_tools = [mock_tool_one, query_tool]
    initial_messages = list(conversation_instance.messages)

    response_ai_content = "Okay, I will use a tool."
    mock_tool_call_dict = {"name": "mock_tool_two", "args": {"arg_x": 123}, "id": "tool_call_123"}

    # Mock AIMessage with tool_calls
    mock_llm_response = AIMessage(content=response_ai_content, id="ai_msg_3")
    mock_llm_response.tool_calls = [mock_tool_call_dict]  # Langchain stores tool calls here
    mock_llm_response.usage_metadata = {"total_tokens": 20}  # This won't be used if tool_calls path is taken
    mock_chat_object.invoke = MagicMock(return_value=mock_llm_response)

    processed_tool_output = "Tool mock_tool_two finished."
    conversation_instance._process_tool_calls = MagicMock(return_value=processed_tool_output)

    with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", [conversation_instance.model_name]):
        msg, token_usage = conversation_instance._primary_query(
            tools=[query_tool], explain_tool_result=True, return_tool_calls_as_ai_message=False
        )

    assert msg == processed_tool_output
    assert token_usage == 20  # Token usage is now returned when available, even with tool calls
    mock_chat_object.bind_tools.assert_called_once_with(all_tools)
    bound_chat_mock = mock_chat_object.bind_tools.return_value
    # The mock records the final state of self.messages, which includes the appended response
    expected_final_messages = initial_messages + [mock_llm_response]
    bound_chat_mock.invoke.assert_called_once_with(expected_final_messages)
    conversation_instance._process_tool_calls.assert_called_once_with(
        tool_calls=[mock_tool_call_dict],
        available_tools=all_tools,
        response_content=response_ai_content,
        explain_tool_result=True,
        return_tool_calls_as_ai_message=False,
        track_tool_calls=False,
    )
    # Note: append_ai_message is NOT called after _process_tool_calls in the actual implementation
    # because _process_tool_calls handles adding messages internally


def test_primary_query_with_tools_model_supports_tool_calling_no_tool_use(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "gpt-4-turbo"
    initial_messages = list(conversation_instance.messages)

    mock_response_content = "AI response, no tool needed."
    mock_llm_response = AIMessage(content=mock_response_content, id="ai_msg_4")
    mock_llm_response.tool_calls = []  # No tool calls
    mock_llm_response.usage_metadata = {"total_tokens": 25}
    mock_chat_object.invoke = MagicMock(return_value=mock_llm_response)

    with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", [conversation_instance.model_name]):
        msg, token_usage = conversation_instance._primary_query(tools=[mock_tool_one])

    assert msg == mock_response_content
    assert token_usage == 25
    mock_chat_object.bind_tools.assert_called_once_with([mock_tool_one])
    # The mock records the final state of self.messages, which includes the appended response
    expected_final_messages = initial_messages + [mock_llm_response]
    mock_chat_object.bind_tools.return_value.invoke.assert_called_once_with(expected_final_messages)
    conversation_instance._process_tool_calls.assert_not_called()
    conversation_instance.append_ai_message.assert_not_called()  # Changed: now uses messages.append(response)


def test_primary_query_tools_model_not_supports_manual_call_success(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "non-tool-model"
    initial_user_message_content = conversation_instance.messages[-1].content

    # LLM returns content that is a parsable JSON tool call
    raw_llm_response_content = """{
  "tool_name": "mock_tool_one",
  "arguments": {"arg1": "manual_test"}
}"""
    mock_llm_response = AIMessage(content=raw_llm_response_content, id="ai_msg_5")
    mock_llm_response.tool_calls = []
    mock_llm_response.usage_metadata = {"total_tokens": 30}
    mock_chat_object.invoke = MagicMock(return_value=mock_llm_response)

    system_tool_prompt_obj = SystemMessage(content="System prompt for tools generated by _create_tool_prompt")
    conversation_instance._create_tool_prompt = MagicMock(return_value=system_tool_prompt_obj)

    processed_manual_tool_output = "Manual tool mock_tool_one executed."
    conversation_instance._process_manual_tool_call = MagicMock(return_value=processed_manual_tool_output)

    with (
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
    ):
        msg, token_usage = conversation_instance._primary_query(tools=[mock_tool_one], explain_tool_result=True)

    assert msg == processed_manual_tool_output
    assert token_usage == 30  # Token usage is now returned when available, even with manual tool calls

    conversation_instance._create_tool_prompt.assert_called_once_with(
        tools=[mock_tool_one], additional_instructions=None
    )
    # Check self.messages was modified for invoke
    # Initial messages: [HumanMessage(content="User query")]
    # After tool prompt: messages passed to invoke should be [system_tool_prompt_obj]
    # because self.messages[-1] = self.tools_prompt
    expected_messages_for_invoke = [system_tool_prompt_obj]
    # The actual self.messages list in instance is modified in place.
    mock_chat_object.invoke.assert_called_once_with(expected_messages_for_invoke)
    assert (
        conversation_instance.messages == expected_messages_for_invoke
    )  # Verify final state of self.messages before append

    expected_parsed_json = {"tool_name": "mock_tool_one", "arguments": {"arg1": "manual_test"}}
    conversation_instance._process_manual_tool_call.assert_called_once_with(
        tool_call=expected_parsed_json, available_tools=[mock_tool_one], explain_tool_result=True
    )
    # Note: append_ai_message is called by _process_manual_tool_call internally, not by _primary_query


def test_primary_query_tools_model_not_supports_invalid_json_response(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "non-tool-model"

    raw_llm_response_content = "This is not a valid JSON tool call."
    mock_llm_response = AIMessage(content=raw_llm_response_content, id="ai_msg_6")
    mock_llm_response.tool_calls = []
    mock_llm_response.usage_metadata = {"total_tokens": 30}  # Not used if error path
    mock_chat_object.invoke = MagicMock(return_value=mock_llm_response)

    system_tool_prompt_obj = SystemMessage(content="Tool prompt")
    conversation_instance._create_tool_prompt = MagicMock(return_value=system_tool_prompt_obj)

    with (
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
    ):
        msg, token_usage = conversation_instance._primary_query(tools=[mock_tool_one])

    # If json.loads fails for a model prompted for tool use (but doesn't support it),
    # the behavior is now to treat the response as a regular message.
    assert msg == raw_llm_response_content
    assert token_usage == mock_llm_response.usage_metadata["total_tokens"]
    # Ensure the manual tool call processing was not (successfully) called
    conversation_instance._process_manual_tool_call.assert_not_called()
    # Note: append_ai_message is NOT called for invalid JSON responses in the current implementation


def test_primary_query_structured_output_model_supports(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "gemini-2.0-flash"  # Assumed in STRUCTURED_OUTPUT_MODELS
    initial_messages = list(conversation_instance.messages)

    structured_response_obj = MockOutputModel(param1="Structured data", param2=100)
    # For structured output, invoke returns the Pydantic model instance directly
    mock_chat_object.invoke = MagicMock(return_value=structured_response_obj)

    with patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", [conversation_instance.model_name]):
        msg, token_usage = conversation_instance._primary_query(
            structured_model=MockOutputModel,  # Pass the class
            wrap_structured_output=False,
        )

    expected_json_output = structured_response_obj.model_dump_json()
    assert msg == expected_json_output
    assert token_usage == 0  # Token usage is 0 for structured outputs (can't count tokens but not an error)

    mock_chat_object.with_structured_output.assert_called_once_with(MockOutputModel)
    structured_chat_mock = mock_chat_object.with_structured_output.return_value
    structured_chat_mock.invoke.assert_called_once_with(initial_messages)
    conversation_instance.append_ai_message.assert_called_once_with(expected_json_output)


def test_primary_query_structured_output_model_supports_wrapped(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "gemini-2.0-flash"  # Assumed in STRUCTURED_OUTPUT_MODELS
    initial_messages = list(conversation_instance.messages)
    structured_response_obj = MockOutputModel(param1="Wrapped data", param2=200)
    mock_chat_object.invoke = MagicMock(return_value=structured_response_obj)

    with patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", [conversation_instance.model_name]):
        msg, token_usage = conversation_instance._primary_query(
            structured_model=MockOutputModel, wrap_structured_output=True
        )

    unwrapped_json = structured_response_obj.model_dump_json()
    expected_wrapped_output = "```json\n" + unwrapped_json + "\n```"
    assert msg == expected_wrapped_output
    assert token_usage == 0  # Token usage is 0 for structured outputs (can't count tokens but not an error)
    mock_chat_object.with_structured_output.assert_called_once_with(MockOutputModel)
    mock_chat_object.with_structured_output.return_value.invoke.assert_called_once_with(initial_messages)
    conversation_instance.append_ai_message.assert_called_once_with(expected_wrapped_output)


def test_primary_query_structured_output_model_not_supports_fallback(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "basic-model"  # Not in STRUCTURED_OUTPUT_MODELS
    initial_human_message = conversation_instance.messages[-1]
    assert isinstance(initial_human_message, HumanMessage)  # Ensure it's HumanMessage
    user_query_content = initial_human_message.content

    structured_response_obj = MockOutputModel(param1="Fallback data", param2=300)
    mock_chat_object.invoke = MagicMock(return_value=structured_response_obj)
    schema_json_str = str(MockOutputModel.model_json_schema())

    with (
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
    ):  # Ensure not tool calling either
        msg, token_usage = conversation_instance._primary_query(
            structured_model=MockOutputModel, wrap_structured_output=False
        )

    expected_json_output = structured_response_obj.model_dump_json()
    assert msg == expected_json_output
    assert token_usage == 0  # Token usage is 0 for structured outputs (can't count tokens but not an error)

    expected_modified_content = (
        user_query_content
        + "\n\nPlease return a structured output following this schema: "
        + schema_json_str
        + " Just return the JSON object and nothing else."
    )

    # Check that the message passed to invoke was modified
    args, _ = mock_chat_object.invoke.call_args
    invoked_messages = args[0]
    assert len(invoked_messages) == 1
    assert isinstance(invoked_messages[0], HumanMessage)  # Type should be preserved
    assert invoked_messages[0].content == expected_modified_content

    mock_chat_object.with_structured_output.assert_not_called()
    conversation_instance.append_ai_message.assert_called_once_with(expected_json_output)


def test_primary_query_structured_output_model_not_supports_fallback_wrapped(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "basic-model"
    user_query_content = conversation_instance.messages[-1].content

    structured_response_obj = MockOutputModel(param1="Fallback wrapped", param2=400)
    mock_chat_object.invoke = MagicMock(return_value=structured_response_obj)
    schema_json_str = str(MockOutputModel.model_json_schema())

    with (
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
    ):
        msg, token_usage = conversation_instance._primary_query(
            structured_model=MockOutputModel,
            wrap_structured_output=True,  # Key change
        )

    unwrapped_json = structured_response_obj.model_dump_json()
    expected_wrapped_output = "```json\n" + unwrapped_json + "\n```"
    assert msg == expected_wrapped_output
    assert token_usage == 0  # Token usage is 0 for structured outputs (can't count tokens but not an error)

    expected_modified_content = (
        user_query_content
        + "\n\nPlease return a structured output following this schema: "
        + schema_json_str
        + " Just return the JSON object wrapped in ```json tags and nothing else."  # Wording change
    )
    args, _ = mock_chat_object.invoke.call_args
    invoked_messages = args[0]
    assert invoked_messages[-1].content == expected_modified_content
    conversation_instance.append_ai_message.assert_called_once_with(expected_wrapped_output)


def test_primary_query_error_tools_and_structured_output(conversation_instance):
    conversation_instance.model_name = "any-model"
    with pytest.raises(ValueError, match="Structured output and tools cannot be used together"):
        conversation_instance._primary_query(tools=[mock_tool_one], structured_model=MockOutputModel)


def test_primary_query_invoke_raises_exception(conversation_instance, mock_chat_object):
    conversation_instance.model_name = "basic-model"
    error_message = "API communication failed spectacularly"
    # Make invoke on the main chat object (not bound/structured ones) raise error
    conversation_instance.chat.invoke = MagicMock(side_effect=Exception(error_message))

    with (
        patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []),
        patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", []),
    ):
        msg, token_usage = conversation_instance._primary_query()

    assert msg == str(Exception(error_message))  # Method returns str(e)
    assert token_usage is None  # Token usage is None when there's an exception
    conversation_instance.append_ai_message.assert_not_called()
