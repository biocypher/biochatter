"""Tests for the OpenRouter LLM connect module."""

import os
from unittest.mock import Mock, patch

import pytest

from biochatter.llm_connect.openrouter import ChatOpenRouter, OpenRouterConversation

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool


# Tests for ChatOpenRouter class
def test_chatopenrouter_initialization():
    """Test ChatOpenRouter initialization with explicit API key."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenAI.__init__") as mock_init:
        mock_init.return_value = None

        chat = ChatOpenRouter(openai_api_key="test_key", model_name="gpt-3.5-turbo")

        mock_init.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", openai_api_key="test_key", model_name="gpt-3.5-turbo"
        )


def test_chatopenrouter_environment_variable():
    """Test ChatOpenRouter uses environment variable when no API key provided."""
    with (
        patch("os.environ.get") as mock_env,
        patch("biochatter.llm_connect.openrouter.ChatOpenAI.__init__") as mock_init,
    ):
        mock_env.return_value = "env_api_key"
        mock_init.return_value = None

        chat = ChatOpenRouter(model_name="gpt-4")

        mock_env.assert_called_with("OPENROUTER_API_KEY")
        mock_init.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", openai_api_key="env_api_key", model_name="gpt-4"
        )


def test_chatopenrouter_lc_secrets():
    """Test ChatOpenRouter lc_secrets property."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenAI.__init__"):
        chat = ChatOpenRouter()
        assert chat.lc_secrets == {"openai_api_key": "OPENROUTER_API_KEY"}


def test_chatopenrouter_base_url():
    """Test ChatOpenRouter sets correct base URL."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenAI.__init__") as mock_init:
        mock_init.return_value = None

        chat = ChatOpenRouter(openai_api_key="test_key")

        # Verify the base_url is set to OpenRouter's API endpoint
        call_args = mock_init.call_args
        assert call_args[1]["base_url"] == "https://openrouter.ai/api/v1"


# Tests for OpenRouterConversation class
def test_empty_messages():
    """Test that an empty conversation returns empty JSON."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    assert convo.get_msg_json() == "[]"


def test_single_message():
    """Test that a single system message is properly formatted."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    assert convo.get_msg_json() == '[{"system": "Hello, world!"}]'


def test_multiple_messages():
    """Test that multiple messages are properly formatted."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    convo.messages.append(HumanMessage(content="How are you?"))
    convo.messages.append(AIMessage(content="I'm doing well, thanks!"))
    assert convo.get_msg_json() == (
        '[{"system": "Hello, world!"}, {"user": "How are you?"}, {"ai": "I\'m doing well, thanks!"}]'
    )


def test_unknown_message_type():
    """Test that unknown message types raise TypeError."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(TypeError):
        convo.get_msg_json()


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_openrouter_catches_authentication_error(mock_openrouter):
    """Test that OpenRouter catches authentication errors during initialization."""
    mock_openrouter.side_effect = Exception("Invalid API key")

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="fake_key",
        user="test_user",
    )

    assert not success


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_openrouter_set_api_key_success(mock_openrouter):
    """Test successful API key setting."""
    mock_chat_instance = Mock()
    mock_openrouter.return_value = mock_chat_instance

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="valid_key",
        user="test_user",
    )

    assert success
    assert convo.user == "test_user"
    assert convo.chat == mock_chat_instance
    assert convo.ca_chat == mock_chat_instance


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_openrouter_set_api_key_with_tools(mock_openrouter):
    """Test API key setting with tools binds them properly."""
    mock_chat_instance = Mock()
    mock_openrouter.return_value = mock_chat_instance

    @tool
    def test_tool(x: int) -> int:
        """A test tool."""
        return x * 2

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
        tools=[test_tool],
    )

    convo.bind_tools = Mock()  # Mock the bind_tools method

    success = convo.set_api_key(
        api_key="valid_key",
        user="test_user",
    )

    assert success
    convo.bind_tools.assert_called_once_with([test_tool])


def test_chat_attribute_not_initialized():
    """Test that accessing chat before initialization raises AttributeError."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.chat

    assert "Chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


def test_ca_chat_attribute_not_initialized():
    """Test that accessing ca_chat before initialization raises AttributeError."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.ca_chat

    assert "Correcting agent chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_chat_attributes_reset_on_auth_error(mock_openrouter):
    """Test that chat attributes are reset to None on authentication error."""
    mock_openrouter.side_effect = Exception("Authentication failed")

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    # Set API key (which will fail)
    success = convo.set_api_key(api_key="fake_key")
    assert not success

    # Verify both chat attributes are None
    with pytest.raises(AttributeError):
        _ = convo.chat
    with pytest.raises(AttributeError):
        _ = convo.ca_chat


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_chat_attributes_set_on_success(mock_openrouter):
    """Test that chat attributes are properly set when authentication succeeds."""
    mock_chat_instance = Mock()
    mock_openrouter.return_value = mock_chat_instance

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    # Set API key (which will succeed)
    success = convo.set_api_key(api_key="valid_key")
    assert success

    # Verify both chat attributes are accessible
    assert convo.chat is not None
    assert convo.ca_chat is not None


def test_openrouter_inherits_from_langchain_conversation():
    """Test that OpenRouterConversation properly inherits from LangChainConversation."""
    from biochatter.llm_connect import LangChainConversation

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    assert isinstance(convo, LangChainConversation)
    assert convo.model_name == "gpt-3.5-turbo"
    assert convo.model_provider == ""  # OpenRouter uses empty string for provider


def test_openrouter_initialization_parameters():
    """Test that OpenRouterConversation accepts and handles initialization parameters."""
    prompts = {"system": "You are a helpful assistant"}

    convo = OpenRouterConversation(
        model_name="gpt-4",
        prompts=prompts,
        correct=True,
        split_correction=True,
        tools=[],
    )

    assert convo.model_name == "gpt-4"
    assert convo.prompts == prompts
    assert convo.correct == True
    assert convo.split_correction == True
    assert convo.tools == []


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_openrouter_api_key_environment_variable(mock_openrouter):
    """Test that OpenRouter uses environment variable for API key."""
    mock_chat_instance = Mock()
    mock_openrouter.return_value = mock_chat_instance

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key()  # No explicit API key provided

    # Should still try to initialize (would use env var in real implementation)
    mock_openrouter.assert_called()


def test_openrouter_model_name_passed_to_chatopenrouter():
    """Test that model name is properly passed to ChatOpenRouter."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenRouter") as mock_openrouter:
        mock_chat_instance = Mock()
        mock_openrouter.return_value = mock_chat_instance

        convo = OpenRouterConversation(
            model_name="claude-3-opus",
            prompts={},
            split_correction=False,
        )

        convo.set_api_key("valid_key")

        # Verify ChatOpenRouter was called with correct model name
        mock_openrouter.assert_called_with(
            model_name="claude-3-opus",
            temperature=0,
        )


def test_openrouter_temperature_setting():
    """Test that temperature is set to 0 for deterministic responses."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenRouter") as mock_openrouter:
        mock_chat_instance = Mock()
        mock_openrouter.return_value = mock_chat_instance

        convo = OpenRouterConversation(
            model_name="gpt-3.5-turbo",
            prompts={},
            split_correction=False,
        )

        convo.set_api_key("valid_key")

        # Verify temperature is set to 0
        mock_openrouter.assert_called_with(
            model_name="gpt-3.5-turbo",
            temperature=0,
        )


@patch("biochatter.llm_connect.openrouter.ChatOpenRouter")
def test_openrouter_usage_stats_tracking(mock_openrouter):
    """Test that usage statistics are properly tracked when user is 'community'."""
    mock_chat_instance = Mock()
    mock_openrouter.return_value = mock_chat_instance

    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="valid_key",
        user="community",
    )

    assert success
    assert convo.user == "community"


def test_openrouter_exception_handling_details():
    """Test that exception handling properly resets chat attributes."""
    with patch("biochatter.llm_connect.openrouter.ChatOpenRouter") as mock_openrouter:
        mock_openrouter.side_effect = ValueError("Connection timeout")

        convo = OpenRouterConversation(
            model_name="gpt-3.5-turbo",
            prompts={},
            split_correction=False,
        )

        success = convo.set_api_key("failing_key")

        assert not success
        # Verify that the private attributes are set to None
        assert convo._chat is None
        assert convo._ca_chat is None


# Live tests (skipped by default)
@pytest.mark.skip(reason="Live test for development purposes")
def test_openrouter_live_query():
    """Live test for OpenRouter API (requires valid API key)."""
    convo = OpenRouterConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        correct=False,
        split_correction=False,
    )

    success = convo.set_api_key(api_key=os.getenv("OPENROUTER_API_KEY"))
    assert success

    result, _, _ = convo.query("What is the capital of France?")
    assert "paris" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_openrouter_image_message():
    """Live test for OpenRouter with image messages (requires valid API key)."""
    convo = OpenRouterConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )

    success = convo.set_api_key(api_key=os.getenv("OPENROUTER_API_KEY"))
    assert success

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    result, _, _ = convo.query(
        "What does this picture show?",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )
    assert "transformer" in result.lower()


def create_tool_functions():
    """Create and return the tool functions used in tests."""

    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int

    @tool
    def add(first_int: int, second_int: int) -> int:
        """Add two integers together."""
        return first_int + second_int

    return multiply, add


@pytest.mark.skip(reason="Live test for development purposes")
def test_openrouter_tool_calling():
    """Live test for OpenRouter with tool calling (requires valid API key)."""
    multiply, _ = create_tool_functions()

    convo = OpenRouterConversation(
        model_name="gpt-4-turbo",
        prompts={},
        split_correction=False,
        tools=[multiply],
        tool_call_mode="auto",
    )

    success = convo.set_api_key(api_key=os.getenv("OPENROUTER_API_KEY"))
    assert success

    convo.query("What is 2 times 3?")
    # Check if tool was called and result is in conversation
    assert any("6" in str(msg.content) for msg in convo.messages if hasattr(msg, "content"))
