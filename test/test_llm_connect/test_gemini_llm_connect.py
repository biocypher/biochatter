"""Tests for the Gemini LLM connect module."""

import os
from unittest.mock import Mock, patch

import pytest
from google.api_core.exceptions import InvalidArgument

from biochatter.llm_connect import GeminiConversation

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool


@pytest.fixture(scope="module", autouse=True)
def manage_test_context():
    import dotenv
    import os
    import pathlib

    # Get the directory containing this script
    script_dir = pathlib.Path(__file__).parent.parent.absolute()

    # Change working directory to the script directory
    original_dir = os.getcwd()
    os.chdir(script_dir)

    # Load environment variables
    dotenv.load_dotenv()

    # Yield to allow tests to run
    yield

    # Restore original working directory after tests
    os.chdir(original_dir)


def test_empty_messages():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )
    assert convo.get_msg_json() == "[]"


def test_single_message():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    assert convo.get_msg_json() == '[{"system": "Hello, world!"}]'


def test_multiple_messages():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
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
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(TypeError):
        convo.get_msg_json()


@patch("biochatter.llm_connect.gemini.ChatGoogleGenerativeAI")
def test_gemini_catches_authentication_error(mock_gemini):
    mock_gemini.side_effect = InvalidArgument("Invalid API key")

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="fake_key",
        user="test_user",
    )

    assert not success


def test_chat_attribute_not_initialized():
    """Test that accessing chat before initialization raises AttributeError."""
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.chat

    assert "Chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


def test_ca_chat_attribute_not_initialized():
    """Test that accessing ca_chat before initialization raises AttributeError."""
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.ca_chat

    assert "Correcting agent chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


@patch("biochatter.llm_connect.gemini.ChatGoogleGenerativeAI")
def test_chat_attributes_reset_on_auth_error(mock_gemini):
    """Test that chat attributes are reset to None on authentication error."""
    mock_gemini.side_effect = InvalidArgument("Invalid API key")

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
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


@patch("biochatter.llm_connect.gemini.ChatGoogleGenerativeAI")
def test_chat_attributes_set_on_success(mock_gemini):
    """Test that chat attributes are properly set when authentication succeeds."""
    # Mock successful authentication
    mock_gemini.return_value = Mock()

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
    )

    # Set API key (which will succeed)
    success = convo.set_api_key(api_key="fake_key")
    assert success

    # Verify both chat attributes are accessible
    assert convo.chat is not None
    assert convo.ca_chat is not None


@pytest.mark.skip(reason="Not yet implemented community usage for Gemini")
def test_gemini_update_usage_stats():
    """Test the _update_usage_stats method in GeminiConversation."""
    # Arrange
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
    )

    # Mock the usage_stats object
    mock_usage_stats = Mock()
    convo.usage_stats = mock_usage_stats
    convo.user = "community"  # Set user to enable stats tracking

    # Mock the update_token_usage callback
    mock_update_callback = Mock()
    convo._update_token_usage = mock_update_callback

    model = "gemini-2.0-flash"
    token_usage = {
        "total_tokens": 80,
    }

    # Act
    convo._update_usage_stats(model, token_usage)

    # Assert
    # Verify increment was called with correct arguments for community stats
    mock_usage_stats.increment.assert_called_once_with(
        "usage:[date]:[user]",
        {
            "total_tokens:gemini-2.0-flash": 80,
        },
    )

    # Verify callback was called with complete token_usage
    mock_update_callback.assert_called_once_with(
        "community",
        "gemini-2.0-flash",
        token_usage,
    )


@pytest.mark.skip(reason="Live test for development purposes")
def test_gemini_default():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"))

    result, _, _ = convo.query("What is the capital of France?")
    result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_local_image_gemini():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    convo.append_image_message(
        message=(
            "This text describes the attached image: "
            "Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads."
        ),
        image_url="figure_panel.jpg",
        local=True,
    )

    result, _, _ = convo.query("Is the description accurate?")
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_local_image_query_gemini():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    result, _, _ = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="figure_panel.jpg",
    )
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_online_image_gemini():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.append_image_message(
        "This is a picture from the internet.",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )

    result, _, _ = convo.query("What does this picture show?")
    assert "transformer" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_online_image_query_gemini():
    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

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
def test_tool_message_auto():
    multiply, _ = create_tool_functions()

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
        tools=[multiply],
        tool_call_mode="auto",
    )

    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.query("What is 2 times 2?")
    assert "4" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
def test_multiple_tool_calls_auto():
    multiply, add = create_tool_functions()

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
        tools=[multiply, add],
        tool_call_mode="auto",
    )

    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.query("What is 2 times 3? and what about 2 plus 3?")
    assert "6" in convo.messages[-2].content
    assert "5" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
def test_tool_message_text():
    multiply, _ = create_tool_functions()

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
        tools=[multiply],
        tool_call_mode="text",
    )

    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    convo.query("What is 2 times 2?")

    # In text mode, the tool call should be formatted as text rather than executed
    # Check that the last message contains the tool name and arguments
    assert "Tool: multiply" in convo.messages[-1].content
    assert "Arguments:" in convo.messages[-1].content
    assert "Tool call id:" in convo.messages[-1].content

@pytest.mark.skip(reason="Live test for development purposes")
def test_multiple_tool_calls_text_mode():
    multiply, add = create_tool_functions()

    convo = GeminiConversation(
        model_name="gemini-2.0-flash",
        prompts={},
        split_correction=False,
        tools=[multiply, add],
        tool_call_mode="text",
    )

    convo.set_api_key(api_key=os.getenv("GOOGLE_API_KEY"), user="test_user")

    # Query that should trigger multiple tool calls
    convo.query("What is 2 times 3? and what about 2 plus 3?")

    # In text mode, the tool calls should be formatted as text rather than executed
    # Check that the last message contains both tool names and their arguments
    assert "Tool: multiply" in convo.messages[-1].content
    assert "Tool: add" in convo.messages[-1].content

    # Verify that the arguments are present
    assert "Arguments:" in convo.messages[-1].content

    # Verify that tool call IDs are included
    assert "Tool call id:" in convo.messages[-1].content

    # Count the number of tool calls by counting occurrences of "Tool:"
    tool_call_count = convo.messages[-1].content.count("Tool:")
    assert tool_call_count >= 2, "Expected at least two tool calls in text mode"
