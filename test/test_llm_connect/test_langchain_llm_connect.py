"""Tests for the LangChain LLM connect module."""

import pytest

from biochatter.llm_connect import LangChainConversation


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool


MODEL_PARAMS = [
    ("google_genai", "gemini-2.0-flash"),
    ("openai", "gpt-4o"),
    ("mistralai", "mistral-large-latest"),
    ("anthropic", "claude-3-7-sonnet-latest"),
]


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_empty_messages(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )
    assert convo.get_msg_json() == "[]"


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_single_message(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    assert convo.get_msg_json() == '[{"system": "Hello, world!"}]'


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_multiple_messages(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    convo.messages.append(HumanMessage(content="How are you?"))
    convo.messages.append(AIMessage(content="I'm doing well, thanks!"))
    assert convo.get_msg_json() == (
        '[{"system": "Hello, world!"}, {"user": "How are you?"}, {"ai": "I\'m doing well, thanks!"}]'
    )


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_unknown_message_type(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(TypeError):
        convo.get_msg_json()


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_chat_attribute_not_initialized(model_provider, model_name):
    """Test that accessing chat before initialization raises AttributeError."""
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.chat

    assert "Chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_ca_chat_attribute_not_initialized(model_provider, model_name):
    """Test that accessing ca_chat before initialization raises AttributeError."""
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.ca_chat

    assert "Correcting agent chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_langchain_default(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key()

    result, _, _ = convo.query("What is the capital of France?")
    result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_append_local_image_gemini(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key()

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
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_local_image_query_gemini(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key()

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    result, _, _ = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="figure_panel.jpg",
    )
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_append_online_image_gemini(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key()

    convo.append_image_message(
        "This is a picture from the internet.",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )

    result, _, _ = convo.query("What does this picture show?")
    assert "transformer" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_online_image_query_gemini(model_provider, model_name):
    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key()

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
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_tool_message_auto(model_provider, model_name):
    multiply, _ = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
        tools=[multiply],
        tool_call_mode="auto",
    )

    convo.set_api_key()

    convo.query("What is 2 times 2?")
    assert "4" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_multiple_tool_calls_auto(model_provider, model_name):
    multiply, add = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
        tools=[multiply, add],
        tool_call_mode="auto",
    )

    convo.set_api_key()

    convo.query("What is 2 times 3? and what about 2 plus 3?")
    assert "6" in convo.messages[-2].content
    assert "5" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_tool_auto_message_passed_to_query(model_provider, model_name):
    multiply, _ = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
    )

    convo.set_api_key()

    convo.query("What is 2 times 3?", tools=[multiply])
    assert "6" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_tool_message_text(model_provider, model_name):
    multiply, _ = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
        tools=[multiply],
        tool_call_mode="text",
    )

    convo.set_api_key()

    convo.query("What is 2 times 2?")

    # In text mode, the tool call should be formatted as text rather than executed
    # Check that the last message contains the tool name and arguments
    assert "Tool: multiply" in convo.messages[-1].content
    assert "Arguments:" in convo.messages[-1].content
    assert "Tool call id:" in convo.messages[-1].content


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_multiple_tool_calls_text_mode(model_provider, model_name):
    multiply, add = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        split_correction=False,
        tools=[multiply, add],
        tool_call_mode="text",
    )

    convo.set_api_key()

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


@pytest.mark.skip(reason="Live test for development purposes")
@pytest.mark.parametrize("model_provider, model_name", MODEL_PARAMS)
def test_tool_text_message_passed_to_query(model_provider, model_name):
    multiply, _ = create_tool_functions()

    convo = LangChainConversation(
        model_provider=model_provider,
        model_name=model_name,
        prompts={},
        tool_call_mode="text",
    )

    convo.set_api_key()

    convo.query("What is 2 times 3?", tools=[multiply])
    assert "Tool: multiply" in convo.messages[-1].content
    assert "Arguments:" in convo.messages[-1].content
    assert "Tool call id:" in convo.messages[-1].content
