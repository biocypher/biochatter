"""Tests for the OpenAI LLM connect module."""
import os
from unittest.mock import Mock, patch

import openai
import pytest
from openai._exceptions import NotFoundError

from biochatter.llm_connect import (
    AIMessage,
    AzureGptConversation,
    GptConversation,
    HumanMessage,
    SystemMessage,
)


@pytest.fixture(scope="module", autouse=True)
def manage_test_context():
    import openai

    base_url = openai.base_url
    api_type = openai.api_type
    api_version = openai.api_version
    api_key = openai.api_key
    organization = openai.organization
    proxy = getattr(openai, "proxy", None)
    yield True

    openai.base_url = base_url
    openai.api_type = api_type
    openai.api_version = api_version
    openai.api_key = api_key
    openai.organization = organization
    if proxy is not None:
        openai.proxy = proxy
    elif hasattr(openai, "proxy"):
        delattr(openai, "proxy")


def test_empty_messages():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    assert convo.get_msg_json() == "[]"


def test_single_message():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    assert convo.get_msg_json() == '[{"system": "Hello, world!"}]'


def test_multiple_messages():
    convo = GptConversation(
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
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(TypeError):
        convo.get_msg_json()


@patch("biochatter.llm_connect.openai.OpenAI")
def test_openai_catches_authentication_error(mock_openai):
    mock_openai.return_value.models.list.side_effect = openai._exceptions.AuthenticationError(
        (
            "Incorrect API key provided: fake_key. You can find your API key"
            " at https://platform.openai.com/account/api-keys."
        ),
        response=Mock(),
        body=None,
    )
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="fake_key",
        user="test_user",
    )

    assert not success


@patch("biochatter.llm_connect.AzureChatOpenAI")
def test_azure_raises_request_error(mock_azure_chat):
    mock_azure_chat.side_effect = NotFoundError(
        message="Resource not found",
        response=Mock(),
        body=None,
    )

    convo = AzureGptConversation(
        model_name="gpt-35-turbo",
        deployment_name="test_deployment",
        prompts={},
        split_correction=False,
        version="2023-03-15-preview",
        base_url="https://api.openai.com",
    )

    with pytest.raises(NotFoundError):
        convo.set_api_key("fake_key")


@patch("biochatter.llm_connect.AzureChatOpenAI")
def test_azure(mock_azure_chat):
    """Test OpenAI Azure endpoint functionality.

    Azure connectivity is enabled by setting the corresponding environment
    variables.
    """
    convo = AzureGptConversation(
        model_name=os.getenv("AZURE_TEST_OPENAI_MODEL_NAME"),
        deployment_name=os.getenv("AZURE_TEST_OPENAI_DEPLOYMENT_NAME"),
        prompts={},
        split_correction=False,
        version=os.getenv("AZURE_TEST_OPENAI_API_VERSION"),
        base_url=os.getenv("AZURE_TEST_OPENAI_API_BASE"),
    )

    mock_azure_chat.return_value = Mock()

    assert convo.set_api_key(os.getenv("AZURE_TEST_OPENAI_API_KEY"))


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_local_image_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    convo.append_image_message(
        message=(
            "This text describes the attached image: "
            "Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads."
        ),
        image_url="test/figure_panel.jpg",
        local=True,
    )

    result, _, _ = convo.query("Is the description accurate?")
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_local_image_query_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    result, _, _ = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="test/figure_panel.jpg",
    )
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_online_image_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_image_message(
        "This is a picture from the internet.",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )

    result, _, _ = convo.query("What does this picture show?")
    assert "transformer" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_online_image_query_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    result, _, _ = convo.query(
        "What does this picture show?",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )
    assert "transformer" in result.lower()


def test_chat_attribute_not_initialized():
    """Test that accessing chat before initialization raises AttributeError."""
    convo = GptConversation(
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
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    with pytest.raises(AttributeError) as exc_info:
        _ = convo.ca_chat

    assert "Correcting agent chat attribute not initialized" in str(exc_info.value)
    assert "Did you call set_api_key()?" in str(exc_info.value)


@patch("biochatter.llm_connect.openai.OpenAI")
def test_chat_attributes_reset_on_auth_error(mock_openai):
    """Test that chat attributes are reset to None on authentication error."""
    mock_openai.return_value.models.list.side_effect = openai._exceptions.AuthenticationError(
        "Invalid API key",
        response=Mock(),
        body=None,
    )

    convo = GptConversation(
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


@pytest.mark.skip(reason="Test depends on langchain-openai implementation which needs to be updated")
@patch("biochatter.llm_connect.openai.OpenAI")
def test_chat_attributes_set_on_success(mock_openai):
    """Test that chat attributes are properly set when authentication succeeds.

    This test is skipped because it depends on the langchain-openai
    implementation which needs to be updated. Fails in CI with:
        __pydantic_self__ = ChatOpenAI()
            data = {'base_url': None, 'model_kwargs': {}, 'model_name': 'gpt-3.5-turbo', 'openai_api_key': 'fake_key', ...}
            values = {'async_client': None, 'cache': None, 'callback_manager': None, 'callbacks': None, ...}
            fields_set = {'model_kwargs', 'model_name', 'openai_api_base', 'openai_api_key', 'temperature'}
            validation_error = ValidationError(model='ChatOpenAI', errors=[{'loc': ('__root__',), 'msg': "AsyncClient.__init__() got an unexpected keyword argument 'proxies'", 'type': 'type_error'}])
                def __init__(__pydantic_self__, **data: Any) -> None:
                    # Uses something other than `self` the first arg to allow "self" as a settable attribute
                    values, fields_set, validation_error = validate_model(__pydantic_self__.__class__, data)
                    if validation_error:
            >           raise validation_error
            E           pydantic.v1.error_wrappers.ValidationError: 1 validation error for ChatOpenAI
            E           __root__
            E             AsyncClient.__init__() got an unexpected keyword argument 'proxies' (type=type_error)
            ../../../.cache/pypoetry/virtualenvs/biochatter-f6F-uYko-py3.11/lib/python3.11/site-packages/pydantic/v1/main.py:341: ValidationError
    """
    # Mock successful authentication
    mock_openai.return_value.models.list.return_value = ["gpt-3.5-turbo"]

    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    # Set API key (which will succeed)
    success = convo.set_api_key(api_key="fake_key")

    assert success

    # Verify both chat attributes are accessible
    assert convo.chat is not None
    assert convo.ca_chat is not None


def test_gpt_update_usage_stats():
    """Test the _update_usage_stats method in GptConversation."""
    # Arrange
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
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

    model = "gpt-3.5-turbo"
    token_usage = {
        "prompt_tokens": 50,
        "completion_tokens": 30,
        "total_tokens": 80,
        "non_numeric_field": "should be ignored",
        "nested_dict": {  # Should be ignored as it's a dictionary
            "sub_field": 100,
            "another_field": 200,
        },
        "another_field": "also ignored",
    }

    # Act
    convo._update_usage_stats(model, token_usage)

    # Assert
    # Verify increment was called with correct arguments for community stats
    # Only numeric values at the top level should be included
    mock_usage_stats.increment.assert_called_once_with(
        "usage:[date]:[user]",
        {
            "prompt_tokens:gpt-3.5-turbo": 50,
            "completion_tokens:gpt-3.5-turbo": 30,
            "total_tokens:gpt-3.5-turbo": 80,
        },
    )

    # Verify callback was called with complete token_usage including nested dict
    mock_update_callback.assert_called_once_with(
        "community",
        "gpt-3.5-turbo",
        token_usage,  # Full dictionary including nested values
    )
