import os
import openai
from openai._exceptions import NotFoundError
from xinference.client import Client
from biochatter.llm_connect import (
    GptConversation,
    AzureGptConversation,
    SystemMessage,
    HumanMessage,
    AIMessage,
    XinferenceConversation,
    WasmConversation,
)
import pytest
from unittest.mock import patch, Mock


@pytest.fixture(scope="module", autouse=True)
def manageTestContext():
    import openai

    base_url = openai.base_url
    api_type = openai.api_type
    api_version = openai.api_version
    api_key = openai.api_key
    # api_key_path = openai.api_key_path
    organization = openai.organization
    yield True

    openai.base_url = base_url
    openai.api_type = api_type
    openai.api_version = api_version
    openai.api_key = api_key
    # openai.api_key_path = api_key_path
    openai.organization = organization


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
        '[{"system": "Hello, world!"}, '
        '{"user": "How are you?"}, '
        '{"ai": "I\'m doing well, thanks!"}]'
    )


def test_unknown_message_type():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(ValueError):
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


def test_azure_raises_request_error():
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
def test_azure(mock_azurechat):
    """
    Test OpenAI Azure endpoint functionality. Azure connectivity is enabled by
    setting the corresponding environment variables.
    """
    # mock_azurechat.return_value.generate =
    openai.proxy = os.getenv("AZURE_TEST_OPENAI_PROXY")
    convo = AzureGptConversation(
        model_name=os.getenv("AZURE_TEST_OPENAI_MODEL_NAME"),
        deployment_name=os.getenv("AZURE_TEST_OPENAI_DEPLOYMENT_NAME"),
        prompts={},
        split_correction=False,
        version=os.getenv("AZURE_TEST_OPENAI_API_VERSION"),
        base_url=os.getenv("AZURE_TEST_OPENAI_API_BASE"),
    )

    assert convo.set_api_key(os.getenv("AZURE_TEST_OPENAI_API_KEY"))


xinference_models = {
    "48c76b62-904c-11ee-a3d2-0242acac0302": {
        "model_type": "embedding",
        "address": "",
        "accelerators": ["0"],
        "model_name": "gte-large",
        "dimensions": 1024,
        "max_tokens": 512,
        "language": ["en"],
        "model_revision": "",
    },
    "a823319a-88bd-11ee-8c78-0242acac0302": {
        "model_type": "LLM",
        "address": "0.0.0.0:46237",
        "accelerators": ["0"],
        "model_name": "llama2-13b-chat-hf",
        "model_lang": ["en"],
        "model_ability": ["embed", "generate", "chat"],
        "model_format": "pytorch",
        "context_length": 4096,
    },
}


def test_xinference_init():
    """
    Test generic LLM connectivity via the Xinference client. Currently depends
    on a test server.
    """
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://llm.biocypher.org")
    with patch("xinference.client.Client") as mock_client:
        mock_client.return_value.list_models.return_value = xinference_models
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            split_correction=False,
        )
        assert convo.set_api_key()


def test_generic_chatting():
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://llm.biocypher.org")
    with patch("xinference.client.Client") as mock_client:
        response = {
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "foo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " Hello there, can you sing me a song?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 93,
                "completion_tokens": 54,
                "total_tokens": 147,
            },
        }
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = (
            response
        )
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            correct=False,
        )
        (msg, token_usage, correction) = convo.query("Hello, world!")
        assert token_usage["completion_tokens"] > 0


def test_wasm_conversation():
    # Initialize the class
    wasm_convo = WasmConversation(
        model_name="test_model",
        prompts={},
        correct=True,
        split_correction=False,
    )

    # Check if the model_name is correctly set
    assert wasm_convo.model_name == "test_model"

    # Check if the prompts are correctly set
    assert wasm_convo.prompts == {}

    # Check if the correct is correctly set
    assert wasm_convo.correct == True

    # Check if the split_correction is correctly set
    assert wasm_convo.split_correction == False

    # Test the query method
    test_query = "Hello, world!"
    result, _, _ = wasm_convo.query(test_query)
    assert result == test_query  # assuming the messages list is initially empty

    # Test the _primary_query method, add another message to the messages list
    wasm_convo.append_system_message("System message")
    result = wasm_convo._primary_query()
    assert result == test_query + "\nSystem message"


@pytest.fixture
def xinference_conversation():
    with patch("xinference.client.Client") as mock_client:
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = (
            {"choices": [{"message": {"content": "Human message"}}]},
            {"completion_tokens": 0},
        )
        conversation = XinferenceConversation(
            base_url="http://llm.biocypher.org",
            prompts={},
            correct=False,
        )
    return conversation


def test_single_system_message_before_human(xinference_conversation):
    xinference_conversation.messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


def test_multiple_system_messages_before_human(xinference_conversation):
    xinference_conversation.messages = [
        SystemMessage(content="System message 1"),
        SystemMessage(content="System message 2"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message 1\nSystem message 2\nHuman message",
    }


def test_multiple_messages_including_ai_before_system_and_human(
    xinference_conversation,
):
    xinference_conversation.messages = [
        HumanMessage(content="Human message history"),
        AIMessage(content="AI message"),
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


def test_multiple_cycles_of_ai_and_human(xinference_conversation):
    xinference_conversation.messages = [
        HumanMessage(content="Human message history"),
        AIMessage(content="AI message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert len(history) == 3
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }
