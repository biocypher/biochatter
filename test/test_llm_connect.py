import os
import openai
from biochatter.llm_connect import (
    GptConversation,
    AzureGptConversation,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from openai.error import InvalidRequestError
import pytest


@pytest.fixture(scope="module", autouse=True)
def manageTestContext():
    import openai

    api_base = openai.api_base
    api_type = openai.api_type
    api_version = openai.api_version
    api_key = openai.api_key
    api_key_path = openai.api_key_path
    organization = openai.organization
    yield True

    openai.api_base = api_base
    openai.api_type = api_type
    openai.api_version = api_version
    openai.api_key = api_key
    openai.api_key_path = api_key_path
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


def test_openai_catches_authentication_error():
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
        base="https://api.openai.com",
    )

    with pytest.raises(InvalidRequestError):
        convo.set_api_key("fake_key")


def test_azure():
    """
    Test OpenAI Azure endpoint functionality. Azure connectivity is enabled by
    setting the corresponding environment variables.
    """
    openai.proxy = os.getenv("AZURE_TEST_OPENAI_PROXY")
    convo = AzureGptConversation(
        model_name=os.getenv("AZURE_TEST_OPENAI_MODEL_NAME"),
        deployment_name=os.getenv("AZURE_TEST_OPENAI_DEPLOYMENT_NAME"),
        prompts={},
        split_correction=False,
        version=os.getenv("AZURE_TEST_OPENAI_API_VERSION"),
        base=os.getenv("AZURE_TEST_OPENAI_API_BASE"),
    )

    assert not convo.set_api_key(os.getenv("AZURE_TEST_OPENAI_API_KEY"))
    
