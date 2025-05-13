"""Tests for the Xinference LLM connect module."""

import os
from unittest.mock import patch

import pytest

from biochatter.llm_connect.xinference import XinferenceConversation

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

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
    """Test generic LLM connectivity via the Xinference client. Currently depends
    on a test server.
    """
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://localhost:9997")
    with patch("xinference.client.Client") as mock_client:
        mock_client.return_value.list_models.return_value = xinference_models
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            split_correction=False,
        )
        assert convo.set_api_key()


def test_xinference_chatting():
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://localhost:9997")
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
                },
            ],
            "usage": {
                "prompt_tokens": 93,
                "completion_tokens": 54,
                "total_tokens": 147,
            },
        }
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = response
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            correct=False,
        )
        query_result = convo.query("Hello, world!")
        assert query_result.token_usage['total_tokens'] > 0


@pytest.fixture
def xinference_conversation():
    with patch("xinference.client.Client") as mock_client:
        # Mock the authentication check
        mock_client.return_value._check_cluster_authenticated.return_value = None
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = (
            {"choices": [{"message": {"content": "Human message"}}]},
            {"completion_tokens": 0},
        )
        conversation = XinferenceConversation(
            base_url="http://localhost:9997",
            prompts={},
            correct=False,
        )
        return conversation


def test_single_system_message_before_human(xinference_conversation: XinferenceConversation):
    xinference_conversation.messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


def test_multiple_system_messages_before_human(xinference_conversation: XinferenceConversation):
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
    xinference_conversation: XinferenceConversation,
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


@pytest.mark.skip(reason="Live test for development purposes")
def test_local_image_query_xinference():
    url = "http://localhost:9997"
    convo = XinferenceConversation(
        base_url=url,
        prompts={},
        correct=False,
    )
    assert convo.set_api_key()

    query_result = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="test/figure_panel.jpg",
    )
    assert isinstance(query_result.response, str)
