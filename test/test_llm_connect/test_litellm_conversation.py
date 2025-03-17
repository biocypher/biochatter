from unittest.mock import MagicMock, Mock, mock_open, patch
import pytest
from litellm.exceptions import NotFoundError

from biochatter.llm_connect import LiteLLMConversation
from biochatter.llm_connect import HumanMessage, SystemMessage


MOCK_PROMPTS = {
    "primary_model_prompts": ["Test system prompt"],
    "correcting_agent_prompts": ["Test correcting agent prompt"],
    "rag_agent_prompts": ["Test RAG agent prompt", "Test RAG agent prompt with {statements}"],
    "tool_prompts": {"test_tool": "Test tool prompt with {df}"}
}

MOCK_MODEL_LIST = ["gpt-4", "gpt-3.5-turbo", "claude-2", "claude-instant-1"]

MOCK_MODELS_BY_PROVIDER = {
    "openai": ["gpt-4", "gpt-3.5-turbo"],
    "anthropic": ["claude-2", "claude-instant-1"]
}

MOCK_MODEL_COST = {
    "gpt-4": {"max_tokens": 8192, "input_cost_per_token": 0.00003, "output_cost_per_token": 0.00006},
    "gpt-3.5-turbo": {"max_tokens": 4096, "input_cost_per_token": 0.0000015, "output_cost_per_token": 0.000002},
    "claude-2": {"max_tokens": 100000, "input_cost_per_token": 0.00001102, "output_cost_per_token": 0.00003268}
}


@patch("biochatter.llm_connect.ChatLiteLLM")
def test_get_litellm_object(mock_chatlite, dummy_api_key="dummy_key"):
    """
    Test that for a model name whether or not get_litellm_object
    calls ChatLiteLLM with the correct parameters.
    """
    uc = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    dummy_instance = MagicMock()
    mock_chatlite.return_value = dummy_instance

    result = uc.get_litellm_object(dummy_api_key)

    mock_chatlite.assert_called_with(
        temperature=0,
        openai_api_key=dummy_api_key,
        model_name="gpt-3.5-turbo"
    )
    assert result == dummy_instance
    
@patch("biochatter.llm_connect.ChatLiteLLM")
def test_get_litellm_object_unsupported_model(mock_chatlite, dummy_api_key="dummy_key"):
    """
    Test that an unsupported model name raises a ValueError.
    """
    uc = LiteLLMConversation(model_name="unknown-model", prompts={})
    with pytest.raises(ValueError, match="Unsupported model: unknown-model"):
        uc.get_litellm_object(dummy_api_key)
    mock_chatlite.assert_not_called()
    
@patch.object(LiteLLMConversation, "get_litellm_object")
def test_set_api_key_success(mock_get_llm, dummy_api_key="dummy_key"):
    """
    Test that set_api_key assigns chat and ca_chat correctly on success.
    """
    dummy_chat_instance = MagicMock()
    mock_get_llm.return_value = dummy_chat_instance

    uc = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    result = uc.set_api_key(dummy_api_key, user="test_user")

    assert result is True
    # The API key should be saved and both chat and ca_chat should be assigned.
    assert uc.api_key == dummy_api_key
    assert uc.chat == dummy_chat_instance
    assert uc.ca_chat == dummy_chat_instance
    assert uc.user == "test_user"

@patch.object(LiteLLMConversation, "get_litellm_object")
def test_set_api_key_failure(mock_get_llm, dummy_api_key="dummy_key"):
    """
    Test that if get_litellm_object throws an exception, set_api_key returns False
    and does not initialize chat attributes.
    """
    mock_get_llm.side_effect = ValueError("Invalid API key")
    uc = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    result = uc.set_api_key(dummy_api_key, user="test_user")

    assert result is False
    with pytest.raises(AttributeError):
        _ = uc.chat
    with pytest.raises(AttributeError):
        _ = uc.ca_chat
    
def valid_response(token_usage):
    return {
        "generations": [
            [
                {
                    "message": {
                        "response_metadata": {
                            "token_usage": token_usage
                        }
                    },
                    "text": "dummy text"
                }
            ]
        ]
    }
  
def test_parse_llm_response_valid():
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    usage = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
    response = valid_response(usage)
    result = conv.parse_llm_response(response)
    assert result == usage

def test_parse_llm_response_missing_generations():
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    # Missing 'generations' key
    response = {"not_generations": []}
    result = conv.parse_llm_response(response)
    assert result is None

def test_parse_llm_response_incomplete_structure():
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    # generations present but missing nested keys
    response = {"generations": [[{"no_message": {}}]]}
    result = conv.parse_llm_response(response)
    assert result is None

def test_parse_llm_response_none_input():
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    # Passing None should be caught and return None
    result = conv.parse_llm_response(None)
    assert result is None

def test_parse_llm_response_wrong_type():
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={})
    # Passing an integer instead of a dict; expect the conversion to fail and return None.
    result = conv.parse_llm_response(12345)
    assert result is None

def test_correct_response_ok():
    """Test _correct_response returns 'OK' when the generated response is OK."""
    # Arrange
    conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts={}, correct=True)
    conv.ca_messages = []  
    conv.ca_model_name = "gpt-3.5-turbo-correct"
    
    # Dummy generation returning "OK"
    dummy_generation = MagicMock()
    dummy_generation.text = "OK"
    dummy_response = MagicMock()
    dummy_response.generations = [[dummy_generation]]
    
    conv.ca_chat = MagicMock()
    conv.ca_chat.generate.return_value = dummy_response
    conv.parse_llm_response = MagicMock(return_value={"prompt_tokens": 5, "completion_tokens": 3})
    conv._update_usage_stats = MagicMock()
    
    # Act
    correction = conv._correct_response("Some response that needs no correction")
    
    # Assert
    assert correction == "OK"
    conv.ca_chat.generate.assert_called_once()
    conv._update_usage_stats.assert_called_once_with("gpt-3.5-turbo-correct", {"prompt_tokens": 5, "completion_tokens": 3})


# Setup and teardown fixtures
@pytest.fixture
def setup_mocks():
    """Setup all the mocks needed for the tests."""
    with patch('litellm.model_list', MOCK_MODEL_LIST), \
         patch('litellm.models_by_provider', MOCK_MODELS_BY_PROVIDER), \
         patch('litellm.model_cost', MOCK_MODEL_COST):
        yield

@pytest.fixture
def conversation(setup_mocks):
    """Create a mock LiteLLMConversation instance."""
    from biochatter.llm_connect import LiteLLMConversation

    conversation = LiteLLMConversation(
        model_name="gpt-4",
        prompts=MOCK_PROMPTS,
        correct=True,
        split_correction=False,
        use_ragagent_selector=False,
        update_token_usage=None
    )
    
    conversation._chat = MagicMock()
    conversation._ca_chat = MagicMock()
    conversation.ca_model_name = "gpt-4"
    
    return conversation

def test_correct_response(conversation):
    """Test the _correct_response method."""
    # Arrange
    test_message = "This is a test message that needs correction."
    expected_correction = "OK"
    
    conversation.ca_messages = [
        SystemMessage(content="Test correcting agent prompt")
    ]
    
    mock_response = MagicMock()
    mock_response.generations = [[MagicMock(text=expected_correction)]]
    conversation.ca_chat.generate.return_value = mock_response
    
    conversation.parse_llm_response = MagicMock(return_value={"input_tokens": 10, "output_tokens": 5})
    
    conversation._update_usage_stats = MagicMock()
    
    result = conversation._correct_response(test_message)
    
    assert result == expected_correction
    conversation.ca_chat.generate.assert_called_once()
    conversation.parse_llm_response.assert_called_once_with(mock_response)
    conversation._update_usage_stats.assert_called_once_with(
        conversation.ca_model_name, 
        {"input_tokens": 10, "output_tokens": 5}
    )
    
    call_args = conversation.ca_chat.generate.call_args[0][0]
    assert len(call_args[0]) == 3  
    assert isinstance(call_args[0][0], SystemMessage)
    assert isinstance(call_args[0][1], HumanMessage)
    assert isinstance(call_args[0][2], SystemMessage)
    assert call_args[0][1].content == test_message

def test_get_model_max_tokens(conversation):
    """Test the get_model_max_tokens method."""
    model_name = "gpt-4"
    expected_max_tokens = 8192
    
    result = conversation.get_model_max_tokens(model_name)
    
    # Assert
    assert result == expected_max_tokens

def test_get_model_max_tokens_not_found(conversation):
    """Test the get_model_max_tokens method with a model that doesn't exist."""
    model_name = "nonexistent-model"
    
    conversation.get_model_info = MagicMock(side_effect=NotFoundError(
        message="Model information is not available.",
        model=model_name,
        llm_provider="Unknown"
    ))
    
    with pytest.raises(NotFoundError):
        conversation.get_model_max_tokens(model_name)

def test_get_model_max_tokens_missing_info(conversation):
    """Test the get_model_max_tokens method with a model that exists but missing max_tokens."""
    model_name = "gpt-4"
    
    conversation.get_model_info = MagicMock(return_value={
        "input_cost_per_token": 0.00003, 
        "output_cost_per_token": 0.00006
    })
    
    with pytest.raises(NotFoundError):
        conversation.get_model_max_tokens(model_name)

def test_get_model_info(conversation):
    """Test the get_model_info method."""
    model_name = "gpt-4"
    expected_info = {
        "max_tokens": 8192, 
        "input_cost_per_token": 0.00003, 
        "output_cost_per_token": 0.00006
    }
    
    result = conversation.get_model_info(model_name)
    
    assert result == expected_info

def test_get_model_info_not_found(conversation):
    """Test the get_model_info method with a model that doesn't exist."""
    model_name = "nonexistent-model"
    
    with pytest.raises(NotFoundError):
        conversation.get_model_info(model_name)

def test_get_all_model_info(conversation):
    """Test the get_all_model_info method."""
    expected_info = MOCK_MODEL_COST
    
    result = conversation.get_all_model_info()
    
    assert result == expected_info

def test_get_models_by_provider(conversation):
    """Test the get_models_by_provider method."""
    expected_models = MOCK_MODELS_BY_PROVIDER
    
    result = conversation.get_models_by_provider()
    
    assert result == expected_models

def test_get_all_model_list(conversation):
    """Test the get_all_model_list method."""
    expected_models = MOCK_MODEL_LIST
    
    result = conversation.get_all_model_list()
    
    assert result == expected_models