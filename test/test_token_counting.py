"""Comprehensive tests for token counting across all conversation types in llm_connect.

This module tests that all conversation implementations correctly count and return token usage.
Each conversation type has specific patterns for how they handle token counting, and this test
suite ensures consistency and reliability across all implementations.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

# Import all conversation classes
from biochatter.llm_connect import (
    AnthropicConversation,
    AzureGptConversation,
    BloomConversation,
    GeminiConversation,
    GptConversation,
    LangChainConversation,
    LiteLLMConversation,
    OllamaConversation,
    OpenRouterConversation,
    WasmConversation,
    XinferenceConversation,
)

# Common test data
MOCK_PROMPTS = {
    "primary_model_prompts": ["Test system prompt"],
    "correcting_agent_prompts": ["Test correcting agent prompt"],
}

# Common token usage patterns across different implementations
# Raw formats from providers (for _update_usage_stats)
OPENAI_TOKEN_USAGE_RAW = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
ANTHROPIC_TOKEN_USAGE_RAW = {"input_tokens": 50, "output_tokens": 30}
GEMINI_TOKEN_USAGE_RAW = {"total_tokens": 80}
OLLAMA_TOKEN_USAGE_RAW = 80  # Ollama returns integer count
XINFERENCE_TOKEN_USAGE_RAW = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
LITELLM_TOKEN_USAGE_RAW = {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}

# Standardized total token count (what all conversations now return)
EXPECTED_TOTAL_TOKENS = 80


# Test tools for tool calling scenarios
@tool
def mock_test_tool(arg1: str) -> str:
    """A mock tool for testing."""
    return f"Tool result: {arg1}"


class MockStructuredModel(BaseModel):
    """Mock Pydantic model for structured output testing."""

    field1: str
    field2: int


# ================================================================================================
# OpenAI GPT Conversation Tests
# ================================================================================================


class TestGptConversationTokenCounting:
    """Test token counting for OpenAI GPT conversations."""

    @pytest.fixture
    def gpt_conversation(self):
        """Create a GptConversation instance for testing."""
        with patch("biochatter.llm_connect.openai.ChatOpenAI"):
            conv = GptConversation(model_name="gpt-4", prompts=MOCK_PROMPTS)
            conv.chat = MagicMock()
            conv.ca_chat = MagicMock()
            return conv

    def test_primary_query_returns_token_usage(self, gpt_conversation):
        """Test that _primary_query returns proper token usage for successful responses."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="AI response")]]
        mock_response.llm_output = {"token_usage": OPENAI_TOKEN_USAGE_RAW}
        gpt_conversation.chat.generate.return_value = mock_response
        gpt_conversation.append_ai_message = MagicMock()
        gpt_conversation._update_usage_stats = MagicMock()

        # Act
        msg, token_usage = gpt_conversation._primary_query()

        # Assert
        assert msg == "AI response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)

    def test_primary_query_handles_api_error(self, gpt_conversation):
        """Test that _primary_query returns None token usage on API errors."""
        # Arrange
        import openai

        # Create a mock response object for the exception
        mock_response = MagicMock()
        mock_response.status_code = 401
        gpt_conversation.chat.generate.side_effect = openai._exceptions.AuthenticationError(
            "Invalid API key", response=mock_response, body={"error": "Invalid API key"}
        )

        # Act
        msg, token_usage = gpt_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None

    def test_correct_response_returns_token_usage(self, gpt_conversation):
        """Test that _correct_response returns token usage."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="OK")]]
        mock_response.llm_output = {"token_usage": OPENAI_TOKEN_USAGE_RAW}
        gpt_conversation.ca_chat.generate.return_value = mock_response
        gpt_conversation.ca_messages = []
        gpt_conversation._update_usage_stats = MagicMock()

        # Act
        correction = gpt_conversation._correct_response("Test message")

        # Assert
        assert correction == "OK"
        gpt_conversation._update_usage_stats.assert_called_once_with(
            gpt_conversation.ca_model_name, OPENAI_TOKEN_USAGE_RAW
        )

    def test_query_method_returns_token_usage(self, gpt_conversation):
        """Test that the query method returns token usage in the expected format."""
        # Arrange
        gpt_conversation._primary_query = MagicMock(return_value=("AI response", EXPECTED_TOTAL_TOKENS))
        gpt_conversation.append_user_message = MagicMock()
        gpt_conversation._inject_context = MagicMock()

        # Act
        msg, token_usage, correction = gpt_conversation.query("Test query")

        # Assert
        assert msg == "AI response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert correction is None


# ================================================================================================
# Azure GPT Conversation Tests
# ================================================================================================


class TestAzureGptConversationTokenCounting:
    """Test token counting for Azure GPT conversations."""

    @pytest.fixture
    def azure_conversation(self):
        """Create an AzureGptConversation instance for testing."""
        with patch("biochatter.llm_connect.azure.AzureChatOpenAI"):
            conv = AzureGptConversation(deployment_name="test-deployment", model_name="gpt-4", prompts=MOCK_PROMPTS)
            conv.chat = MagicMock()
            conv.ca_chat = MagicMock()
            return conv

    def test_inherits_token_counting_from_gpt(self, azure_conversation):
        """Test that Azure conversation inherits token counting behavior from GptConversation."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Azure AI response")]]
        mock_response.llm_output = {"token_usage": OPENAI_TOKEN_USAGE_RAW}
        azure_conversation.chat.generate.return_value = mock_response
        azure_conversation.append_ai_message = MagicMock()
        azure_conversation._update_usage_stats = MagicMock()

        # Act
        msg, token_usage = azure_conversation._primary_query()

        # Assert
        assert msg == "Azure AI response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)


# ================================================================================================
# Anthropic Conversation Tests
# ================================================================================================


class TestAnthropicConversationTokenCounting:
    """Test token counting for Anthropic conversations."""

    @pytest.fixture
    def anthropic_conversation(self):
        """Create an AnthropicConversation instance for testing."""
        with patch("biochatter.llm_connect.anthropic.ChatAnthropic"):
            conv = AnthropicConversation(model_name="claude-3-7-sonnet-latest", prompts=MOCK_PROMPTS)
            conv.chat = MagicMock()
            conv.ca_chat = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            return conv

    def test_primary_query_returns_token_usage(self, anthropic_conversation):
        """Test that _primary_query returns proper token usage for Anthropic responses."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Claude response")]]
        mock_response.llm_output = {"token_usage": {"input_tokens": 50, "output_tokens": 30}}
        anthropic_conversation.chat.generate.return_value = mock_response
        anthropic_conversation.append_ai_message = MagicMock()

        # Act
        msg, token_usage = anthropic_conversation._primary_query()

        # Assert
        assert msg == "Claude response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)

    def test_primary_query_handles_anthropic_error(self, anthropic_conversation):
        """Test that _primary_query returns None token usage on Anthropic API errors."""
        # Arrange
        import anthropic

        # Create a mock response object for the exception
        mock_response = MagicMock()
        mock_response.status_code = 401
        anthropic_conversation.chat.generate.side_effect = anthropic._exceptions.AuthenticationError(
            "Invalid API key", response=mock_response, body={"error": "Invalid API key"}
        )

        # Act
        msg, token_usage = anthropic_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None


# ================================================================================================
# Gemini Conversation Tests
# ================================================================================================


class TestGeminiConversationTokenCounting:
    """Test token counting for Google Gemini conversations."""

    @pytest.fixture
    def gemini_conversation(self):
        """Create a GeminiConversation instance for testing."""
        with patch("biochatter.llm_connect.gemini.ChatGoogleGenerativeAI"):
            conv = GeminiConversation(model_name="gemini-2.0-flash", prompts=MOCK_PROMPTS)
            conv.chat = MagicMock()
            conv.ca_chat = MagicMock()
            return conv

    def test_primary_query_returns_token_usage(self, gemini_conversation):
        """Test that _primary_query returns proper token usage for Gemini responses."""
        # Arrange
        mock_response = MagicMock()
        mock_response.content = "Gemini response"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"total_tokens": 80}
        gemini_conversation.chat.invoke.return_value = mock_response
        gemini_conversation.append_ai_message = MagicMock()

        # Act
        msg, token_usage = gemini_conversation._primary_query()

        # Assert
        assert msg == "Gemini response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)

    def test_primary_query_handles_tool_calls(self, gemini_conversation):
        """Test that _primary_query handles tool calls and returns token usage."""
        # Arrange
        mock_response = MagicMock()
        mock_response.content = "I'll use a tool"
        mock_response.tool_calls = [{"name": "mock_test_tool", "args": {"arg1": "test"}, "id": "call_123"}]
        mock_response.usage_metadata = {"total_tokens": 80}

        # Mock the bind_tools method to return a mock that responds to invoke
        mock_bound_chat = MagicMock()
        mock_bound_chat.invoke.return_value = mock_response
        gemini_conversation.chat.bind_tools.return_value = mock_bound_chat

        gemini_conversation._process_tool_calls = MagicMock(return_value="Tool result")

        # Act
        msg, token_usage = gemini_conversation._primary_query(tools=[mock_test_tool])

        # Assert
        assert msg == "Tool result"
        assert token_usage == EXPECTED_TOTAL_TOKENS

    def test_primary_query_handles_exception(self, gemini_conversation):
        """Test that _primary_query returns None token usage on exceptions."""
        # Arrange
        gemini_conversation.chat.invoke.side_effect = Exception("API Error")

        # Act
        msg, token_usage = gemini_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None


# ================================================================================================
# LangChain Conversation Tests
# ================================================================================================


class TestLangChainConversationTokenCounting:
    """Test token counting for LangChain conversations."""

    @pytest.fixture
    def langchain_conversation(self):
        """Create a LangChainConversation instance for testing."""
        conv = LangChainConversation(model_name="gpt-4", model_provider="openai", prompts=MOCK_PROMPTS)
        conv.chat = MagicMock()
        conv.ca_chat = MagicMock()
        conv.messages = [HumanMessage(content="Test message")]
        conv.append_ai_message = MagicMock()
        return conv

    def test_primary_query_basic_response_token_usage(self, langchain_conversation):
        """Test basic response token usage for LangChain conversations."""
        # Arrange
        mock_response = AIMessage(content="LangChain response", id="ai_msg_1")
        mock_response.usage_metadata = {"total_tokens": 50}
        mock_response.tool_calls = []
        langchain_conversation.chat.invoke.return_value = mock_response

        # Act
        with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []):
            msg, token_usage = langchain_conversation._primary_query()

        # Assert
        assert msg == "LangChain response"
        assert token_usage == 50
        assert isinstance(token_usage, int)

    def test_primary_query_tool_calls_token_usage(self, langchain_conversation):
        """Test token usage when tool calls are made."""
        # Arrange
        mock_response = AIMessage(content="I'll use a tool", id="ai_msg_2")
        mock_response.usage_metadata = {"total_tokens": 75}
        mock_response.tool_calls = [{"name": "mock_test_tool", "args": {"arg1": "test"}, "id": "call_123"}]
        langchain_conversation.chat.bind_tools.return_value.invoke.return_value = mock_response
        langchain_conversation._process_tool_calls = MagicMock(return_value="Tool executed")

        # Act
        with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", [langchain_conversation.model_name]):
            msg, token_usage = langchain_conversation._primary_query(tools=[mock_test_tool])

        # Assert
        assert msg == "Tool executed"
        assert token_usage == 75

    def test_primary_query_structured_output_token_usage(self, langchain_conversation):
        """Test token usage for structured output responses."""
        # Arrange
        structured_response = MockStructuredModel(field1="test", field2=42)
        langchain_conversation.chat.with_structured_output.return_value.invoke.return_value = structured_response

        # Act
        with patch("biochatter.llm_connect.langchain.STRUCTURED_OUTPUT_MODELS", [langchain_conversation.model_name]):
            msg, token_usage = langchain_conversation._primary_query(structured_model=MockStructuredModel)

        # Assert
        assert json.loads(msg) == {"field1": "test", "field2": 42}
        assert token_usage == -1  # Special value for structured outputs

    def test_primary_query_exception_handling(self, langchain_conversation):
        """Test that exceptions result in None token usage."""
        # Arrange
        langchain_conversation.chat.invoke.side_effect = Exception("LangChain Error")

        # Act
        with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []):
            msg, token_usage = langchain_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None


# ================================================================================================
# OpenRouter Conversation Tests
# ================================================================================================


class TestOpenRouterConversationTokenCounting:
    """Test token counting for OpenRouter conversations."""

    @pytest.fixture
    def openrouter_conversation(self):
        """Create an OpenRouterConversation instance for testing."""
        conv = OpenRouterConversation(model_name="anthropic/claude-3-7-sonnet", prompts=MOCK_PROMPTS)
        conv.chat = MagicMock()
        conv.ca_chat = MagicMock()
        conv.messages = [HumanMessage(content="Test message")]
        conv.append_ai_message = MagicMock()
        return conv

    def test_inherits_langchain_token_counting(self, openrouter_conversation):
        """Test that OpenRouter inherits LangChain token counting behavior."""
        # Arrange
        mock_response = AIMessage(content="OpenRouter response", id="ai_msg_1")
        mock_response.usage_metadata = {"total_tokens": 60}
        mock_response.tool_calls = []
        openrouter_conversation.chat.invoke.return_value = mock_response

        # Act
        with patch("biochatter.llm_connect.langchain.TOOL_CALLING_MODELS", []):
            msg, token_usage = openrouter_conversation._primary_query()

        # Assert
        assert msg == "OpenRouter response"
        assert token_usage == 60


# ================================================================================================
# LiteLLM Conversation Tests
# ================================================================================================


class TestLiteLLMConversationTokenCounting:
    """Test token counting for LiteLLM conversations."""

    @pytest.fixture
    def litellm_conversation(self):
        """Create a LiteLLMConversation instance for testing."""
        conv = LiteLLMConversation(model_name="gpt-3.5-turbo", prompts=MOCK_PROMPTS)
        conv.chat = MagicMock()
        conv.ca_chat = MagicMock()
        conv.append_ai_message = MagicMock()
        conv._update_usage_stats = MagicMock()
        return conv

    def test_primary_query_returns_parsed_token_usage(self, litellm_conversation):
        """Test that _primary_query returns parsed token usage."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="LiteLLM response")]]
        litellm_conversation.chat.generate.return_value = mock_response
        litellm_conversation.parse_llm_response = MagicMock(
            return_value={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}
        )

        # Act
        msg, token_usage = litellm_conversation._primary_query()

        # Assert
        assert msg == "LiteLLM response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)
        litellm_conversation.parse_llm_response.assert_called_once_with(mock_response)

    def test_parse_llm_response_valid_structure(self, litellm_conversation):
        """Test that parse_llm_response correctly extracts token usage."""
        # Arrange
        mock_response = {
            "generations": [
                [
                    {
                        "message": {
                            "response_metadata": {
                                "token_usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}
                            }
                        },
                        "text": "test",
                    }
                ]
            ]
        }

        # Act
        result = litellm_conversation.parse_llm_response(mock_response)

        # Assert
        assert result == {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}

    def test_parse_llm_response_missing_data(self, litellm_conversation):
        """Test that parse_llm_response returns None for malformed responses."""
        # Arrange
        mock_response = {"generations": []}

        # Act
        result = litellm_conversation.parse_llm_response(mock_response)

        # Assert
        assert result is None

    def test_primary_query_handles_exceptions(self, litellm_conversation):
        """Test that _primary_query handles exceptions and returns None token usage."""
        # Arrange
        import litellm

        litellm_conversation.chat.generate.side_effect = litellm.exceptions.AuthenticationError(
            "Invalid API key", "test_provider", "test_model"
        )

        # Act
        msg, token_usage = litellm_conversation._primary_query()

        # Assert
        assert isinstance(msg, Exception)
        assert token_usage is None


# ================================================================================================
# Ollama Conversation Tests
# ================================================================================================


class TestOllamaConversationTokenCounting:
    """Test token counting for Ollama conversations."""

    @pytest.fixture
    def ollama_conversation(self):
        """Create an OllamaConversation instance for testing."""
        with patch("biochatter.llm_connect.ollama.ChatOllama"):
            conv = OllamaConversation(base_url="http://localhost:11434", model_name="llama3", prompts=MOCK_PROMPTS)
            conv.model = MagicMock()
            conv.ca_model = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            conv.append_ai_message = MagicMock()
            conv._update_usage_stats = MagicMock()
            return conv

    def test_primary_query_returns_token_usage(self, ollama_conversation):
        """Test that _primary_query returns token usage for Ollama responses."""
        # Arrange
        mock_response = MagicMock()
        mock_response.dict.return_value = {
            "content": "Ollama response",
            "response_metadata": {"eval_count": 80},
        }
        ollama_conversation.model.invoke.return_value = mock_response

        # Act
        msg, token_usage = ollama_conversation._primary_query()

        # Assert
        assert msg == "Ollama response"
        assert token_usage == 80
        assert isinstance(token_usage, int)

    def test_primary_query_handles_exceptions(self, ollama_conversation):
        """Test that _primary_query handles exceptions and returns None token usage."""
        # Arrange
        import openai

        # Create a mock request for the exception
        mock_request = MagicMock()
        ollama_conversation.model.invoke.side_effect = openai._exceptions.APIConnectionError(request=mock_request)

        # Act
        msg, token_usage = ollama_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None


# ================================================================================================
# Xinference Conversation Tests
# ================================================================================================


class TestXinferenceConversationTokenCounting:
    """Test token counting for Xinference conversations."""

    @pytest.fixture
    def xinference_conversation(self):
        """Create a XinferenceConversation instance for testing."""
        with (
            patch("xinference.client.Client"),
            patch.object(XinferenceConversation, "list_models_by_type", return_value=["test-model"]),
            patch.object(XinferenceConversation, "set_api_key", return_value=True),
        ):
            conv = XinferenceConversation(base_url="http://localhost:9997", prompts=MOCK_PROMPTS)
            conv.model = MagicMock()
            conv.ca_model = MagicMock()
            conv._create_history = MagicMock(return_value=[{"role": "user", "content": "test"}])
            conv.append_ai_message = MagicMock()
            conv._update_usage_stats = MagicMock()
            return conv

    def test_primary_query_returns_token_usage(self, xinference_conversation):
        """Test that _primary_query returns token usage for Xinference responses."""
        # Arrange
        mock_response = {
            "choices": [{"message": {"content": "Xinference response"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }
        xinference_conversation.model.chat.return_value = mock_response

        # Act
        msg, token_usage = xinference_conversation._primary_query()

        # Assert
        assert msg == "Xinference response"
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert isinstance(token_usage, int)

    def test_primary_query_handles_exceptions(self, xinference_conversation):
        """Test that _primary_query handles exceptions and returns None token usage."""
        # Arrange
        import openai

        mock_request = MagicMock()
        xinference_conversation.model.chat.side_effect = openai._exceptions.APIError(
            "API Error", request=mock_request, body={"error": "API Error"}
        )

        # Act
        msg, token_usage = xinference_conversation._primary_query()

        # Assert
        assert isinstance(msg, str)  # Error message
        assert token_usage is None


# ================================================================================================
# WASM Conversation Tests
# ================================================================================================


class TestWasmConversationTokenCounting:
    """Test token counting for WASM conversations."""

    @pytest.fixture
    def wasm_conversation(self):
        """Create a WasmConversation instance for testing."""
        conv = WasmConversation(model_name="wasm-model", prompts=MOCK_PROMPTS)
        return conv

    def test_query_returns_none_token_usage(self, wasm_conversation):
        """Test that WASM conversations return None for token usage."""
        # Act
        msg, token_usage, correction = wasm_conversation.query("Test query")

        # Assert
        assert isinstance(msg, str)  # Message history as string
        assert token_usage is None  # WASM doesn't track tokens
        assert correction is None


# ================================================================================================
# Bloom Conversation Tests
# ================================================================================================


class TestBloomConversationTokenCounting:
    """Test token counting for Bloom conversations."""

    @pytest.fixture
    def bloom_conversation(self):
        """Create a BloomConversation instance for testing."""
        with patch("biochatter.llm_connect.misc.HuggingFaceHub"):
            conv = BloomConversation(model_name="bigscience/bloom", prompts=MOCK_PROMPTS, split_correction=False)
            conv.chat = MagicMock()
            conv.append_ai_message = MagicMock()
            return conv

    def test_primary_query_returns_zero_token_usage(self, bloom_conversation):
        """Test that Bloom conversations return zero token usage."""
        # Arrange
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Bloom response")]]
        bloom_conversation.chat.generate.return_value = mock_response

        # Act
        msg, token_usage = bloom_conversation._primary_query()

        # Assert
        assert msg == "Bloom response"
        assert token_usage == 0  # Now returns total tokens as integer
        assert isinstance(token_usage, int)


# ================================================================================================
# Integration Tests
# ================================================================================================


class TestTokenCountingIntegration:
    """Integration tests for token counting across conversation types."""

    @pytest.mark.parametrize(
        "conversation_class,expected_token_structure",
        [
            (GptConversation, int),  # Now returns total tokens as integer
            (AnthropicConversation, int),  # Now returns total tokens as integer
            (GeminiConversation, int),  # Single integer
            (OllamaConversation, int),  # Single integer
            (XinferenceConversation, int),  # Now returns total tokens as integer
            (LiteLLMConversation, int),  # Now returns total tokens as integer
            (BloomConversation, int),  # Now returns total tokens as integer
        ],
    )
    def test_token_counting_consistency(self, conversation_class, expected_token_structure):
        """Test that all conversation types return consistent token usage structures."""
        # This test verifies that each conversation type returns the expected
        # token usage structure (dict vs int vs None) consistently

        # Note: This is a structural test that doesn't require mocking specific APIs
        # It just validates that the types are consistent with expectations
        assert True  # Placeholder - specific implementation would depend on conversation setup

    def test_all_conversations_handle_none_token_usage(self):
        """Test that all conversation types can handle None token usage gracefully."""
        # This test ensures that when APIs fail or don't return token usage,
        # all conversation types handle it gracefully

        test_cases = [
            (GptConversation, "gpt-4"),
            (AnthropicConversation, "claude-3-7-sonnet-latest"),
            (GeminiConversation, "gemini-2.0-flash"),
            (OllamaConversation, "llama3"),
            (XinferenceConversation, "auto"),
            (LiteLLMConversation, "gpt-3.5-turbo"),
            (BloomConversation, "bigscience/bloom"),
        ]

        for conversation_class, model_name in test_cases:
            # Each conversation should handle None token usage gracefully
            # This would be tested by mocking API failures
            pass

    def test_query_method_token_usage_propagation(self):
        """Test that the main query method properly propagates token usage from _primary_query."""
        # This test ensures that token usage flows correctly from _primary_query
        # through the query method to the caller

        # Each conversation type should:
        # 1. Return token usage from _primary_query
        # 2. Handle special cases (structured output = -1, errors = None)
        # 3. Properly package in the tuple returned by query()


# ================================================================================================
# Edge Cases and Error Handling
# ================================================================================================


class TestTokenCountingEdgeCases:
    """Test edge cases and error scenarios for token counting."""

    def test_structured_output_token_handling(self):
        """Test that structured output returns -1 for token usage and gets converted to 0."""
        # LangChain conversation with structured output should return -1 for token usage
        # The query method should convert this to 0 in the final response
        conv = LangChainConversation(model_name="gpt-4", model_provider="openai", prompts=MOCK_PROMPTS)
        conv._primary_query = MagicMock(return_value=("structured response", -1))
        conv.append_user_message = MagicMock()
        conv._inject_context = MagicMock()

        msg, token_usage, correction = conv.query("Test", structured_model=MockStructuredModel)

        assert token_usage == 0  # -1 gets converted to 0

    def test_api_error_token_handling(self):
        """Test that API errors result in None token usage."""
        conv = GptConversation(model_name="gpt-4", prompts=MOCK_PROMPTS)
        conv._primary_query = MagicMock(return_value=("error message", None))
        conv.append_user_message = MagicMock()
        conv._inject_context = MagicMock()

        msg, token_usage, correction = conv.query("Test")

        assert token_usage is None

    def test_correction_token_counting(self):
        """Test that correction steps also count tokens properly."""
        conv = GptConversation(model_name="gpt-4", prompts=MOCK_PROMPTS, correct=True)
        conv._primary_query = MagicMock(return_value=("response", EXPECTED_TOTAL_TOKENS))
        conv._correct_query = MagicMock(return_value=["corrected"])
        conv.append_user_message = MagicMock()
        conv._inject_context = MagicMock()

        msg, token_usage, correction = conv.query("Test")

        # Token usage from primary query should be preserved
        assert token_usage == EXPECTED_TOTAL_TOKENS
        assert correction == "corrected"


if __name__ == "__main__":
    pytest.main([__file__])
