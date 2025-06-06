"""Tests for token extraction functions in the Conversation class.

This module tests the token extraction utility functions that standardize
token counting across different LLM providers.
"""

import pytest
from biochatter.llm_connect.conversation import Conversation


class TestTokenExtraction:
    """Test the token extraction utility functions."""

    @pytest.fixture
    def conversation_instance(self):
        """Create a minimal conversation instance for testing."""

        # Create a concrete implementation for testing
        class TestConversation(Conversation):
            def set_api_key(self, api_key: str, user: str | None = None) -> None:
                pass

            def _primary_query(self, text: str) -> tuple[str, dict | None]:
                return "test", None

            def _correct_response(self, msg: str) -> str:
                return "corrected"

        return TestConversation(model_name="test-model", prompts={})

    # ================================================================================================
    # Test data for different provider formats
    # ================================================================================================

    @pytest.fixture
    def openai_token_usage(self):
        """OpenAI/Azure style token usage."""
        return {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}

    @pytest.fixture
    def anthropic_token_usage(self):
        """Anthropic style token usage."""
        return {"input_tokens": 50, "output_tokens": 30}

    @pytest.fixture
    def gemini_token_usage(self):
        """Gemini style token usage."""
        return {"total_tokens": 80}

    @pytest.fixture
    def gemini_detailed_token_usage(self):
        """Gemini style with detailed token breakdown."""
        return {"prompt_tokens": 50, "candidates_tokens": 30, "total_tokens": 80}

    @pytest.fixture
    def litellm_token_usage(self):
        """LiteLLM style token usage."""
        return {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}

    @pytest.fixture
    def ollama_token_usage(self):
        """Ollama style token usage (integer)."""
        return 80

    @pytest.fixture
    def partial_input_only(self):
        """Token usage with only input tokens."""
        return {"input_tokens": 50}

    @pytest.fixture
    def partial_output_only(self):
        """Token usage with only output tokens."""
        return {"output_tokens": 30}

    @pytest.fixture
    def empty_dict(self):
        """Empty token usage dictionary."""
        return {}

    # ================================================================================================
    # Tests for _extract_total_tokens
    # ================================================================================================

    def test_extract_total_tokens_openai_format(self, conversation_instance, openai_token_usage):
        """Test extracting total tokens from OpenAI format."""
        result = conversation_instance._extract_total_tokens(openai_token_usage)
        assert result == 80

    def test_extract_total_tokens_anthropic_format(self, conversation_instance, anthropic_token_usage):
        """Test extracting total tokens from Anthropic format (calculate from input + output)."""
        result = conversation_instance._extract_total_tokens(anthropic_token_usage)
        assert result == 80

    def test_extract_total_tokens_gemini_format(self, conversation_instance, gemini_token_usage):
        """Test extracting total tokens from Gemini format."""
        result = conversation_instance._extract_total_tokens(gemini_token_usage)
        assert result == 80

    def test_extract_total_tokens_litellm_format(self, conversation_instance, litellm_token_usage):
        """Test extracting total tokens from LiteLLM format."""
        result = conversation_instance._extract_total_tokens(litellm_token_usage)
        assert result == 80

    def test_extract_total_tokens_ollama_format(self, conversation_instance, ollama_token_usage):
        """Test extracting total tokens from Ollama format (integer)."""
        result = conversation_instance._extract_total_tokens(ollama_token_usage)
        assert result == 80

    def test_extract_total_tokens_fallback_input_only(self, conversation_instance, partial_input_only):
        """Test fallback to input tokens when only input available."""
        result = conversation_instance._extract_total_tokens(partial_input_only)
        assert result == 50

    def test_extract_total_tokens_fallback_output_only(self, conversation_instance, partial_output_only):
        """Test fallback to output tokens when only output available."""
        result = conversation_instance._extract_total_tokens(partial_output_only)
        assert result == 30

    def test_extract_total_tokens_none_input(self, conversation_instance):
        """Test handling None input."""
        result = conversation_instance._extract_total_tokens(None)
        assert result is None

    def test_extract_total_tokens_empty_dict(self, conversation_instance, empty_dict):
        """Test handling empty dictionary."""
        result = conversation_instance._extract_total_tokens(empty_dict)
        assert result is None

    def test_extract_total_tokens_invalid_input(self, conversation_instance):
        """Test handling invalid input types."""
        result = conversation_instance._extract_total_tokens("invalid")
        assert result is None

    # ================================================================================================
    # Tests for _extract_input_tokens
    # ================================================================================================

    def test_extract_input_tokens_openai_format(self, conversation_instance, openai_token_usage):
        """Test extracting input tokens from OpenAI format."""
        result = conversation_instance._extract_input_tokens(openai_token_usage)
        assert result == 50

    def test_extract_input_tokens_anthropic_format(self, conversation_instance, anthropic_token_usage):
        """Test extracting input tokens from Anthropic format."""
        result = conversation_instance._extract_input_tokens(anthropic_token_usage)
        assert result == 50

    def test_extract_input_tokens_gemini_detailed_format(self, conversation_instance, gemini_detailed_token_usage):
        """Test extracting input tokens from detailed Gemini format."""
        result = conversation_instance._extract_input_tokens(gemini_detailed_token_usage)
        # Should prefer input_tokens if available, fallback to prompt_tokens
        assert result == 50

    def test_extract_input_tokens_litellm_format(self, conversation_instance, litellm_token_usage):
        """Test extracting input tokens from LiteLLM format."""
        result = conversation_instance._extract_input_tokens(litellm_token_usage)
        assert result == 50

    def test_extract_input_tokens_gemini_basic_format(self, conversation_instance, gemini_token_usage):
        """Test extracting input tokens from basic Gemini format (only total available)."""
        result = conversation_instance._extract_input_tokens(gemini_token_usage)
        assert result is None  # Can't extract input from total only

    def test_extract_input_tokens_ollama_format(self, conversation_instance, ollama_token_usage):
        """Test extracting input tokens from Ollama format (integer - can't distinguish)."""
        result = conversation_instance._extract_input_tokens(ollama_token_usage)
        assert result is None

    def test_extract_input_tokens_output_only(self, conversation_instance, partial_output_only):
        """Test extracting input tokens when only output available."""
        result = conversation_instance._extract_input_tokens(partial_output_only)
        assert result is None

    def test_extract_input_tokens_none_input(self, conversation_instance):
        """Test handling None input."""
        result = conversation_instance._extract_input_tokens(None)
        assert result is None

    def test_extract_input_tokens_empty_dict(self, conversation_instance, empty_dict):
        """Test handling empty dictionary."""
        result = conversation_instance._extract_input_tokens(empty_dict)
        assert result is None

    def test_extract_input_tokens_invalid_input(self, conversation_instance):
        """Test handling invalid input types."""
        result = conversation_instance._extract_input_tokens("invalid")
        assert result is None

    # ================================================================================================
    # Tests for _extract_output_tokens
    # ================================================================================================

    def test_extract_output_tokens_openai_format(self, conversation_instance, openai_token_usage):
        """Test extracting output tokens from OpenAI format."""
        result = conversation_instance._extract_output_tokens(openai_token_usage)
        assert result == 30

    def test_extract_output_tokens_anthropic_format(self, conversation_instance, anthropic_token_usage):
        """Test extracting output tokens from Anthropic format."""
        result = conversation_instance._extract_output_tokens(anthropic_token_usage)
        assert result == 30

    def test_extract_output_tokens_gemini_detailed_format(self, conversation_instance, gemini_detailed_token_usage):
        """Test extracting output tokens from detailed Gemini format."""
        result = conversation_instance._extract_output_tokens(gemini_detailed_token_usage)
        # Should prefer output_tokens if available, fallback to completion_tokens, then candidates_tokens
        assert result == 30

    def test_extract_output_tokens_litellm_format(self, conversation_instance, litellm_token_usage):
        """Test extracting output tokens from LiteLLM format."""
        result = conversation_instance._extract_output_tokens(litellm_token_usage)
        assert result == 30

    def test_extract_output_tokens_gemini_basic_format(self, conversation_instance, gemini_token_usage):
        """Test extracting output tokens from basic Gemini format (only total available)."""
        result = conversation_instance._extract_output_tokens(gemini_token_usage)
        assert result is None  # Can't extract output from total only

    def test_extract_output_tokens_ollama_format(self, conversation_instance, ollama_token_usage):
        """Test extracting output tokens from Ollama format (integer - can't distinguish)."""
        result = conversation_instance._extract_output_tokens(ollama_token_usage)
        assert result is None

    def test_extract_output_tokens_input_only(self, conversation_instance, partial_input_only):
        """Test extracting output tokens when only input available."""
        result = conversation_instance._extract_output_tokens(partial_input_only)
        assert result is None

    def test_extract_output_tokens_none_input(self, conversation_instance):
        """Test handling None input."""
        result = conversation_instance._extract_output_tokens(None)
        assert result is None

    def test_extract_output_tokens_empty_dict(self, conversation_instance, empty_dict):
        """Test handling empty dictionary."""
        result = conversation_instance._extract_output_tokens(empty_dict)
        assert result is None

    def test_extract_output_tokens_invalid_input(self, conversation_instance):
        """Test handling invalid input types."""
        result = conversation_instance._extract_output_tokens("invalid")
        assert result is None

    # ================================================================================================
    # Tests for priority handling (when multiple fields are available)
    # ================================================================================================

    def test_input_tokens_priority_anthropic_over_openai(self, conversation_instance):
        """Test that input_tokens takes priority over prompt_tokens."""
        mixed_format = {
            "input_tokens": 60,  # Should be preferred
            "prompt_tokens": 50,
            "output_tokens": 30,
            "completion_tokens": 25,
            "total_tokens": 90,
        }
        result = conversation_instance._extract_input_tokens(mixed_format)
        assert result == 60

    def test_output_tokens_priority_anthropic_over_openai(self, conversation_instance):
        """Test that output_tokens takes priority over completion_tokens."""
        mixed_format = {
            "input_tokens": 50,
            "prompt_tokens": 55,
            "output_tokens": 35,  # Should be preferred
            "completion_tokens": 30,
            "total_tokens": 85,
        }
        result = conversation_instance._extract_output_tokens(mixed_format)
        assert result == 35

    def test_output_tokens_priority_completion_over_candidates(self, conversation_instance):
        """Test that completion_tokens takes priority over candidates_tokens."""
        mixed_format = {
            "completion_tokens": 25,  # Should be preferred
            "candidates_tokens": 30,
            "total_tokens": 75,
        }
        result = conversation_instance._extract_output_tokens(mixed_format)
        assert result == 25

    def test_total_tokens_priority_direct_over_calculated(self, conversation_instance):
        """Test that direct total_tokens takes priority over calculated total."""
        mixed_format = {
            "input_tokens": 50,
            "output_tokens": 30,  # Would calculate to 80
            "total_tokens": 85,  # Should be preferred
        }
        result = conversation_instance._extract_total_tokens(mixed_format)
        assert result == 85

    # ================================================================================================
    # Edge cases and special scenarios
    # ================================================================================================

    def test_zero_token_values(self, conversation_instance):
        """Test handling zero token values."""
        zero_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        total = conversation_instance._extract_total_tokens(zero_tokens)
        input_tokens = conversation_instance._extract_input_tokens(zero_tokens)
        output_tokens = conversation_instance._extract_output_tokens(zero_tokens)

        assert total == 0
        assert input_tokens == 0
        assert output_tokens == 0

    def test_negative_token_values(self, conversation_instance):
        """Test handling negative token values (edge case)."""
        negative_tokens = {"input_tokens": -1, "output_tokens": 30, "total_tokens": 29}

        total = conversation_instance._extract_total_tokens(negative_tokens)
        input_tokens = conversation_instance._extract_input_tokens(negative_tokens)
        output_tokens = conversation_instance._extract_output_tokens(negative_tokens)

        assert total == 29
        assert input_tokens == -1
        assert output_tokens == 30

    def test_non_integer_token_values(self, conversation_instance):
        """Test handling non-integer token values."""
        float_tokens = {"input_tokens": 50.5, "output_tokens": 30.7, "total_tokens": 81.2}

        total = conversation_instance._extract_total_tokens(float_tokens)
        input_tokens = conversation_instance._extract_input_tokens(float_tokens)
        output_tokens = conversation_instance._extract_output_tokens(float_tokens)

        assert total == 81.2
        assert input_tokens == 50.5
        assert output_tokens == 30.7

    def test_string_token_values(self, conversation_instance):
        """Test handling string token values (should work if they're numeric)."""
        string_tokens = {"input_tokens": "50", "output_tokens": "30", "total_tokens": "80"}

        total = conversation_instance._extract_total_tokens(string_tokens)
        input_tokens = conversation_instance._extract_input_tokens(string_tokens)
        output_tokens = conversation_instance._extract_output_tokens(string_tokens)

        assert total == "80"
        assert input_tokens == "50"
        assert output_tokens == "30"

    def test_nested_token_structure(self, conversation_instance):
        """Test handling nested token structures (shouldn't work but shouldn't crash)."""
        nested_tokens = {"usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}}

        total = conversation_instance._extract_total_tokens(nested_tokens)
        input_tokens = conversation_instance._extract_input_tokens(nested_tokens)
        output_tokens = conversation_instance._extract_output_tokens(nested_tokens)

        assert total is None
        assert input_tokens is None
        assert output_tokens is None

    # ================================================================================================
    # Integration test with real provider formats
    # ================================================================================================

    def test_real_world_provider_formats(self, conversation_instance):
        """Test with actual token usage formats from different providers."""
        # Real OpenAI response format
        openai_real = {
            "prompt_tokens": 156,
            "completion_tokens": 89,
            "total_tokens": 245,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

        # Real Anthropic response format
        anthropic_real = {"input_tokens": 156, "output_tokens": 89}

        # Real Gemini response format
        gemini_real = {"prompt_tokens": 156, "candidates_tokens": 89, "total_tokens": 245}

        # Test all formats
        for provider_format, expected_total, expected_input, expected_output in [
            (openai_real, 245, 156, 89),
            (anthropic_real, 245, 156, 89),  # Total calculated from input + output
            (gemini_real, 245, 156, 89),
        ]:
            total = conversation_instance._extract_total_tokens(provider_format)
            input_tokens = conversation_instance._extract_input_tokens(provider_format)
            output_tokens = conversation_instance._extract_output_tokens(provider_format)

            assert total == expected_total, f"Failed for {provider_format}"
            assert input_tokens == expected_input, f"Failed for {provider_format}"
            assert output_tokens == expected_output, f"Failed for {provider_format}"
