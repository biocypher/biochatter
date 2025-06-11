"""Test module for warning functionality when unused kwargs are passed to conversation classes."""

import warnings
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from biochatter.llm_connect.anthropic import AnthropicConversation
from biochatter.llm_connect.gemini import GeminiConversation
from biochatter.llm_connect.langchain import LangChainConversation
from biochatter.llm_connect.llmlite import LiteLLMConversation
from biochatter.llm_connect.ollama import OllamaConversation
from biochatter.llm_connect.openai import GptConversation
from biochatter.llm_connect.xinference import XinferenceConversation


@pytest.fixture
def common_prompts():
    """Common prompts fixture for all conversation classes."""
    return {
        "primary_model_prompts": [],
        "correcting_agent_prompts": [],
        "rag_agent_prompts": [],
    }


class TestAnthropicConversationWarnings:
    """Test warnings for AnthropicConversation."""

    def test_anthropic_warns_on_unused_kwargs(self, common_prompts):
        """Test that AnthropicConversation warns when kwargs are passed but not used."""
        conv = AnthropicConversation(
            model_name="claude-3-5-sonnet-20240620",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods to avoid actual API calls but return success
            conv.chat = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            mock_response = MagicMock()
            mock_response.generations = [[MagicMock(text="Test response")]]
            mock_response.llm_output = {"token_usage": {"total_tokens": 100}}
            conv.chat.generate.return_value = mock_response
            conv.append_ai_message = MagicMock()

            # Call with unused kwargs - this should just warn, not raise an exception
            result = conv._primary_query(unused_param="test", another_param="value")

            # Check that warning was triggered
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert "unused_param" in str(w[0].message)
            assert "another_param" in str(w[0].message)
            assert w[0].category == UserWarning

            # Check that the method still works normally
            assert result[0] == "Test response"

    def test_anthropic_no_warning_without_kwargs(self, common_prompts):
        """Test that AnthropicConversation does not warn when no kwargs are passed."""
        conv = AnthropicConversation(
            model_name="claude-3-5-sonnet-20240620",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            mock_response = MagicMock()
            mock_response.generations = [[MagicMock(text="Test response")]]
            mock_response.llm_output = {"token_usage": {"total_tokens": 100}}
            conv.chat.generate.return_value = mock_response
            conv.append_ai_message = MagicMock()

            # Call without kwargs
            result = conv._primary_query()

            # Check that no warning was triggered
            assert len(w) == 0
            # Check that the method still works normally
            assert result[0] == "Test response"


class TestOllamaConversationWarnings:
    """Test warnings for OllamaConversation."""

    def test_ollama_warns_on_unused_kwargs(self, common_prompts):
        """Test that OllamaConversation warns when kwargs are passed but not used."""
        conv = OllamaConversation(
            base_url="http://localhost:11434",
            prompts=common_prompts,
            model_name="llama3",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.model = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            mock_response = MagicMock()
            mock_response.dict.return_value = {"content": "Test response", "response_metadata": {"eval_count": 100}}
            conv.model.invoke.return_value = mock_response
            conv.append_ai_message = MagicMock()

            # Call with unused kwargs - this should warn but still work
            result = conv._primary_query(temperature=0.5, max_tokens=100)

            # Check that warning was triggered
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert w[0].category == UserWarning
            # Check that the method still works normally
            assert result[0] == "Test response"


class TestXinferenceConversationWarnings:
    """Test warnings for XinferenceConversation."""

    @patch("xinference.client.Client")
    def test_xinference_warns_on_unused_kwargs(self, mock_client, common_prompts):
        """Test that XinferenceConversation warns when kwargs are passed but not used."""
        # Mock the client to avoid connection issues
        mock_client_instance = MagicMock()
        mock_client_instance.list_models.return_value = {"test-id": {"model_name": "test-model", "id": "test-id"}}
        mock_client_instance.get_model.return_value = MagicMock()
        mock_client.return_value = mock_client_instance

        conv = XinferenceConversation(
            base_url="http://localhost:9997",
            prompts=common_prompts,
            model_name="test-model",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.model = MagicMock()
            conv._create_history = MagicMock(return_value=[{"content": "test"}])
            mock_response = {"choices": [{"message": {"content": "Test response"}}], "usage": {"total_tokens": 100}}
            conv.model.chat.return_value = mock_response
            conv.append_ai_message = MagicMock()

            # Call with unused kwargs - should warn but still work
            result = conv._primary_query(stream=True, top_p=0.9)

            # Check that warning was triggered
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert w[0].category == UserWarning
            # Check that the method still works normally
            assert result[0] == "Test response"


class TestLiteLLMConversationWarnings:
    """Test warnings for LiteLLMConversation."""

    def test_litellm_warns_on_unused_kwargs(self, common_prompts):
        """Test that LiteLLMConversation warns when kwargs are passed but not used."""
        conv = LiteLLMConversation(
            model_name="gpt-3.5-turbo",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            conv.chat.generate.side_effect = Exception("API Error")
            conv.append_ai_message = MagicMock()

            # Call with unused kwargs
            try:
                conv._primary_query(frequency_penalty=0.5, presence_penalty=0.3)
            except Exception:
                pass  # Expected due to mocked API error

            # Check that warning was triggered
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert w[0].category == UserWarning


class TestGeminiConversationWarnings:
    """Test warnings for GeminiConversation."""

    def test_gemini_warns_on_unused_kwargs_but_not_tools(self, common_prompts):
        """Test that GeminiConversation warns for unused kwargs but not for tools parameter."""
        conv = GeminiConversation(
            model_name="gemini-2.0-flash",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            conv.chat.bind_tools = MagicMock(return_value=conv.chat)
            conv.model_name = "gemini-2.0-flash"  # Ensure it's in TOOL_CALLING_MODELS

            # Mock a successful response
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.tool_calls = []
            mock_response.usage_metadata = {"total_tokens": 100}
            conv.chat.invoke.return_value = mock_response
            conv.append_ai_message = MagicMock()

            # Call with both tools (should not warn) and other kwargs (should warn)
            result = conv._primary_query(tools=[], temperature=0.8, top_k=40)

            # Check that warning was triggered for non-tools kwargs
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert "temperature" in str(w[0].message)
            assert "top_k" in str(w[0].message)
            assert "tools" not in str(w[0].message)  # tools should not be in warning
            assert w[0].category == UserWarning
            # Check that the method still works normally
            assert result[0] == "Test response"

    def test_gemini_no_warning_with_only_tools(self, common_prompts):
        """Test that GeminiConversation does not warn when only tools are passed."""
        conv = GeminiConversation(
            model_name="gemini-2.0-flash",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            conv.chat.bind_tools = MagicMock(return_value=conv.chat)
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.tool_calls = []
            mock_response.usage_metadata = {"total_tokens": 100}
            conv.chat.invoke.return_value = mock_response
            conv.append_ai_message = MagicMock()
            conv.model_name = "gemini-2.0-flash"

            # Call with only tools parameter
            result = conv._primary_query(tools=[])

            # Check that no warning was triggered
            assert len(w) == 0
            # Check that the method still works normally
            assert result[0] == "Test response"


class TestOpenAIConversationWarnings:
    """Test warnings for GptConversation (OpenAI)."""

    def test_openai_warns_on_unused_kwargs(self, common_prompts):
        """Test that GptConversation warns when kwargs are passed but not used."""
        conv = GptConversation(
            model_name="gpt-4",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            conv.chat.generate.side_effect = Exception("API Error")
            conv.append_ai_message = MagicMock()

            # Call with unused kwargs
            try:
                conv._primary_query(seed=42, logit_bias={})
            except Exception:
                pass  # Expected due to mocked API error

            # Check that warning was triggered
            assert len(w) == 1
            assert "are not used by this class" in str(w[0].message)
            assert w[0].category == UserWarning


class TestLangChainConversationNoWarnings:
    """Test that LangChainConversation does NOT warn (since it uses its parameters)."""

    def test_langchain_no_warning_with_parameters(self, common_prompts):
        """Test that LangChainConversation does not warn since it uses its parameters."""
        conv = LangChainConversation(
            model_name="gpt-4",
            model_provider="openai",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the necessary methods
            conv.chat = MagicMock()
            mock_response = MagicMock()
            mock_response.tool_calls = []
            mock_response.content = "Test response"
            mock_response.usage_metadata = None
            conv.chat.invoke.return_value = mock_response
            conv.append_ai_message = MagicMock()
            conv._extract_total_tokens = MagicMock(return_value=100)

            # Call with parameters that LangChain actually uses
            result = conv._primary_query(
                tools=[],
                explain_tool_result=True,
                return_tool_calls_as_ai_message=False,
            )

            # Check that no warning was triggered (LangChain uses these parameters)
            kwargs_warnings = [warning for warning in w if "are not used by this class" in str(warning.message)]
            assert len(kwargs_warnings) == 0

            # Check that method returns expected result
            assert result[0] == "Test response"


class TestWarningIntegration:
    """Integration tests for warning functionality."""

    def test_warning_message_format(self, common_prompts):
        """Test that warning messages have the expected format."""
        conv = AnthropicConversation(
            model_name="claude-3-5-sonnet-20240620",
            prompts=common_prompts,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock to avoid API calls
            conv.chat = MagicMock()
            conv._create_history = MagicMock(return_value=[])
            conv.chat.generate.side_effect = Exception("API Error")
            conv.append_ai_message = MagicMock()

            # Call with specific kwargs
            try:
                conv._primary_query(param1="value1", param2="value2")
            except Exception:
                pass  # Expected due to mocked API error

            # Check warning message format
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert warning_msg.startswith("Warning:")
            assert "are not used by this class" in warning_msg
            assert "param1" in warning_msg
            assert "param2" in warning_msg
            assert "value1" in warning_msg
            assert "value2" in warning_msg

    def test_multiple_conversation_classes_consistency(self, common_prompts):
        """Test that all conversation classes that should warn have consistent behavior."""
        conversation_classes = [
            (AnthropicConversation, {"model_name": "claude-3-5-sonnet-20240620", "prompts": common_prompts}),
            (
                OllamaConversation,
                {"base_url": "http://localhost:11434", "prompts": common_prompts, "model_name": "llama3"},
            ),
            (LiteLLMConversation, {"model_name": "gpt-3.5-turbo", "prompts": common_prompts}),
            (GptConversation, {"model_name": "gpt-4", "prompts": common_prompts}),
        ]

        for ConvClass, init_kwargs in conversation_classes:
            conv = ConvClass(**init_kwargs)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Mock common attributes to avoid API calls
                if hasattr(conv, "chat"):
                    conv.chat = MagicMock()
                if hasattr(conv, "model"):
                    conv.model = MagicMock()
                if hasattr(conv, "_create_history"):
                    conv._create_history = MagicMock(return_value=[])
                conv.append_ai_message = MagicMock()

                # Make the mocked methods raise exceptions to simulate API errors
                try:
                    if hasattr(conv.chat if hasattr(conv, "chat") else None, "generate"):
                        conv.chat.generate.side_effect = Exception("API Error")
                    if hasattr(conv.model if hasattr(conv, "model") else None, "invoke"):
                        conv.model.invoke.side_effect = Exception("API Error")
                except AttributeError:
                    pass

                # Call with test kwargs
                try:
                    conv._primary_query(test_param="test_value")
                except Exception:
                    pass  # Expected due to mocked errors

                # Check that warning was triggered
                assert len(w) >= 1, f"No warning triggered for {ConvClass.__name__}"
                kwargs_warnings = [warning for warning in w if "are not used by this class" in str(warning.message)]
                assert len(kwargs_warnings) >= 1, f"No kwargs warning for {ConvClass.__name__}"


# Test parameter to ensure the warning filter in pytest.ini doesn't suppress our tests
def test_warning_system_works():
    """Meta-test to ensure that the warning system is working in our test environment."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("Test warning", UserWarning)
        assert len(w) == 1
        assert "Test warning" in str(w[0].message)
        assert w[0].category == UserWarning
