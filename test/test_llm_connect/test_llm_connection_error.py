"""Tests for LLMConnectionError propagation through llm_connect and prompts."""

from unittest.mock import MagicMock, patch

import pytest

from biochatter.llm_connect import Conversation, GptConversation, LLMConnectionError
from biochatter.llm_connect.exceptions import LLMConnectionError as LLMConnectionErrorDirect
from biochatter.prompts import BioCypherPromptEngine


class TestLLMConnectionErrorExport:
    def test_importable_from_llm_connect_package(self):
        assert LLMConnectionError is LLMConnectionErrorDirect


class TestConversationQueryErrorPropagation:
    @pytest.fixture
    def conversation(self):
        conv = GptConversation(model_name="gpt-4", prompts={"primary_model_prompts": []}, correct=False)
        conv.chat = MagicMock()
        return conv

    def test_query_propagates_llm_connection_error(self, conversation):
        conversation._primary_query = MagicMock(
            side_effect=LLMConnectionError("API down", provider="openai", model="gpt-4"),
        )

        with pytest.raises(LLMConnectionError, match="API down") as exc_info:
            conversation.query("hello")

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4"

    def test_query_normalizes_unexpected_exceptions(self, conversation):
        conversation._primary_query = MagicMock(side_effect=RuntimeError("unexpected boom"))

        with pytest.raises(LLMConnectionError, match="unexpected boom") as exc_info:
            conversation.query("hello")

        assert exc_info.value.provider == "GptConversation"
        assert exc_info.value.model == "gpt-4"


class TestBioCypherPromptEngineErrorContext:
    @pytest.fixture
    def prompt_engine(self):
        return BioCypherPromptEngine(
            schema_config_or_info_path="test/test_schema_info.yaml",
        )

    def test_select_entities_surfaces_step_context(self, prompt_engine):
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.query.side_effect = LLMConnectionError(
            "403 blocked",
            provider="google_genai",
            model="gemini-3.5-flash",
        )

        with pytest.raises(LLMConnectionError, match="Entity selection failed: 403 blocked") as exc_info:
            prompt_engine._select_entities(question="Which proteins?", conversation=mock_conversation)

        assert exc_info.value.provider == "google_genai"
        assert exc_info.value.model == "gemini-3.5-flash"

    def test_generate_query_surfaces_step_context(self, prompt_engine):
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.query.side_effect = LLMConnectionError(
            "timeout",
            provider="openai",
            model="gpt-4",
        )

        with pytest.raises(LLMConnectionError, match="Query generation failed: timeout"):
            prompt_engine._generate_query(
                question="Find proteins",
                entities=["Protein"],
                relationships={},
                properties={},
                query_language="cypher",
                conversation=mock_conversation,
            )
