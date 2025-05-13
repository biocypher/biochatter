"""Tests for the Anthropic LLM connect module."""

import os

import pytest

from biochatter.llm_connect import AnthropicConversation


@pytest.mark.skip(reason="Live test for development purposes")
def test_anthropic():
    conv = AnthropicConversation(
        model_name="claude-3-5-sonnet-20240620",
        prompts={},
        split_correction=False,
    )
    assert conv.set_api_key(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        user="test_user",
    )
