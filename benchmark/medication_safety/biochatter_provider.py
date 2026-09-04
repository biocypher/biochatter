"""Create BioChatter conversations for benchmark providers."""

from __future__ import annotations

import os

from biochatter.llm_connect import (
    AnthropicConversation,
    GeminiConversation,
    GptConversation,
    OpenRouterConversation,
)


DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
DEFAULT_BASE_URL = {
    "deepseek": "https://api.deepseek.com",
}


def create_conversation(
    provider: str,
    model_name: str,
    api_key_env: str | None = None,
    base_url: str | None = None,
):
    """Create and authenticate a BioChatter conversation."""
    key_env = api_key_env or DEFAULT_KEY_ENV[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        msg = f"Required API key environment variable is not set: {key_env}"
        raise RuntimeError(msg)

    if provider == "anthropic":
        conversation = AnthropicConversation(model_name, prompts={}, correct=False)
    elif provider == "gemini":
        conversation = GeminiConversation(model_name, prompts={}, correct=False)
    elif provider == "openrouter":
        os.environ.setdefault("OPENROUTER_API_KEY", api_key)
        conversation = OpenRouterConversation(model_name, prompts={}, correct=False)
    else:
        resolved_base_url = base_url or DEFAULT_BASE_URL.get(provider)
        if provider == "openai-compatible" and not resolved_base_url:
            msg = "--base-url is required for an OpenAI-compatible provider."
            raise ValueError(msg)
        conversation = GptConversation(
            model_name,
            prompts={},
            correct=False,
            base_url=resolved_base_url,
        )

    if not conversation.set_api_key(api_key, user="benchmark_user"):
        msg = f"BioChatter could not authenticate {model_name} via {provider}."
        raise RuntimeError(msg)
    return conversation
