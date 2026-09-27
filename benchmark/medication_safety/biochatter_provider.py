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


class ConfiguredChat:
    """Forward explicit settings through both BioChatter chat call styles."""

    def __init__(self, chat, options: dict) -> None:
        """Wrap a chat without changing its provider client or credentials."""
        self._chat = chat
        self._options = options

    def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self._chat.generate(*args, **{**self._options, **kwargs})

    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self._chat.invoke(*args, **{**self._options, **kwargs})

    def __getattr__(self, name: str):  # noqa: ANN204
        """Preserve access to the underlying chat attributes."""
        return getattr(self._chat, name)


def create_conversation(  # noqa: C901, PLR0913
    provider: str,
    model_name: str,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
):
    """Create and authenticate a BioChatter conversation."""
    if provider in {"anthropic", "gemini"} and (reasoning_effort or thinking):
        msg = "Reasoning/thinking options require an OpenAI-compatible provider."
        raise ValueError(msg)
    if max_tokens is not None and max_tokens < 1:
        msg = "max_tokens must be positive."
        raise ValueError(msg)
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
    options = {"temperature": temperature}
    if max_tokens is not None:
        options["max_output_tokens" if provider == "gemini" else "max_tokens"] = max_tokens
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    if thinking:
        options["extra_body"] = {"thinking": {"type": thinking}}
    conversation.chat = ConfiguredChat(conversation.chat, options)
    return conversation
