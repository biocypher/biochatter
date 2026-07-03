"""Define the available models for the LLM connect module.

This module defines the available models for the LLM connect module and their
token limits.

The models are defined as an Enum, which allows for easy lookup of the model
token limits.
"""

from enum import Enum

_CTX_1M = 1_048_576
_CTX_400K = 400_000
_CTX_128K = 128_000


class OpenAIModels(str, Enum):
    """Enum for OpenAI models."""

    GPT_4o = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_54 = "gpt-5.4"
    GPT_54_MINI = "gpt-5.4-mini"
    GPT_54_NANO = "gpt-5.4-nano"
    GPT_55 = "gpt-5.5"
    GPT_55_PRO = "gpt-5.5-pro"


class GeminiModels(str, Enum):
    """Enum for Gemini models."""

    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_31_PRO = "gemini-3.1-pro-preview"


class MistralModels(str, Enum):
    """Enum for Mistral models."""

    MISTRAL_LARGE_LATEST = "mistral-large-latest"
    MISTRAL_MEDIUM_LATEST = "mistral-medium-latest"
    MISTRAL_SMALL_LATEST = "mistral-small-latest"
    CODESTRAL_LATEST = "codestral-latest"


class AnthropicModels(str, Enum):
    """Enum for Anthropic models."""

    CLAUDE_SONNET_46 = "claude-sonnet-4-6"
    CLAUDE_SONNET_45 = "claude-sonnet-4-5"
    CLAUDE_HAIKU_45 = "claude-haiku-4-5"
    CLAUDE_OPUS_48 = "claude-opus-4-8"


class HuggingFaceModels(str, Enum):
    """Enum for HuggingFace models."""

    BLOOM = "bigscience/bloom"


class XInferenceModels(str, Enum):
    """Enum for XInference models."""

    CUSTOM_ENDPOINT = "custom-endpoint"


class TokenLimits(Enum):
    """Enum for token limits of different models.

    Uses a tuple (model_type, token_limit) to ensure unique values while
    maintaining the token limit information.
    """

    GPT_4o = ("gpt-4o", _CTX_128K)
    GPT_4O_MINI = ("gpt-4o-mini", _CTX_128K)
    GPT_41 = ("gpt-4.1", _CTX_1M)
    GPT_41_MINI = ("gpt-4.1-mini", _CTX_1M)
    GPT_41_NANO = ("gpt-4.1-nano", _CTX_1M)
    GPT_5 = ("gpt-5", _CTX_1M)
    GPT_5_MINI = ("gpt-5-mini", _CTX_1M)
    GPT_5_NANO = ("gpt-5-nano", _CTX_1M)
    GPT_54 = ("gpt-5.4", _CTX_1M)
    GPT_54_MINI = ("gpt-5.4-mini", _CTX_400K)
    GPT_54_NANO = ("gpt-5.4-nano", _CTX_400K)
    GPT_55 = ("gpt-5.5", _CTX_1M)
    GPT_55_PRO = ("gpt-5.5-pro", _CTX_1M)
    GEMINI_25_FLASH = ("gemini-2.5-flash", _CTX_1M)
    GEMINI_31_FLASH_LITE = ("gemini-3.1-flash-lite", _CTX_1M)
    GEMINI_35_FLASH = ("gemini-3.5-flash", _CTX_1M)
    GEMINI_31_PRO = ("gemini-3.1-pro-preview", _CTX_1M)
    BLOOM = ("bigscience/bloom", 1000)
    CUSTOM_ENDPOINT = ("custom-endpoint", 1)

    @property
    def limit(self):
        """Return the token limit value."""
        return self.value[1]


# Define a list of base model names that support tool calling
# These are matched against model names using prefix matching to handle date suffixes
# (e.g., "gpt-4.1-mini-2025-04-14" matches "gpt-4.1-mini")
_TOOL_CALLING_BASE_MODELS = frozenset(
    [
        GeminiModels.GEMINI_25_FLASH.value,
        GeminiModels.GEMINI_31_FLASH_LITE.value,
        GeminiModels.GEMINI_35_FLASH.value,
        GeminiModels.GEMINI_31_PRO.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_4O_MINI.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_MINI.value,
        OpenAIModels.GPT_41_NANO.value,
        OpenAIModels.GPT_5.value,
        OpenAIModels.GPT_5_MINI.value,
        OpenAIModels.GPT_5_NANO.value,
        OpenAIModels.GPT_54.value,
        OpenAIModels.GPT_54_MINI.value,
        OpenAIModels.GPT_54_NANO.value,
        OpenAIModels.GPT_55.value,
        OpenAIModels.GPT_55_PRO.value,
        MistralModels.MISTRAL_LARGE_LATEST.value,
        MistralModels.MISTRAL_MEDIUM_LATEST.value,
        MistralModels.MISTRAL_SMALL_LATEST.value,
        MistralModels.CODESTRAL_LATEST.value,
        AnthropicModels.CLAUDE_SONNET_46.value,
        AnthropicModels.CLAUDE_SONNET_45.value,
        AnthropicModels.CLAUDE_HAIKU_45.value,
        AnthropicModels.CLAUDE_OPUS_48.value,
    ]
)

# Additional patterns for models that support tool calling but aren't in the enum yet
# These are checked as prefixes (e.g., "gpt-4o-mini" matches "gpt-4o-mini-2024-07-18")
# Note: Only include models NOT in the enum/base models here. Models in base models are already handled
# by the prefix matching logic in supports_tool_calling().
_TOOL_CALLING_PREFIXES = frozenset(
    [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
    ]
)


def supports_tool_calling(model_name: str) -> bool:
    """Check if a model supports tool calling.

    This function uses prefix matching to handle model variants with date suffixes
    (e.g., "gpt-4.1-mini-2025-04-14" matches "gpt-4.1-mini").

    Args:
    ----
        model_name: The model name to check (e.g., "gpt-4.1-mini-2025-04-14")

    Returns:
    -------
        True if the model supports tool calling, False otherwise.

    """
    # First check exact match (for backward compatibility)
    if model_name in _TOOL_CALLING_BASE_MODELS:
        return True

    # Check if model_name starts with any base model name
    # This handles cases like "gpt-4.1-mini-2025-04-14" matching "gpt-4.1-mini"
    for base_model in _TOOL_CALLING_BASE_MODELS:
        if model_name.startswith(base_model + "-") or model_name == base_model:
            return True

    # Check against additional prefixes for models not yet in the enum
    for prefix in _TOOL_CALLING_PREFIXES:
        if model_name.startswith(prefix + "-") or model_name == prefix:
            return True

    return False


# For backward compatibility, keep TOOL_CALLING_MODELS as a set
# but it's now primarily used for documentation/exact matching
TOOL_CALLING_MODELS = _TOOL_CALLING_BASE_MODELS

# Define a list of base model names that support structured output
_STRUCTURED_OUTPUT_BASE_MODELS = frozenset(
    [
        GeminiModels.GEMINI_25_FLASH.value,
        GeminiModels.GEMINI_31_FLASH_LITE.value,
        GeminiModels.GEMINI_35_FLASH.value,
        GeminiModels.GEMINI_31_PRO.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_4O_MINI.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_MINI.value,
        OpenAIModels.GPT_41_NANO.value,
        OpenAIModels.GPT_5.value,
        OpenAIModels.GPT_5_MINI.value,
        OpenAIModels.GPT_5_NANO.value,
        OpenAIModels.GPT_54.value,
        OpenAIModels.GPT_54_MINI.value,
        OpenAIModels.GPT_54_NANO.value,
        OpenAIModels.GPT_55.value,
        OpenAIModels.GPT_55_PRO.value,
    ]
)


def supports_structured_output(model_name: str) -> bool:
    """Check if a model supports structured output.

    This function uses prefix matching to handle model variants with date suffixes.

    Args:
    ----
        model_name: The model name to check

    Returns:
    -------
        True if the model supports structured output, False otherwise.

    """
    # First check exact match
    if model_name in _STRUCTURED_OUTPUT_BASE_MODELS:
        return True

    # Check if model_name starts with any base model name
    for base_model in _STRUCTURED_OUTPUT_BASE_MODELS:
        if model_name.startswith(base_model + "-") or model_name == base_model:
            return True

    return False


# For backward compatibility
STRUCTURED_OUTPUT_MODELS = _STRUCTURED_OUTPUT_BASE_MODELS

# Models that only support temperature=1 (default); passing temperature=0 causes API errors.
# GPT-5+ and reasoning models use an internal multi-step process and reject custom temperature.
_TEMPERATURE_1_ONLY_PREFIXES = frozenset(
    [
        "gpt-5-",  # gpt-5-2025-08-07, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07
        "gpt-5.4",
        "gpt-5.5",
        "o1",  # o1, o1-mini, o1-pro
        "o3",  # o3, o3-mini
        "o4-mini",
    ]
)


def get_temperature_for_model(model_name: str) -> float:
    """Return temperature to use for a model. Some models (e.g. GPT-5) only support temperature=1.

    Args:
    ----
        model_name: The model name (e.g. "gpt-5-2025-08-07")

    Returns:
    -------
        Temperature value: 1.0 for models that reject temperature=0, else 0.0

    """
    if not model_name:
        return 0.0
    for prefix in _TEMPERATURE_1_ONLY_PREFIXES:
        if model_name.startswith(prefix) or model_name == prefix.rstrip("-"):
            return 1.0
    return 0.0


# For backward compatibility (even if not sure if this is needed)
OPENAI_MODELS = [model.value for model in OpenAIModels]
GEMINI_MODELS = [model.value for model in GeminiModels]
HUGGINGFACE_MODELS = [model.value for model in HuggingFaceModels]
XINFERENCE_MODELS = [model.value for model in XInferenceModels]

# For backward compatibility and easy lookup
TOKEN_LIMITS = {entry.value[0]: entry.limit for entry in TokenLimits}
