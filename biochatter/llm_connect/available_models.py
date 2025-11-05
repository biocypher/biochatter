"""Define the available models for the LLM connect module.

This module defines the available models for the LLM connect module and their
token limits.

The models are defined as an Enum, which allows for easy lookup of the model
token limits.
"""

from enum import Enum


class OpenAIModels(str, Enum):
    """Enum for OpenAI models."""

    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"  # further updated 3.5-turbo
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"  # gpt-4 turbo, 128k tokens
    GPT_4o = "gpt-4o"
    GPT_41 = "gpt-4.1"
    GPT_41_mini = "gpt-4.1-mini"
    GPT_5 = "gpt-5"
    GPT_5_mini = "gpt-5-mini"
    GPT_5_nano = "gpt-5-nano"


class GeminiModels(str, Enum):
    """Enum for Gemini models."""

    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_25_FLASH = "gemini-2.5-flash-preview-04-17"


class MistralModels(str, Enum):
    """Enum for Mistral models."""

    MISTRAL_LARGE_LATEST = "mistral-large-latest"


class AnthropicModels(str, Enum):
    """Enum for Anthropic models."""

    CLAUDE_3_7_SONNET_LATEST = "claude-3-7-sonnet-latest"
    CLAUDE_3_5_HAIKU_LATEST = "claude-3-5-haiku-latest"


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

    GPT_35_TURBO = ("gpt-3.5-turbo", 4000)
    GPT_35_TURBO_16K = ("gpt-3.5-turbo-16k", 16000)
    GPT_35_TURBO_1106 = ("gpt-3.5-turbo-1106", 16000)
    GPT_4 = ("gpt-4", 8000)
    GPT_4_32K = ("gpt-4-32k", 32000)
    GPT_4_1106_PREVIEW = ("gpt-4-1106-preview", 128000)
    GPT_4o = ("gpt-4o", 32000)
    GPT_41 = ("gpt-4.1", 1048576)
    GPT_41_mini = ("gpt-4.1-mini", 1048576)
    GPT_5 = ("gpt-5", 1048576)  # Assuming similar to GPT-4.1
    GPT_5_mini = ("gpt-5-mini", 1048576)
    GPT_5_nano = ("gpt-5-nano", 1048576)
    BLOOM = ("bigscience/bloom", 1000)
    CUSTOM_ENDPOINT = ("custom-endpoint", 1)
    GEMINI_20_FLASH = ("gemini-2.0-flash", 1000000)
    GEMINI_25_FLASH = ("gemini-2.5-flash-preview-04-17", 1048576)

    @property
    def limit(self):
        """Return the token limit value."""
        return self.value[1]


# Define a list of base model names that support tool calling
# These are matched against model names using prefix matching to handle date suffixes
# (e.g., "gpt-4.1-mini-2025-04-14" matches "gpt-4.1-mini")
_TOOL_CALLING_BASE_MODELS = frozenset(
    [
        GeminiModels.GEMINI_20_FLASH.value,
        GeminiModels.GEMINI_25_FLASH.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_mini.value,
        OpenAIModels.GPT_5.value,
        OpenAIModels.GPT_5_mini.value,
        OpenAIModels.GPT_5_nano.value,
        MistralModels.MISTRAL_LARGE_LATEST.value,
        AnthropicModels.CLAUDE_3_7_SONNET_LATEST.value,
        AnthropicModels.CLAUDE_3_5_HAIKU_LATEST.value,
    ]
)

# Additional patterns for models that support tool calling but aren't in the enum yet
# These are checked as prefixes (e.g., "gpt-4o-mini" matches "gpt-4o-mini-2024-07-18")
# Note: Only include models NOT in the enum/base models here. Models in base models are already handled
# by the prefix matching logic in supports_tool_calling().
_TOOL_CALLING_PREFIXES = frozenset(
    [
        "gpt-4o-mini",  # e.g., gpt-4o-mini-2024-07-18 (not in enum, only has date variants)
        "claude-3-5-sonnet",  # e.g., claude-3-5-sonnet-20240620 (not in enum, only has date variants)
        "claude-3-opus",  # e.g., claude-3-opus-20240229 (not in enum, only has date variants)
    ]
)


def supports_tool_calling(model_name: str) -> bool:
    """Check if a model supports tool calling.
    
    This function uses prefix matching to handle model variants with date suffixes
    (e.g., "gpt-4.1-mini-2025-04-14" matches "gpt-4.1-mini").
    
    Args:
        model_name: The model name to check (e.g., "gpt-4.1-mini-2025-04-14")
    
    Returns:
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
        GeminiModels.GEMINI_20_FLASH.value,
        GeminiModels.GEMINI_25_FLASH.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_mini.value,
        OpenAIModels.GPT_5.value,
        OpenAIModels.GPT_5_mini.value,
        OpenAIModels.GPT_5_nano.value,
    ]
)

# Additional patterns for structured output models
# Note: Only include models NOT in the enum/base models here. Models in base models are already handled
# by the prefix matching logic in supports_structured_output().
_STRUCTURED_OUTPUT_PREFIXES = frozenset(
    [
        "gpt-4o-mini",  # e.g., gpt-4o-mini-2024-07-18 (not in enum, only has date variants)
    ]
)


def supports_structured_output(model_name: str) -> bool:
    """Check if a model supports structured output.
    
    This function uses prefix matching to handle model variants with date suffixes.
    
    Args:
        model_name: The model name to check
    
    Returns:
        True if the model supports structured output, False otherwise.
    """
    # First check exact match
    if model_name in _STRUCTURED_OUTPUT_BASE_MODELS:
        return True
    
    # Check if model_name starts with any base model name
    for base_model in _STRUCTURED_OUTPUT_BASE_MODELS:
        if model_name.startswith(base_model + "-") or model_name == base_model:
            return True
    
    # Check against additional prefixes
    for prefix in _STRUCTURED_OUTPUT_PREFIXES:
        if model_name.startswith(prefix + "-") or model_name == prefix:
            return True
    
    return False


# For backward compatibility
STRUCTURED_OUTPUT_MODELS = _STRUCTURED_OUTPUT_BASE_MODELS

# For backward compatibility (even if not sure if this is needed)
OPENAI_MODELS = [model.value for model in OpenAIModels]
GEMINI_MODELS = [model.value for model in GeminiModels]
HUGGINGFACE_MODELS = [model.value for model in HuggingFaceModels]
XINFERENCE_MODELS = [model.value for model in XInferenceModels]

# For backward compatibility and easy lookup
TOKEN_LIMITS = {
    OpenAIModels.GPT_35_TURBO.value: TokenLimits.GPT_35_TURBO.limit,
    OpenAIModels.GPT_35_TURBO_16K.value: TokenLimits.GPT_35_TURBO_16K.limit,
    OpenAIModels.GPT_35_TURBO_1106.value: TokenLimits.GPT_35_TURBO_1106.limit,
    OpenAIModels.GPT_4.value: TokenLimits.GPT_4.limit,
    OpenAIModels.GPT_4_32K.value: TokenLimits.GPT_4_32K.limit,
    OpenAIModels.GPT_4_1106_PREVIEW.value: TokenLimits.GPT_4_1106_PREVIEW.limit,
    HuggingFaceModels.BLOOM.value: TokenLimits.BLOOM.limit,
    XInferenceModels.CUSTOM_ENDPOINT.value: TokenLimits.CUSTOM_ENDPOINT.limit,
    GeminiModels.GEMINI_20_FLASH.value: TokenLimits.GEMINI_20_FLASH.limit,
    GeminiModels.GEMINI_25_FLASH.value: TokenLimits.GEMINI_25_FLASH.limit,
    OpenAIModels.GPT_41.value: TokenLimits.GPT_41.limit,
    OpenAIModels.GPT_41_mini.value: TokenLimits.GPT_41_mini.limit,
    OpenAIModels.GPT_5.value: TokenLimits.GPT_5.limit,
    OpenAIModels.GPT_5_mini.value: TokenLimits.GPT_5_mini.limit,
    OpenAIModels.GPT_5_nano.value: TokenLimits.GPT_5_nano.limit,
}
