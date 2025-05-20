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
    BLOOM = ("bigscience/bloom", 1000)
    CUSTOM_ENDPOINT = ("custom-endpoint", 1)
    GEMINI_20_FLASH = ("gemini-2.0-flash", 1000000)
    GEMINI_25_FLASH = ("gemini-2.5-flash-preview-04-17", 1048576)

    @property
    def limit(self):
        """Return the token limit value."""
        return self.value[1]


# Define a list of models that support tool calling
TOOL_CALLING_MODELS = frozenset(
    [
        GeminiModels.GEMINI_20_FLASH.value,
        GeminiModels.GEMINI_25_FLASH.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_mini.value,
        MistralModels.MISTRAL_LARGE_LATEST.value,
        AnthropicModels.CLAUDE_3_7_SONNET_LATEST.value,
        AnthropicModels.CLAUDE_3_5_HAIKU_LATEST.value,
    ]
)

# Define a list of models that support structured output
STRUCTURED_OUTPUT_MODELS = frozenset(
    [
        GeminiModels.GEMINI_20_FLASH.value,
        GeminiModels.GEMINI_25_FLASH.value,
        OpenAIModels.GPT_4o.value,
        OpenAIModels.GPT_41.value,
        OpenAIModels.GPT_41_mini.value,
    ]
)

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
}
