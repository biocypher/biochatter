"""Exceptions raised by LLM connection and query operations."""


class _LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMInitializationError(_LLMError):
    """Raised when set_api_key() fails to initialize the chat client."""


class LLMConnectionError(_LLMError):
    """Raised when an LLM API call fails to return a usable response."""
