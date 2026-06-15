"""Exceptions raised by LLM connection and query operations."""


class LLMConnectionError(RuntimeError):
    """Raised when an LLM API call fails to return a usable response."""

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
