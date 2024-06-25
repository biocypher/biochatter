from abc import ABC
from typing import Optional
from pydantic import BaseModel, Field


class BaseAPIQuery(BaseModel, ABC):
    """
    Abstract base class for any API query request, providing a generic
    structure that can be extended for specific APIs.
    """

    url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for the API endpoint. "
            "Must be overridden by subclasses."
        ),
    )
    cmd: Optional[str] = Field(
        default="Put",
        description=(
            "Command to execute against the API. "
            "'Put' for submitting a query, 'Get' for retrieving results."
        ),
    )
