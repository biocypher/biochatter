"""BioCypher structured prompt module.

This module provides functionality for generating structured prompts and parsing
responses for knowledge graph queries using Pydantic models.
"""

from typing import Literal

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel


def list_output_model(allowed_choices: list[str]) -> PydanticOutputParser:
    """Create a dynamic output parser for structured responses.

    This function generates a Pydantic output parser that can validate and parse
    responses containing selections from a predefined list of allowed choices.

    Args:
        allowed_choices: A list of strings representing the allowed values that
            can be selected in the response.

    Returns:
        A PydanticOutputParser configured with a dynamically created model that
        validates responses against the allowed choices.

    """
    # Dynamically create a Literal type using the allowed choices.
    literal_type = Literal[tuple(allowed_choices)]

    class OutputModel(BaseModel):
        """Pydantic model for structured output validation.

        Attributes:
            result: A list of values, each of which must be one of the allowed choices.
                   The list cannot be empty.

        """

        result: list[literal_type]

    return OutputModel
