# straight copy from BioCypher repo
# TODO: have a utils package for both repos

import json
import re
from pydantic import BaseModel
from collections.abc import Generator, ItemsView, Iterable, KeysView, Mapping, ValuesView
from typing import (
    Any,
)

import stringcase

__all__ = ["LIST_LIKE", "SIMPLE_TYPES", "ensure_iterable", "to_list"]

SIMPLE_TYPES = (
    bytes,
    str,
    int,
    float,
    bool,
    type(None),
)

LIST_LIKE = (
    list,
    set,
    tuple,
    Generator,
    ItemsView,
    KeysView,
    Mapping,
    ValuesView,
)


def to_list(value: Any) -> list:
    """Ensures that ``value`` is a list."""
    if isinstance(value, LIST_LIKE):
        value = list(value)

    else:
        value = [value]

    return value


def ensure_iterable(value: Any) -> Iterable:
    """Returns iterables, except strings, wraps simple types into tuple."""
    return value if isinstance(value, LIST_LIKE) else (value,)


# string conversion, adapted from Biolink Model Toolkit
lowercase_pattern = re.compile(r"[a-zA-Z]*[a-z][a-zA-Z]*")
underscore_pattern = re.compile(r"(?<!^)(?=[A-Z][a-z])")


def from_pascal(s: str, sep: str = " ") -> str:
    underscored = underscore_pattern.sub(sep, s)
    lowercased = lowercase_pattern.sub(
        lambda match: match.group(0).lower(),
        underscored,
    )
    return lowercased


def pascalcase_to_sentencecase(s: str) -> str:
    """Convert PascalCase to sentence case.

    Args:
    ----
        s: Input string in PascalCase

    Returns:
    -------
        string in sentence case form

    """
    return from_pascal(s, sep=" ")


def snakecase_to_sentencecase(s: str) -> str:
    """Convert snake_case to sentence case.

    Args:
    ----
        s: Input string in snake_case

    Returns:
    -------
        string in sentence case form

    """
    return stringcase.sentencecase(s).lower()


def sentencecase_to_snakecase(s: str) -> str:
    """Convert sentence case to snake_case.

    Args:
    ----
        s: Input string in sentence case

    Returns:
    -------
        string in snake_case form

    """
    return stringcase.snakecase(s).lower()


def sentencecase_to_pascalcase(s: str) -> str:
    """Convert sentence case to PascalCase.

    Args:
    ----
        s: Input string in sentence case

    Returns:
    -------
        string in PascalCase form

    """
    return re.sub(r"(?:^| )([a-zA-Z])", lambda match: match.group(1).upper(), s)


def to_lower_sentence_case(s: str) -> str:
    """Convert any string to lower sentence case. Works with snake_case,
    PascalCase, and sentence case.

    Args:
    ----
        s: Input string

    Returns:
    -------
        string in lower sentence case form

    """
    if "_" in s:
        return snakecase_to_sentencecase(s)
    if " " in s:
        return s.lower()
    if s[0].isupper():
        return pascalcase_to_sentencecase(s)
    return s


def extract_json(text: str) -> str:
    """Given a string containing a ```json ... ``` code fence,
    return only the JSON text inside the fences.
    """
    # Regex to capture the JSON object inside a ```json ... ``` block
    pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1)


def pydantic_manual_validator(raw: str, model_cls: BaseModel):
    try:
        json_text = extract_json(raw)
        data = json.loads(json_text)
        return model_cls.parse_obj(data)
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}") from e
