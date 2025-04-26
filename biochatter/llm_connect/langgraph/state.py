"""Definition of the states."""

import operator
from typing import Annotated

from typing_extensions import TypedDict


class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    iteration: int = 0
    current_question: str | None
    search_queries: list[str]
    tool_calls: list
