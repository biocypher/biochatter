"""Prompt templates and instruction strings for BioChatter LLM agents.

This module defines reusable prompt templates and instruction strings for
various agent roles in the BioChatter framework.
"""

import datetime

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# === Instruction Strings ===

FIRST_INSTRUCTION: str = (
    "Provide a detailed answer (don't surpass 4000 words)"
)

REVISE_INSTRUCTIONS: str = (
    "Revise your previous answer using the new information.\n"
    "- You should use the previous critique to add important information to your answer.\n"
    "    - You MUST include numerical citations in your revised answer to ensure it can be verified.\n"
    '    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:\n'
    "        - [1] paper reference\n"
    "        - [2] paper reference\n"
    "- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 2000 words."
)

# === Prompt Template Factories ===

ACTOR_SYSTEM_PROMPT = (
    "You are expert computational biologist.\n"
    "Current time: {time}\n\n"
    "1. {first_instruction}\n"
    "2. Reflect and critique your answer. Be severe to maximize improvement.\n"
    "3. After the reflection, **list 1-3 search queries on pubmed separately** for researching improvements. "
    "Do not include them inside the reflection.\n"
)

def get_actor_prompt_template(first_instruction: str) -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for the actor agent with a given instruction."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ACTOR_SYSTEM_PROMPT.format(first_instruction=first_instruction, time="{time}"),
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's question above using the required format."),
        ]
    ).partial(
        time=lambda: datetime.datetime.now().isoformat(),
    )

TOOL_FORMULATOR_SYSTEM_PROMPT = (
    "You are a tool call formulator. Given the original question, the generated queries and the available tools, "
    "you parametrize the tool calls. You should create a tool call for each query."
)
TOOL_FORMULATOR_USER_PROMPT = (
    "Original question: {current_question}\nGenerated queries:\n {search_queries}"
)

TOOL_FORMULATOR_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", TOOL_FORMULATOR_SYSTEM_PROMPT),
        ("user", TOOL_FORMULATOR_USER_PROMPT),
    ]
)

# === Pre-configured Templates ===

FIRST_RESPONDER_PROMPT_TEMPLATE: ChatPromptTemplate = get_actor_prompt_template(FIRST_INSTRUCTION)
REVISOR_PROMPT_TEMPLATE: ChatPromptTemplate = get_actor_prompt_template(REVISE_INSTRUCTIONS)


# === Pydantic Models ===

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(
        description="A detailed answer to the question (don't surpass 4000 words)")
    search_queries: list[str] = Field(
        description="1-3 search queries on pubmed for researching improvements to address the critique of your current answer."
    )
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )
