#!/usr/bin/env python
# coding: utf-8


from __future__ import annotations

from biochatter.llm_connect import GptConversation


conversation = GptConversation(
	model_name="gpt-4o-mini",
	prompts={},
)
conversation.set_api_key(api_key="sk-proj-...")

response, token_usage, correction = conversation.query("How many crimes happened in total?")

print(response)


from biochatter.prompts import BioCypherPromptEngine


def create_conversation():
	conversation = GptConversation(model_name="gpt-4o-mini", prompts={})
	conversation.set_api_key(api_key="sk-proj-...")
	return conversation

prompt_engine = BioCypherPromptEngine(
	schema_config_or_info_path="[~]/pole/config/schema_config.yaml",
	#
	conversation_factory=create_conversation,
)

cypher_query = prompt_engine.generate_query(
	question="How many crimes happened in total?",
	query_language="Cypher"
)

dbAgent = DatabaseAgent(connection_args={
	"host": "localhost",
	#
	"port": "7687",	
	#
	"user": "neo4j",
	#
	"password": "...",
})

dbAgent.connect()


dbAgentConnectedBool = dbAgent.is_connected()


k = int(3)


results = dbAgent.driver._driver.query(querystr=cypher_query)

if results is None or len(results) == 0 or results[0] is None:
	query_results = []
query_results = dbAgent._build_response(
	results=results[0],
	cypher_query=cypher_query,
	results_num=k,
)


### START Code amended from lines 1--13 ['"""Abstract base classes for API interaction components.'] and 130--157 ["def summarise_results(..."] of "[~]/biochatter/biochatter/api_agent/base/agent_abc.py"
#
#
"""Abstract base classes for API interaction components.

Provides base classes for query builders, fetchers, and interpreters used in
API interactions and result processing.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, create_model

from biochatter.llm_connect import Conversation


def summarise_results_fromBaseInterpreter(    	
    question: str,
    conversation_factory: Callable,
    response_text: str,
) -> str:
    """Summarise an answer based on the given parameters.

    Args:
    ----
        question (str): The question that was asked.

        conversation_factory (Callable): A function that creates a
            BioChatter conversation.

        response_text (str): The response.text returned from the request.

    Returns:
    -------
        A summary of the answer.

    Todo:
    ----
        Genericise (remove file path and n_lines parameters, and use a
        generic way to get the results). The child classes should manage the
        specifics of the results.

    """
#
#
### END Code amended from lines 1--13 and 130--157 of "[~]/biochatter/biochatter/api_agent/base/agent_abc.py".


### START Code extracted-then-amended from lines 1--7 ["def summarise_results(..."] and 86--140 ["def execute(self, question: str) ..."] of "[~]/biochatter/biochatter/api_agent/base/api_agent.py", and REF "https://biocypher.org/BioChatter/quickstart" --> "API Integration" --> "Interpret the results using the LLM": "result = [APIA]gent.execute(...)"
#
#
"""Base API agent module."""

from collections.abc import Callable

from pydantic import BaseModel


def summarise_results_standalone(	
    question: str,
    response_text: str,
) -> str | None:
    """Summarise the retrieved results to extract the answer to the question."""
    try:
        return summarise_results_fromBaseInterpreter(
            question=question,
            conversation_factory=create_conversation,
            response_text=response_text,
        )
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def execute(question: str, response_text: str) -> str | None:

    """Wrapper that uses class methods to execute the API agent logic. Consists
    of 1) query generation, 2) query submission, 3) results fetching, and
    4) answer extraction. The final answer is stored in the final_answer
    attribute.

    Args:
    ----
        question (str): The question to be answered.

    """
    # Extract answer from results
    try:

        final_answer = summarise_results_standalone(question, response_text)

        if not final_answer:
            raise ValueError("Failed to extract answer from results.")
    except ValueError as e:
        print(e)

    return final_answer
#
#
### END Code extracted-then-amended from lines 1--7 and 86--140 of "[~]/biochatter/biochatter/api_agent/base/api_agent.py".
#
result = execute(cypher_query, query_results)

print(result)


# END OF FILE.
