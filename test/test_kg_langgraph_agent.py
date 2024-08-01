from types import UnionType
from typing import Any, List
from collections.abc import Callable

from langgraph.graph import END
from langchain_core.messages import AIMessage, BaseMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
import pytest
import shortuuid

from biochatter.kg_langgraph_agent import (
    ReviseQuery,
    GenerateQuery,
    KGQueryReflexionAgent,
)
from biochatter.langgraph_agent_base import ResponderWithRetries


class InitialResponder:
    def invoke(self, msg_obj: dict[str, List[BaseMessage]]) -> BaseMessage:
        msg = AIMessage(content="initial test")
        id = "call_" + shortuuid.uuid()
        msg.additional_kwargs = {
            "tool_calls": [
                {
                    "id": id,
                    "function": {
                        "arguments": '{"answer":"MATCH (g:gene {name: \'EOMES\'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5",\
                                    "reflection":"The query correctly identifies genes regulated by EOMES, which can help determine its primary and secondary functions based on the regulated genes\' roles.",\
                                        "search_queries":["MATCH (g:gene {name: \'EOMES\'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5"]}',
                        "name": "GenerateQuery",
                    },
                    "type": "function",
                }
            ]
        }
        msg.id = id
        msg.tool_calls = [
            {
                "name": "GenerateQuery",
                "args": {
                    "answer": "MATCH (g:gene {name: 'EOMES'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5",
                    "reflection": "The query correctly identifies genes regulated by EOMES, which can help determine its primary and secondary functions based on the regulated genes' roles.",
                    "search_queries": [
                        "MATCH (g:gene {name: 'EOMES'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5"
                    ],
                },
                "id": id,
            }
        ]
        return msg


class ReviseResponder:
    _enter_count = 0

    def invoke(self, msg_obj: dict[str, List[BaseMessage]]) -> BaseMessage:
        ReviseResponder._enter_count += 1
        if ReviseResponder._enter_count == 1:
            return AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call_wTO40b9rXHhNhxNEG3IHn3Nj",
                            "function": {
                                "arguments": '{"answer":"MATCH (g:gene {name: \'EOMES\'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5","reflection":"The initial query returned no results. To improve the query, I will remove the relationship constraint and try to find any connections or properties related to EOMES that might indicate its primary and secondary functions.","search_queries":["MATCH (g:gene {name: \'EOMES\'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5"],"revised_query":"MATCH (g:gene {name: \'EOMES\'})-[]->(c) RETURN DISTINCT c.name, labels(c) LIMIT 5"}',
                                "name": "ReviseQuery",
                            },
                            "type": "function",
                        }
                    ]
                },
                response_metadata={
                    "token_usage": {
                        "completion_tokens": 146,
                        "prompt_tokens": 419,
                        "total_tokens": 565,
                    },
                    "model_name": "gpt-4o-2024-05-13",
                    "system_fingerprint": "fp_abc28019ad",
                    "prompt_filter_results": [
                        {
                            "prompt_index": 0,
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "self_harm": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                                "sexual": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                                "violence": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                            },
                        }
                    ],
                    "finish_reason": "stop",
                    "logprobs": None,
                    "content_filter_results": {},
                },
                id="run-ad95b309-4f99-4886-b70f-bcb8b59f537f-0",
                tool_calls=[
                    {
                        "name": "ReviseQuery",
                        "args": {
                            "answer": "MATCH (g:gene {name: 'EOMES'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5",
                            "reflection": "The initial query returned no results. To improve the query, I will remove the relationship constraint and try to find any connections or properties related to EOMES that might indicate its primary and secondary functions.",
                            "search_queries": [
                                "MATCH (g:gene {name: 'EOMES'})-[:regulate]->(c:gene) RETURN DISTINCT c.name LIMIT 5"
                            ],
                            "revised_query": "MATCH (g:gene {name: 'EOMES'})-[]->(c) RETURN DISTINCT c.name, labels(c) LIMIT 5",
                        },
                        "id": "call_wTO40b9rXHhNhxNEG3IHn3Nj",
                    }
                ],
            )
        else:
            return AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call_Eciqf2ZviAjYinzi4zpgiNej",
                            "function": {
                                "arguments": '{"answer":"MATCH (g:gene {name: \'EOMES\'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;","reflection":"The revised query successfully returned results by removing the relationship constraint. This indicates that EOMES is connected to other genes, but the specific \'regulate\' relationship might not be explicitly defined in the database. The revised query is more flexible and provides the needed information.","search_queries":["MATCH (g:gene {name: \'EOMES\'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;"],"revised_query":"MATCH (g:gene {name: \'EOMES\'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;"}',
                                "name": "ReviseQuery",
                            },
                            "type": "function",
                        }
                    ]
                },
                response_metadata={
                    "token_usage": {
                        "completion_tokens": 150,
                        "prompt_tokens": 646,
                        "total_tokens": 796,
                    },
                    "model_name": "gpt-4o-2024-05-13",
                    "system_fingerprint": "fp_abc28019ad",
                    "prompt_filter_results": [
                        {
                            "prompt_index": 0,
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "self_harm": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                                "sexual": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                                "violence": {
                                    "filtered": False,
                                    "severity": "safe",
                                },
                            },
                        }
                    ],
                    "finish_reason": "stop",
                    "logprobs": None,
                    "content_filter_results": {},
                },
                id="run-8af61ddf-14b7-4280-826a-d07984dcfaa8-0",
                tool_calls=[
                    {
                        "name": "ReviseQuery",
                        "args": {
                            "answer": "MATCH (g:gene {name: 'EOMES'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;",
                            "reflection": "The revised query successfully returned results by removing the relationship constraint. This indicates that EOMES is connected to other genes, but the specific 'regulate' relationship might not be explicitly defined in the database. The revised query is more flexible and provides the needed information.",
                            "search_queries": [
                                "MATCH (g:gene {name: 'EOMES'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;"
                            ],
                            "revised_query": "MATCH (g:gene {name: 'EOMES'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;",
                        },
                        "id": "call_Eciqf2ZviAjYinzi4zpgiNej",
                    }
                ],
            )


class KGQueryReflexionAgentMock(KGQueryReflexionAgent):
    _query_graph_db_count = 0

    def __init__(
        self,
        conversation_factory: Callable[..., Any],
        connection_args: dict[str, str],
        query_lang: str | None = "Cypher",
        recursion_limit: int | None = 20,
    ):
        super().__init__(
            conversation_factory, connection_args, query_lang, recursion_limit
        )

    def _create_initial_responder(
        self, prompt: str | None = None
    ) -> ResponderWithRetries:
        runnable = InitialResponder()
        validator = PydanticToolsParser(tools=[GenerateQuery])
        return ResponderWithRetries(
            runnable=runnable,
            validator=validator,
        )

    def _create_revise_responder(
        self, prompt: str | None = None
    ) -> ResponderWithRetries:
        runnable = ReviseResponder()
        validator = PydanticToolsParser(tools=[ReviseQuery])
        return ResponderWithRetries(runnable=runnable, validator=validator)

    def _connect_db(self):
        pass

    def _query_graph_database(self, query: str):
        KGQueryReflexionAgentMock._query_graph_db_count += 1
        if KGQueryReflexionAgentMock._query_graph_db_count == 1:
            return [[{"c.name": None}]]
        else:
            return [
                [
                    {"c.name": None, "labels(c)": ["chr_chain"]},
                    {
                        "c.name": "EOMES-201",
                        "labels(c)": ["coding_elements", "transcript"],
                    },
                    {
                        "c.name": "EOMES-203",
                        "labels(c)": ["coding_elements", "transcript"],
                    },
                    {
                        "c.name": "EOMES-202",
                        "labels(c)": ["coding_elements", "transcript"],
                    },
                    {
                        "c.name": None,
                        "labels(c)": ["ontology", "cell_line_or_tissue"],
                    },
                ]
            ]


class ChatOpenAIMock:
    def __init__(self) -> None:
        self.chat = None


@pytest.fixture
def kgQueryAgent():
    return KGQueryReflexionAgentMock(
        connection_args={"host": "localhost", "port": "7687"},
        conversation_factory=lambda: ChatOpenAIMock(),
    )


def test_execute(kgQueryAgent):
    question = "What genes does EOMES primarily regulate?"
    agent_result = kgQueryAgent.execute(question=question)
    assert (
        agent_result.answer
        == "MATCH (g:gene {name: 'EOMES'})-->(c:gene) RETURN DISTINCT c.name LIMIT 5;"
    )
