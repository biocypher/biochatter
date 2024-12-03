import json
import logging
from collections.abc import Callable
from datetime import datetime

import neo4j_utils as nu
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from biochatter.langgraph_agent_base import (
    EXECUTE_TOOL_NODE,
    ReflexionAgent,
    ReflexionAgentLogger,
    ReflexionAgentResult,
    ResponderWithRetries,
)

logger = logging.getLogger(__name__)


class KGQueryReflexionAgentLogger(ReflexionAgentLogger):
    def __init__(self) -> None:
        super().__init__()

    def log_step_message(self, step: int, node_name: str, output: BaseMessage):
        try:
            parsed_output = self.parser.invoke(output)
            self._log_message(f"## {step}, {node_name}")
            self._log_message(f'Answer: {parsed_output[0]["args"]["answer"]}')
            self._log_message(
                f'Reflection | Improving: {parsed_output[0]["args"]["reflection"]}',
            )
            self._log_message("Reflection | Search Queries:")
            for i, sq in enumerate(parsed_output[0]["args"][SEARCH_QUERIES]):
                self._log_message(f"{i+1}: {sq}")
            if REVISED_QUERY in parsed_output[0]["args"]:
                self._log_message("Reflection | Revised Query:")
                self._log_message(parsed_output[0]["args"][REVISED_QUERY])
            self._log_message(
                "-------------------------------- Node Output --------------------------------",
            )
        except Exception:
            self._log_message(str(output)[:100] + " ...", "error")

    def log_final_result(self, final_result: ReflexionAgentResult) -> None:
        self._log_message(
            "\n\n-------------------------------- Final Generated Response --------------------------------",
        )
        obj = vars(final_result)
        self._log_message(json.dumps(obj))


ANSWER = "answer"
SEARCH_QUERIES = "search_queries"
SEARCH_QUERIES_DESCRIPTION = "query for graph database"
REVISED_QUERY = "revised_query"
REVISED_QUERY_DESCRIPTION = "Revised query based on the reflection."
SCORE_DESCRIPTION = (
    "the score for the query based on its query result"
    " and relevance to the user's question,"
    " with 0 representing the lowest score and 10 representing the highest score."
)


class GenerateQuery(BaseModel):
    """Generate the query."""

    answer: str = Field(
        description="Cypher query for graph database according to user's question.",
    )
    reflection: str = Field(
        description="Your reflection on the initial answer, critique of what to improve",
    )
    search_queries: list[str] = Field(description=SEARCH_QUERIES_DESCRIPTION)


class ReviseQuery(GenerateQuery):
    """Revise your previous query according to your question."""

    revised_query: str = Field(description=REVISED_QUERY_DESCRIPTION)
    score: str = Field(description=SCORE_DESCRIPTION)


class KGQueryReflexionAgent(ReflexionAgent):
    def __init__(
        self,
        conversation_factory: Callable,
        connection_args: dict[str, str],
        query_lang: str | None = "Cypher",
        max_steps: int | None = 20,
    ):
        r"""LLM agent reflexion framework:

        start -> draft -> execute tool -> revise -> evaluation -> end
                            /|\                        |
                             ---------------------------

        Adapts base class to build and refine a knowledge graph query, default
        language Cypher. Currently hardcoded to connect to Neo4j for the KG
        query implementation.

        Args:
        ----
            conversation_factory: function to return the Conversation to use for
                the LLM connection

            connection_args: connection arguments for connecting to the database

            query_lang: graph query language to use

            max_steps: the maximum number of steps to execute in the graph

        """
        super().__init__(
            conversation_factory,
            max_steps,
            agent_logger=KGQueryReflexionAgentLogger(),
        )
        self.actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "As a senior biomedical researcher and graph database expert, "
                        f"your task is to generate '{query_lang}' queries to extract data from our graph database based on the user's question. "
                        """Current time {time}. {instruction}"""
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    (
                        "Note: 1. Only generate query according to the user's question above.\n"
                        "2. Please limit the results to a maximum of 30 items"
                    ),
                ),
            ],
        ).partial(time=lambda: datetime.now().isoformat())
        self.parser = JsonOutputToolsParser(return_id=True)
        self.connection_args = connection_args
        self.neodriver = None

    def _connect_db(self):
        if self.neodriver is not None:
            return
        try:
            db_uri = "bolt://" + self.connection_args.get("host") + ":" + self.connection_args.get("port")
            self.neodriver = nu.Driver(
                db_name=self.connection_args.get("db_name") or "neo4j",
                db_uri=db_uri,
            )
        except Exception as e:
            logger.error(e)

    def _query_graph_database(self, query: str):
        """Try to execute the query in Neo4j and return the result.

        Args:
        ----
            query: the query string

        """
        self._connect_db()
        try:
            return self.neodriver.query(query)
        except Exception as e:
            logger.error(str(e))
            return []  # empty result

    def _create_initial_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        llm: ChatOpenAI = self.conversation.chat
        initial_chain = self.actor_prompt_template.partial(
            instruction=prompt if prompt is not None else "",
        ) | llm.bind_tools(
            tools=[GenerateQuery],
            tool_choice="GenerateQuery",
        )
        validator = PydanticToolsParser(tools=[GenerateQuery])
        return ResponderWithRetries(runnable=initial_chain, validator=validator)

    def _create_revise_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        revision_instruction = """
        Revise your previous query using the query result and follow the guidelines:
        1. If you consistently obtain empty results, please consider removing constraints such as relationship constraints to try to obtain a result.
        2. You should use previous critique to improve your query.
        3. Only generate a query without returning any other text.
        """
        llm: ChatOpenAI = self.conversation.chat
        revision_chain = self.actor_prompt_template.partial(
            instruction=revision_instruction,
        ) | llm.bind_tools(
            tools=[ReviseQuery],
            tool_choice="ReviseQuery",
        )
        validator = PydanticToolsParser(tools=[ReviseQuery])
        return ResponderWithRetries(
            runnable=revision_chain,
            validator=validator,
        )

    def _tool_function(self, state: list[BaseMessage]):
        tool_message: AIMessage = state[-1]
        parsed_tool_messages = self.parser.invoke(tool_message)
        results = []
        for parsed_message in parsed_tool_messages:
            try:
                parsed_args = parsed_message["args"]
                query = (
                    parsed_args[REVISED_QUERY]
                    if REVISED_QUERY in parsed_args
                    else (parsed_args[REVISED_QUERY_DESCRIPTION] if REVISED_QUERY_DESCRIPTION in parsed_args else None)
                )
                if query is not None:
                    result = self._query_graph_database(query)
                    results.append({"query": query, "result": result[0]})
                    continue
                queries = (
                    parsed_args[SEARCH_QUERIES]
                    if SEARCH_QUERIES in parsed_args
                    else parsed_args[SEARCH_QUERIES_DESCRIPTION]
                )
                queries = queries if len(queries) > 0 else [parsed_args[ANSWER]]
                for query in queries:
                    result = self._query_graph_database(query)
                    results.append(
                        {
                            "query": query,
                            "result": result[0] if len(result) > 0 else [],
                        },
                    )
            except Exception as e:
                logger.error(f"Error occurred: {e!s}")

        content = None
        if len(results) > 1:
            # If there are multiple results, we only return
            # the first non-empty result
            for res in results:
                if res["result"] and len(res["result"]) > 0:
                    content = json.dumps(res)
        if content is None:
            content = json.dumps(results[0]) if len(results) > 0 else ""
        return ToolMessage(
            content=content,
            tool_call_id=parsed_message["id"],
        )

    @staticmethod
    def _get_last_tool_results_num(state: list[BaseMessage]):
        i = 0
        for m in state[::-1]:
            if not isinstance(m, ToolMessage):
                continue
            message: ToolMessage = m
            logger.info(f"query result: {message.content}")
            results = (
                json.loads(message.content)
                if message.content is not None and len(message.content) > 0
                else {"result": []}
            )
            empty = True
            if len(results["result"]) > 0:
                # check if it is really not empty, remove the case: {"result": [{"c.name": None}]}
                for res in results["result"]:
                    for k in res.keys():
                        if res[k] is None:
                            continue
                        if isinstance(res[k], str) and (res[k] == "None" or res[k] == "null"):
                            continue
                        empty = False
                        break
                    if not empty:
                        break
            return len(results["result"]) if not empty else 0

        return 0

    def _get_last_score(self, state: list[BaseMessage]) -> int | None:
        for m in state[::-1]:
            if not isinstance(m, AIMessage):
                continue
            message: AIMessage = m
            parsed_msg = self.parser.invoke(message)
            try:
                score = parsed_msg[0]["args"]["score"]
                return int(score)
            except Exception:
                return None
        return None

    def _should_continue(self, state: list[BaseMessage]):
        res = super()._should_continue(state)
        if res == END:
            return res
        score = self._get_last_score(state)
        if score is not None and score >= 7:
            return END
        query_results_num = KGQueryReflexionAgent._get_last_tool_results_num(
            state,
        )
        return END if query_results_num > 0 else EXECUTE_TOOL_NODE

    def _parse_final_result(
        self,
        messages: list[BaseMessage],
    ) -> ReflexionAgentResult:
        output = messages[-1]
        result = self.parser.invoke(output)[0]["args"]
        tool_result = ReflexionAgent._get_last_tool_result(messages)
        return ReflexionAgentResult(
            answer=result["answer"] if "answer" in result else None,
            tool_result=tool_result,
        )
