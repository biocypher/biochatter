from typing import Any, Optional, Dict
from datetime import datetime
from collections.abc import Callable
import json
import logging

from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain.output_parsers.openai_tools import (
    PydanticToolsParser,
    JsonOutputToolsParser,
)

from biochatter.langgraph_agent_base import (
    END_NODE,
    ReflexionAgent,
    ReflexionAgentLogger,
    ReflexionAgentResult,
    ResponderWithRetries,
)

logger = logging.getLogger(__name__)


ANSWER = "answer"
REVISED_ANSWER = "revised_answer"
REVISED_ANSWER_DESCRIPTION = "Revised answer based on the reflection."
SCORE_DESCRIPTION = (
    "the score for the answer based on the relevance to the user's question, "
    "with 0 representing the lowest score and 10 representing the highest score."
)

DRAFT_NODE = "draft"
REVISE_NODE = "revise"


class CorrectionReflexionAgentLogger(ReflexionAgentLogger):
    def __init__(self) -> None:
        super().__init__()

    def log_step_message(self, step: int, node_name: str, output: BaseMessage):
        try:
            parsed_output = self.parser.invoke(output)
            self._log_message(f"## {step}, {node_name}")
            self._log_message(f'Answer: {parsed_output[0]["args"]["answer"]}')
            self._log_message(
                f'Reflection | Improving: {parsed_output[0]["args"]["reflection"]}'
            )
            if REVISED_ANSWER in parsed_output[0]["args"]:
                self._log_message("Reflection | Revised Answer:")
                self._log_message(parsed_output[0]["args"][REVISED_ANSWER])
            self._log_message(
                "-------------------------------- Node Output --------------------------------"
            )
        except Exception as e:
            self._log_message(str(output)[:100] + " ...", "error")

    def log_final_result(self, final_result: ReflexionAgentResult) -> None:
        self._log_message(
            "\n\n-------------------------------- Final Generated Response --------------------------------"
        )
        obj = vars(final_result)
        self._log_message(json.dumps(obj))


class GenerateAnswer(BaseModel):
    """Generate the answer."""

    answer: str = Field(
        description="Initial LLM answer to the user's question."
    )
    reflection: str = Field(
        description="Your reflection on the initial answer, critique of what to improve"
    )


class ReviseAnswer(GenerateAnswer):
    """Revise your previous answer according to your question."""

    revised_answer: str = Field(description=REVISED_ANSWER_DESCRIPTION)
    score: str = Field(description=SCORE_DESCRIPTION)


class CorrectionReflexionAgent(ReflexionAgent):
    def __init__(
        self,
        conversation_factory: Callable,
        max_steps: Optional[int] = 20,
    ):
        """
        LLM agent reflexion framework:

        start -> draft -> revise -> evaluate -> end
                            /|\        |
                             -----------

        Adapts base class to build and refine an LLM response to a user's
        question.

        Args:
            conversation_factory: function to return the Conversation to use for
                the LLM connection

            max_steps: the maximum number of steps to execute in the graph

        """
        super().__init__(
            conversation_factory,
            max_steps,
            agent_logger=CorrectionReflexionAgentLogger(),
        )
        self.actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "As a senior biomedical researcher, your task is to "
                        "generate a factually correct answer to the user's question. "
                        """Current time {time}. {instruction}"""
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(time=lambda: datetime.now().isoformat())
        self.parser = JsonOutputToolsParser(return_id=True)

    def _create_initial_responder(
        self, prompt: Optional[str] = None
    ) -> ResponderWithRetries:
        llm: ChatOpenAI = self.conversation.chat
        initial_chain = (
            self.actor_prompt_template.partial(
                instruction=prompt if prompt is not None else ""
            )
            | llm
        )
        validator = PydanticToolsParser(tools=[GenerateAnswer])
        return ResponderWithRetries(runnable=initial_chain, validator=validator)

    def _create_revise_responder(
        self, prompt: str | None = None
    ) -> ResponderWithRetries:
        revision_instruction = """
        Revise your previous answer using the guidelines:
        1. The primary aim is to ensure factual correctness.
        2. You should use previous critique to improve your answer.
        3. Only generate a new answer without returning any other text.
        """
        llm: ChatOpenAI = self.conversation.chat
        revision_chain = (
            self.actor_prompt_template.partial(instruction=revision_instruction)
            | llm
        )
        validator = PydanticToolsParser(tools=[ReviseAnswer])
        return ResponderWithRetries(
            runnable=revision_chain, validator=validator
        )

    def _tool_function(self, state: list[BaseMessage]):
        tool_message: AIMessage = state[-1]
        return ToolMessage(
            content=tool_message.content,
            tool_call_id=tool_message.tool_call_id,
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
                        if isinstance(res[k], str) and (
                            res[k] == "None" or res[k] == "null"
                        ):
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
        if not score is None and score >= 7:
            return END
        query_results_num = CorrectionReflexionAgent._get_last_tool_results_num(
            state
        )
        return END if query_results_num > 0 else REVISE_NODE

    def _parse_final_result(
        self, messages: list[BaseMessage]
    ) -> ReflexionAgentResult:
        output = messages[-1]
        result = self.parser.invoke(output)[0]["args"]
        tool_result = ReflexionAgent._get_last_tool_result(messages)
        return ReflexionAgentResult(
            answer=result["answer"] if "answer" in result else None,
            tool_result=tool_result,
        )

    def _build_graph(self, prompt: Optional[str] = None):
        """
        Build Langgraph graph for execution of chained LLM processes. Adjusted
        for missing tool step in the graph.

        Args:
          prompt str: prompt for LLM

        Returns:
          CompiledGraph | None: a Langgraph graph or None in case of errors
        """
        try:
            self.initial_responder = self._create_initial_responder(prompt)
            self.revise_responder = self._create_revise_responder(prompt)
            builder = MessageGraph()
            builder.add_node(DRAFT_NODE, self.initial_responder.respond)
            builder.add_node(REVISE_NODE, self.revise_responder.respond)
            builder.add_edge(DRAFT_NODE, REVISE_NODE)

            builder.add_conditional_edges(REVISE_NODE, self._should_continue)
            builder.set_entry_point(DRAFT_NODE)
            graph = builder.compile()
            return graph
        except Exception as e:
            logger.error(e)
            return None
