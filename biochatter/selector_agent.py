import json
from collections.abc import Callable
from datetime import datetime

from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from .langgraph_agent_base import (
    ReflexionAgent,
    ReflexionAgentLogger,
    ReflexionAgentResult,
    ResponderWithRetries,
)
from .rag_agent import RagAgent


class RagAgentSelectLogger(ReflexionAgentLogger):
    """RagAgentSelector logger"""

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
            if "revised_answer" in parsed_output[0]["args"]:
                self._log_message("Reflection | Revised Answer:")
                self._log_message(parsed_output[0]["args"]["revised_answer"])
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


class RagAgentChoiceModel(BaseModel):
    """Choose RagAgent"""

    answer: str = Field(
        description="RagAgent name that is the most appropriate to provide supplementary information",
    )
    reflection: str = Field(
        description="Your reflection on the initial answer, critique of what to improve",
    )


class RagAgentRevisionModel(RagAgentChoiceModel):
    """Revise your previous answer"""

    revised_answer: str = Field(description="Revised RagAgent name")
    score: str = Field(
        description=(
            "the score for the chosen rag agent based on its inquired result"
            "and the relevance to user's question, "
            " with 0 representing the lowest score and 10 representing the highest score."
        ),
    )


class RagAgentSelector(ReflexionAgent):
    def __init__(
        self,
        rag_agents: list[RagAgent],
        conversation_factory: Callable,
    ):
        """The class RagAgentSelector uses an LLM to choose the appropriate rag agent
        for a given user question.
        """
        super().__init__(
            conversation_factory=conversation_factory,
            agent_logger=RagAgentSelectLogger(),
        )
        self.rag_agents = rag_agents
        agent_desc = [
            f"{ix}. {rag_agents[ix].mode}: {rag_agents[ix].get_description()}" for ix in range(len(rag_agents))
        ]
        rag_agents_desc = "\n\n".join(agent_desc)
        self.actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You have access to the following rag agents:\n"
                        "{rag_agents_desc}\n\n"
                        " Your task is to choose the most appropriate rag "
                        "agent from them based on user's question to generate "
                        "information. "
                        """Current time {time}. {instruction}"""
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Only return rag agent name according to user's question above.",
                ),
            ],
        ).partial(
            time=lambda: datetime.now().isoformat(),
            rag_agents_desc=rag_agents_desc,
        )
        self.parser = JsonOutputToolsParser(return_id=True)

    def _create_initial_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        llm: ChatOpenAI = self.conversation.chat
        initial_chain = self.actor_prompt_template.partial(
            instruction="",
        ) | llm.bind_tools(
            tools=[RagAgentChoiceModel],
            tool_choice="RagAgentChoiceModel",
        )
        validator = PydanticToolsParser(tools=[RagAgentChoiceModel])
        return ResponderWithRetries(runnable=initial_chain, validator=validator)

    def _create_revise_responder(
        self,
        prompt: str | None = None,
    ) -> ResponderWithRetries:
        revision_instruction = """
Revise your previous chosen rag agent based on the result of the rag agent and follow the guidelines:
1. You should use previous critique to improve your chosen rag agent
2. Only generate rag agent name without returning any other text.
"""
        llm = self.conversation.chat
        revision_chain = self.actor_prompt_template.partial(
            instruction=revision_instruction,
        ) | llm.bind_tools(
            tools=[RagAgentRevisionModel],
            tool_choice="RagAgentRevisionModel",
        )
        validator = PydanticToolsParser(tools=[RagAgentRevisionModel])
        return ResponderWithRetries(
            runnable=revision_chain,
            validator=validator,
        )

    def _tool_function(self, state: list[BaseMessage]) -> ToolMessage:
        user_question = ReflexionAgent._get_user_question(state)
        assert user_question is not None
        tool_message: AIMessage = state[-1]
        parsed_tool_message = self.parser.invoke(tool_message)
        content = None
        for parsed_msg in parsed_tool_message:
            parsed_args = parsed_msg["args"]
            agent_name = (
                parsed_args["revised_answer"]
                if "revised_answer" in parsed_args and len(parsed_args["revised_answer"]) > 0
                else parsed_args["answer"]
            )
            # get agent according to agent name
            found_agent = None
            for agent in self.rag_agents:
                if agent.mode == agent_name:
                    found_agent = agent
                    break
            if found_agent == None:
                content = json.dumps({"rag_agent": agent_name, "result": []})
                continue
            result = found_agent.generate_responses(user_question)
            if len(result) > 0:
                content = json.dumps(
                    {"rag_agent": found_agent.mode, "result": result},
                )
                break
        content = (
            content
            if content is not None
            else json.dumps(
                {
                    "rag_agent": "",
                    "result": [],
                },
            )
        )
        return ToolMessage(content=content, tool_call_id=parsed_msg["id"])

    def _should_continue(self, state: list[BaseMessage]):
        return END  # here we use one-pass loop for sake of performance

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
