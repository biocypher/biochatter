"""Manage connections to LLM providers and handle conversations.

This module provides the general conversation class, which is used to manage
connections to different LLM APIs (OpenAI, Anthropic, Ollama, etc.) and
handling conversations with them, including message history, context injection,
and response correction capabilities.
"""

import json
import logging
import urllib.parse
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from typing import Literal

import nltk
from pydantic import BaseModel

try:
    import streamlit as st
except ImportError:
    st = None


import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from biochatter._image import encode_image, encode_image_from_url
from biochatter.llm_connect.available_models import TOOL_CALLING_MODELS
from biochatter.rag_agent import RagAgent
from biochatter.selector_agent import RagAgentSelector

logger = logging.getLogger(__name__)

TOOL_USAGE_PROMPT = """
<general_instructions>
You are a helpful assistant that can use tools to help the user.
You will need to return only a json formatted response that can than be used to parametrize the call to a function to answer the user's question.
The json should only contain have as key the name of the arguments and as value the value of the argument plus a key "tool_name" with the name of the tool.
</general_instructions>

<user_question>
{user_question}
</user_question>

<tools>
{tools}
</tools>

<additional_tools_instructions>
{additional_tools_instructions}
</additional_tools_instructions>
"""

GENERAL_TOOL_RESULT_INTERPRETATION_PROMPT = """
You are a helpful assistant that interprets the results of tool calls to answer user questions.
Your task is to analyze the tool's output and provide a clear, detailed explanation that directly addresses the user's original question.
Focus on extracting the most relevant information from the tool result and presenting it in a user-friendly way."""

ADDITIONAL_TOOL_RESULT_INTERPRETATION_PROMPT = """
Based on the tool result above, provide a helpful response to the original question.
Make sure your explanation is accurate and based solely on the information provided by the tool.
If the tool result doesn't fully answer the question, acknowledge the limitations and explain what information is available.
"""

TOOL_RESULT_INTERPRETATION_PROMPT = """
<general_instructions>
{general_instructions}
</general_instructions>

<original_question>
{original_question}
</original_question>

<tool_result>
{tool_result}
</tool_result>

<additional_instructions>
{additional_instructions}
</additional_instructions>
"""


class Conversation(ABC):
    """Use this class to set up a connection to an LLM API.

    Can be used to set the user name and API key, append specific messages for
    system, user, and AI roles (if available), set up the general context as
    well as manual and tool-based data inputs, and finally to query the API
    with prompts made by the user.

    The conversation class is expected to have a `messages` attribute to store
    the conversation, and a `history` attribute, which is a list of messages in
    a specific format for logging / printing.

    """

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        use_ragagent_selector: bool = False,
        tools: list[Callable] = None,
        tool_call_mode: Literal["auto", "text"] = "auto",
        mcp: bool = False,
        additional_tools_instructions: str = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.prompts = prompts
        self.correct = correct
        self.split_correction = split_correction
        self.rag_agents: list[RagAgent] = []
        self.history = []
        self.messages = []
        self.ca_messages = []
        self.tool_calls = deque()
        self.current_statements = []
        self._use_ragagent_selector = use_ragagent_selector
        self._chat = None
        self._ca_chat = None
        self.tools = tools
        self.tool_call_mode = tool_call_mode
        self.tools_prompt = None
        self.mcp = mcp
        self.additional_tools_instructions = additional_tools_instructions if additional_tools_instructions else ""

    @property
    def chat(self):
        """Access the chat attribute with error handling."""
        if self._chat is None:
            msg = "Chat attribute not initialized. Did you call set_api_key()?"
            logger.error(msg)
            raise AttributeError(msg)
        return self._chat

    @chat.setter
    def chat(self, value):
        """Set the chat attribute."""
        self._chat = value

    @property
    def ca_chat(self):
        """Access the correcting agent chat attribute with error handling."""
        if self._ca_chat is None:
            msg = "Correcting agent chat attribute not initialized. Did you call set_api_key()?"
            logger.error(msg)
            raise AttributeError(msg)
        return self._ca_chat

    @ca_chat.setter
    def ca_chat(self, value):
        """Set the correcting agent chat attribute."""
        self._ca_chat = value

    @property
    def use_ragagent_selector(self) -> bool:
        """Whether to use the ragagent selector."""
        return self._use_ragagent_selector

    @use_ragagent_selector.setter
    def use_ragagent_selector(self, val: bool) -> None:
        """Set the use_ragagent_selector attribute."""
        self._use_ragagent_selector = val

    def set_user_name(self, user_name: str) -> None:
        """Set the user name."""
        self.user_name = user_name

    def set_rag_agent(self, agent: RagAgent) -> None:
        """Update or insert rag_agent.

        If the rag_agent with the same mode already exists, it will be updated.
        Otherwise, the new rag_agent will be inserted.
        """
        i, _ = self.find_rag_agent(agent.mode)
        if i < 0:
            # insert
            self.rag_agents.append(agent)
        else:
            # update
            self.rag_agents[i] = agent

    def find_rag_agent(self, mode: str) -> tuple[int, RagAgent]:
        """Find the rag_agent with the given mode."""
        for i, val in enumerate(self.rag_agents):
            if val.mode == mode:
                return i, val
        return -1, None

    @abstractmethod
    def set_api_key(self, api_key: str, user: str | None = None) -> None:
        """Set the API key."""

    def get_prompts(self) -> dict:
        """Get the prompts."""
        return self.prompts

    def set_prompts(self, prompts: dict) -> None:
        """Set the prompts."""
        self.prompts = prompts

    def _tool_formatter(self, tools: list[Callable], mcp: bool = False) -> str:
        """Format the tools. Only for model not supporting tool calling."""
        tools_description = ""

        for idx, tool in enumerate(tools):
            tools_description += f"<tool_{idx}>\n"
            tools_description += f"Tool name: {tool.name}\n"
            tools_description += f"Tool description: {tool.description}\n"
            if mcp:
                tools_description += f"Tool call schema:\n {tool.tool_call_schema}\n"
            else:
                tools_description += f"Tool call schema:\n {tool.args}\n"
            tools_description += f"</tool_{idx}>\n"
        return tools_description

    def _create_tool_prompt(
        self, tools: list[Callable], additional_tools_instructions: str = None, mcp: bool = False
    ) -> str:
        """Create the tool prompt. Only for model not supporting tool calling."""
        prompt_template = ChatPromptTemplate.from_template(TOOL_USAGE_PROMPT)
        tools_description = self._tool_formatter(tools, mcp=mcp)
        new_message = prompt_template.invoke(
            {
                "user_question": self.messages[-1].content,
                "tools": tools_description,
                "additional_tools_instructions": additional_tools_instructions if additional_tools_instructions else "",
            }
        )
        return new_message.messages[0]

    def bind_tools(self, tools: list[Callable]) -> None:
        """Bind tools to the chat."""
        # Check if the model supports tool calling
        # (exploit the enum class in available_models.py)
        if self.model_name in TOOL_CALLING_MODELS and self.ca_chat:
            self.chat = self.chat.bind_tools(tools)
            self.ca_chat = self.ca_chat.bind_tools(tools)

        elif self.model_name in TOOL_CALLING_MODELS:
            self.chat = self.chat.bind_tools(tools)

        # elif self.model_name not in TOOL_CALLING_MODELS:
        #    self.tools_prompt = self._create_tool_prompt(tools, additional_instructions)

        # If not, fail gracefully
        # raise ValueError(f"Model {self.model_name} does not support tool calling.")

    def append_ai_message(self, message: str) -> None:
        """Add a message from the AI to the conversation.

        Args:
        ----
            message (str): The message from the AI.

        """
        self.messages.append(
            AIMessage(
                content=message,
            ),
        )

    def append_system_message(self, message: str) -> None:
        """Add a system message to the conversation.

        Args:
        ----
            message (str): The system message.

        """
        self.messages.append(
            SystemMessage(
                content=message,
            ),
        )

    def append_ca_message(self, message: str) -> None:
        """Add a message to the correcting agent conversation.

        Args:
        ----
            message (str): The message to the correcting agent.

        """
        self.ca_messages.append(
            SystemMessage(
                content=message,
            ),
        )

    def append_user_message(self, message: str) -> None:
        """Add a message from the user to the conversation.

        Args:
        ----
            message (str): The message from the user.

        """
        self.messages.append(
            HumanMessage(
                content=message,
            ),
        )

    def append_image_message(
        self,
        message: str,
        image_url: str,
        local: bool = False,
    ) -> None:
        """Add a user message with an image to the conversation.

        Also checks, in addition to the `local` flag, if the image URL is a
        local file path. If it is local, the image will be encoded as a base64
        string to be passed to the LLM.

        Args:
        ----
            message (str): The message from the user.
            image_url (str): The URL of the image.
            local (bool): Whether the image is local or not. If local, it will
                be encoded as a base64 string to be passed to the LLM.

        """
        parsed_url = urllib.parse.urlparse(image_url)
        if local or not parsed_url.netloc:
            image_url = f"data:image/jpeg;base64,{encode_image(image_url)}"
        else:
            image_url = f"data:image/jpeg;base64,{encode_image_from_url(image_url)}"

        self.messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            ),
        )

    def setup(self, context: str) -> None:
        """Set up the conversation with general prompts and a context."""
        for msg in self.prompts["primary_model_prompts"]:
            if msg:
                self.append_system_message(msg)

        for msg in self.prompts["correcting_agent_prompts"]:
            if msg:
                self.append_ca_message(msg)

        self.context = context
        msg = f"The topic of the research is {context}."
        self.append_system_message(msg)

    def setup_data_input_manual(self, data_input: str) -> None:
        """Set up the data input manually."""
        self.data_input = data_input
        msg = f"The user has given information on the data input: {data_input}."
        self.append_system_message(msg)

    def setup_data_input_tool(self, df, input_file_name: str) -> None:
        """Set up the data input tool."""
        self.data_input_tool = df

        for tool_name in self.prompts["tool_prompts"]:
            if tool_name in input_file_name:
                msg = self.prompts["tool_prompts"][tool_name].format(df=df)
                self.append_system_message(msg)

    def query(
        self,
        text: str,
        image_url: str | None = None,
        structured_model: BaseModel | None = None,
        wrap_structured_output: bool | None = None,
        tools: list[Callable] | None = None,
        explain_tool_result: bool | None = None,
        additional_tools_instructions: str | None = None,
        general_instructions_tool_interpretation: str | None = None,
        additional_instructions_tool_interpretation: str | None = None,
        mcp: bool | None = None,
        return_tool_calls_as_ai_message: bool | None = None,
        track_tool_calls: bool | None = None,
        **kwargs,
    ) -> tuple[str, dict | None, str | None]:
        """Query the LLM API using the user's query.

        Appends the most recent query to the conversation, optionally injects
        context from the RAG agent, and runs the primary query method of the
        child class.

        Args:
        ----
            text (str): The user query.

            image_url (str): The URL of an image to include in the conversation.
                Optional and only supported for models with vision capabilities.

            structured_model (BaseModel): The structured output model to use for the query.

            wrap_structured_output (bool): Whether to wrap the structured output in JSON quotes.

            tools (list[Callable]): The tools to use for the query.

            explain_tool_result (bool): Whether to explain the tool result.

            additional_tools_instructions (str): The additional instructions for the query.
                Mainly used for tools that do not support tool calling.

            general_instructions_tool_interpretation (str): The general
                instructions for the tool interpretation.
                Overrides the default prompt in `GENERAL_TOOL_RESULT_INTERPRETATION_PROMPT`.

            additional_instructions_tool_interpretation (str): The additional
                instructions for the tool interpretation.
                Overrides the default prompt in `ADDITIONAL_TOOL_RESULT_INTERPRETATION_PROMPT`.

            mcp (bool): If you want to use MCP mode, this should be set to True.

            return_tool_calls_as_ai_message (bool): If you want to return the tool calls as an AI message, this should be set to True.

            track_tool_calls (bool): If you want to track the tool calls, this should be set to True.

            **kwargs: Additional keyword arguments.

        Returns:
        -------
            tuple: A tuple containing the response from the API, the token usage
                information, and the correction if necessary/desired.

        """
        if mcp:
            self.mcp = True

        # save the last human prompt that may be used for answer enhancement
        self.last_human_prompt = text

        # if additional_tools_instructions are provided, save them
        if additional_tools_instructions:
            self.additional_tools_instructions = additional_tools_instructions

        # override the default prompts if other provided
        self.general_instructions_tool_interpretation = (
            general_instructions_tool_interpretation
            if general_instructions_tool_interpretation
            else GENERAL_TOOL_RESULT_INTERPRETATION_PROMPT
        )
        self.additional_instructions_tool_interpretation = (
            additional_instructions_tool_interpretation
            if additional_instructions_tool_interpretation
            else ADDITIONAL_TOOL_RESULT_INTERPRETATION_PROMPT
        )
        if not image_url:
            self.append_user_message(text)
        else:
            self.append_image_message(text, image_url)

        self._inject_context(text)

        # tools passed at this step are used only for this message
        msg, token_usage = self._primary_query(
            tools=tools,
            explain_tool_result=explain_tool_result,
            return_tool_calls_as_ai_message=return_tool_calls_as_ai_message,
            structured_model=structured_model,
            wrap_structured_output=wrap_structured_output,
            track_tool_calls=track_tool_calls,
        )

        if not token_usage:
            # indicates error
            return (msg, token_usage, None)

        if not self.correct:
            return (msg, token_usage, None)

        cor_msg = "Correcting (using single sentences) ..." if self.split_correction else "Correcting ..."

        if st:
            with st.spinner(cor_msg):
                corrections = self._correct_query(text)
        else:
            corrections = self._correct_query(text)

        if not corrections:
            return (msg, token_usage, None)

        correction = "\n".join(corrections)
        return (msg, token_usage, correction)

    def _correct_query(self, msg: str) -> list[str]:
        corrections = []
        if self.split_correction:
            nltk.download("punkt")
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            sentences = tokenizer.tokenize(msg)
            for sentence in sentences:
                correction = self._correct_response(sentence)

                if str(correction).lower() not in ["ok", "ok."]:
                    corrections.append(correction)
        else:
            correction = self._correct_response(msg)

            if str(correction).lower() not in ["ok", "ok."]:
                corrections.append(correction)

        return corrections

    @abstractmethod
    def _primary_query(self, **kwargs) -> tuple[str, dict | None]:
        """Run the primary query.

        Args:
        ----
            **kwargs: Keyword arguments that may include:
                - text: The user query.
                - tools: List of tools for tool-calling models
                - explain_tool_result: Whether to explain tool results
                - return_tool_calls_as_ai_message: Whether to return tool calls as AI message
                - structured_model: Structured output model
                - wrap_structured_output: Whether to wrap structured output
                - track_tool_calls: Whether to track tool calls
                - Other model-specific parameters

        Returns:
        -------
            tuple: A tuple containing the response message and token usage information.

        """

    @abstractmethod
    def _correct_response(self, msg: str) -> str:
        """Correct the response."""

    def _process_manual_tool_call(
        self,
        tool_call: list[dict],
        available_tools: list[Callable],
        explain_tool_result: bool = False,
    ) -> str:
        """Process manual tool calls from the model response.

        This method handles the processing of tool calls for models that don't natively
        support tool calling. It takes the parsed JSON response and executes the
        appropriate tool.

        Args:
        ----
            tool_call (list[dict]): The parsed tool call information from the model response.
            available_tools (list[Callable]): The tools available for execution.
            explain_tool_result (bool): Whether to explain the tool result.

        Returns:
        -------
            str: The processed message containing the tool name, arguments, and result.

        """
        tool_name = tool_call["tool_name"]
        tool_func = next((t for t in available_tools if t.name == tool_name), None)

        # Remove the tool name from the tool call in order to invoke the tool
        # This is beacause tool_name is not a valid argument for the tool
        del tool_call["tool_name"]

        # Execute the tool based on whether we're in async context or not
        if self.mcp:
            loop = asyncio.get_running_loop()
            tool_result = loop.run_until_complete(tool_func.ainvoke(tool_call))
        else:
            tool_result = tool_func.invoke(tool_call)

        msg = f"Tool: {tool_name}\nArguments: {tool_call}\nTool result: {tool_result}"

        if explain_tool_result:
            tool_result_interpretation = self.chat.invoke(
                TOOL_RESULT_INTERPRETATION_PROMPT.format(
                    original_question=self.last_human_prompt,
                    tool_result=tool_result,
                    general_instructions=self.general_instructions_tool_interpretation,
                    additional_instructions=self.additional_instructions_tool_interpretation,
                )
            )
            msg += f"\nTool result interpretation: {tool_result_interpretation.content}"

        self.append_ai_message(msg)

        return msg

    def _process_tool_calls(
        self,
        tool_calls: list[dict],
        available_tools: list[Callable],
        response_content: str,
        explain_tool_result: bool = False,
        return_tool_calls_as_ai_message: bool = False,
        track_tool_calls: bool = False,
    ) -> str:
        """Process tool calls from the model response.

        This method handles the processing of tool calls returned by the model.
        It can either automatically execute the tools and return their results,
        or format the tool calls as text.

        Args:
        ----
            tool_calls: The tool calls from the model response.
            response_content: The text content of the response (used as fallback).
            available_tools: The tools available in the chat.
            explain_tool_result (bool): Whether to explain the tool result.
            return_tool_calls_as_ai_message (bool): If you want to return the tool calls as an AI message, this should be set to True.
            track_tool_calls (bool): If you want to track the tool calls, this should be set to True.

        Returns:
        -------
            str: The processed message, either tool results or formatted tool calls.

        """
        if not tool_calls:
            return response_content

        msg = ""

        if self.tool_call_mode == "auto":
            # Collect tool results for collective explanation when multiple tools are called
            tool_results_for_explanation = []

            for idx, tool_call in enumerate(tool_calls):
                # Extract tool name and arguments
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                # Find the matching tool function
                tool_func = next((t for t in available_tools if t.name == tool_name), None)

                if tool_func:
                    # Execute the tool
                    try:
                        if self.mcp:
                            loop = asyncio.get_running_loop()
                            tool_result = loop.run_until_complete(tool_func.ainvoke(tool_args))
                        else:
                            tool_result = tool_func.invoke(tool_args)
                        # Add the tool result to the conversation
                        if return_tool_calls_as_ai_message:
                            self.append_ai_message(f"Tool call ({tool_name}) \nResult: {tool_result!s}")
                        else:
                            self.messages.append(
                                ToolMessage(content=str(tool_result), name=tool_name, tool_call_id=tool_call_id)
                            )

                        if track_tool_calls:
                            self.tool_calls.append(
                                {"name": tool_name, "args": tool_args, "id": tool_call_id, "result": tool_result}
                            )

                        if idx > 0:
                            msg += "\n"
                        msg += f"Tool call ({tool_name}) result: {tool_result!s}"

                        # Collect tool results for explanation if needed
                        if explain_tool_result:
                            tool_results_for_explanation.append(
                                {"name": tool_name, "args": tool_args, "result": tool_result}
                            )

                    except Exception as e:
                        # Handle tool execution errors
                        error_message = f"Error executing tool {tool_name}: {e!s}"
                        self.messages.append(
                            ToolMessage(content=error_message, name=tool_name, tool_call_id=tool_call_id)
                        )

                        # Track failed tool calls
                        if track_tool_calls:
                            self.tool_calls.append(
                                {"name": tool_name, "args": tool_args, "id": tool_call_id, "error": str(e)}
                            )

                        if idx > 0:
                            msg += "\n"
                        msg += error_message
                # Handle missing/unknown tool
                elif track_tool_calls:
                    self.tool_calls.append(
                        {"name": tool_name, "args": tool_args, "id": tool_call_id, "error": "Tool not found"}
                    )

            # Handle tool result explanation
            if explain_tool_result and tool_results_for_explanation:
                if len(tool_results_for_explanation) > 1:
                    # Multiple tools: explain all results together
                    combined_tool_results = "\n\n".join(
                        [
                            f"Tool: {tr['name']}\nArguments: {tr['args']}\nResult: {tr['result']}"
                            for tr in tool_results_for_explanation
                        ]
                    )

                    tool_result_interpretation = self.chat.invoke(
                        TOOL_RESULT_INTERPRETATION_PROMPT.format(
                            original_question=self.last_human_prompt,
                            tool_result=combined_tool_results,
                            general_instructions=self.general_instructions_tool_interpretation,
                            additional_instructions=self.additional_instructions_tool_interpretation,
                        )
                    )
                    self.append_ai_message(tool_result_interpretation.content)
                    msg += f"\nTool results interpretation: {tool_result_interpretation.content}"
                else:
                    # Single tool: explain individual result (maintain current behavior)
                    tool_result_data = tool_results_for_explanation[0]
                    tool_result_interpretation = self.chat.invoke(
                        TOOL_RESULT_INTERPRETATION_PROMPT.format(
                            original_question=self.last_human_prompt,
                            tool_result=tool_result_data["result"],
                            general_instructions=self.general_instructions_tool_interpretation,
                            additional_instructions=self.additional_instructions_tool_interpretation,
                        )
                    )
                    self.append_ai_message(tool_result_interpretation.content)
                    msg += f"\nTool result interpretation: {tool_result_interpretation.content}"

            return msg

        if self.tool_call_mode == "text":
            # Join all tool calls in a text format
            tool_calls_text = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                tool_calls_text.append(f"Tool: {tool_name} - Arguments: {tool_args} - Tool call id: {tool_call_id}")

            # Join with line breaks and set as the message
            msg = "\n".join(tool_calls_text)

            # Append the formatted tool calls as an AI message
            self.append_ai_message(msg)
            return msg

        # Invalid tool call mode, log warning and return original content
        logger.warning(f"Invalid tool call mode: {self.tool_call_mode}. Using original response content.")
        return response_content

    def _inject_context_by_ragagent_selector(self, text: str) -> list[str]:
        """Inject the context generated by RagAgentSelector.

        The RagAgentSelector will choose the appropriate rag agent to generate
        context according to user's question.

        Args:
        ----
            text (str): The user query to be used for choosing rag agent

        """
        rag_agents: list[RagAgent] = [agent for agent in self.rag_agents if agent.use_prompt]
        decider_agent = RagAgentSelector(
            rag_agents=rag_agents,
            conversation_factory=lambda: self,
        )
        result = decider_agent.execute(text)
        if result.tool_result is not None and len(result.tool_result) > 0:
            return result.tool_result
        # find rag agent selected
        rag_agent = next(
            [agent for agent in rag_agents if agent.mode == result.answer],
            None,
        )
        if rag_agent is None:
            return None
        return rag_agent.generate_responses(text)

    def _inject_context(self, text: str) -> None:
        """Inject the context received from the RAG agent into the prompt.

        The RAG agent will find the most similar n text fragments and add them
        to the message history object for usage in the next prompt. Uses the
        document summarisation prompt set to inject the context. The ultimate
        prompt should include the placeholder for the statements, `{statements}`
        (used for formatting the string).

        Args:
        ----
            text (str): The user query to be used for similarity search.

        """
        sim_msg = "Performing similarity search to inject fragments ..."

        if st:
            with st.spinner(sim_msg):
                statements = []
                if self.use_ragagent_selector:
                    statements = self._inject_context_by_ragagent_selector(text)
                else:
                    for agent in self.rag_agents:
                        try:
                            docs = agent.generate_responses(text)
                            statements = statements + [doc[0] for doc in docs]
                        except ValueError as e:
                            logger.warning(e)

        else:
            statements = []
            if self.use_ragagent_selector:
                statements = self._inject_context_by_ragagent_selector(text)
            else:
                for agent in self.rag_agents:
                    try:
                        docs = agent.generate_responses(text)
                        statements = statements + [doc[0] for doc in docs]
                    except ValueError as e:
                        logger.warning(e)

        if statements and len(statements) > 0:
            prompts = self.prompts["rag_agent_prompts"]
            self.current_statements = statements
            for i, prompt in enumerate(prompts):
                # if last prompt, format the statements into the prompt
                if i == len(prompts) - 1:
                    self.append_system_message(
                        prompt.format(statements=statements),
                    )
                else:
                    self.append_system_message(prompt)

    def get_last_injected_context(self) -> list[dict]:
        """Get a formatted list of the last context.

        Get the last context injected into the conversation. Contains one
        dictionary for each RAG mode.

        Returns
        -------
            List[dict]: A list of dictionaries containing the mode and context
            for each RAG agent.

        """
        return [{"mode": agent.mode, "context": agent.last_response} for agent in self.rag_agents]

    def get_msg_json(self) -> str:
        """Return a JSON representation of the conversation.

        Returns a list of dicts of the messages in the conversation in JSON
        format. The keys of the dicts are the roles, the values are the
        messages.

        Returns
        -------
            str: A JSON representation of the messages in the conversation.

        """
        d = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "ai"
            else:
                error_msg = f"Unknown message type: {type(msg)}"
                raise TypeError(error_msg)

            d.append({role: msg.content})

        return json.dumps(d)

    def reset(self) -> None:
        """Reset the conversation to the initial state."""
        self.history = []
        self.messages = []
        self.ca_messages = []
        self.current_statements = []
        self.tool_calls.clear()
