import json
from collections.abc import Callable
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from biochatter.llm_connect.available_models import TOOL_CALLING_MODELS
from biochatter.llm_connect.conversation import Conversation


class LangChainConversation(Conversation):
    """Conversation class for a generic LangChain model."""

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        tools: list[Callable] = None,
        tool_call_mode: Literal["auto", "text"] = "auto",
        async_mode: bool = False,
        mcp: bool = False,
    ) -> None:
        """Initialise the LangChainConversation class.

        Connect to a generic LangChain model and set up a conversation with the user.
        Also initialise a second conversational agent to provide corrections to
        the model output, if necessary.

        Args:
        ----
            model_name (str): The name of the model to use.
            model_provider (str): The provider of the model to use.
            prompts (dict): A dictionary of prompts to use for the conversation.
            correct (bool): Whether to correct the model output.
            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.
            tools (list[Callable]): List of tool functions to use with the model.
            tool_call_mode (str): The mode to use for tool calls.
                "auto": Automatically call tools.
                "text": Only return text output of the tool call.
            async_mode (bool): Whether to run in async mode. Defaults to False.
            mcp (bool): If you want to use MCP mode, this should be set to True.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            tools=tools,
            tool_call_mode=tool_call_mode,
            mcp=mcp,
        )

        self.model_name = model_name
        self.model_provider = model_provider
        self.async_mode = async_mode

    # TODO: the name of this method is overloaded, since the api key is loaded
    # from the environment variables and not as an argument
    def set_api_key(self, api_key: str | None = None, user: str | None = None) -> bool:
        """Set the API key for the model provider.

        If the key is valid, initialise the conversational agent. Optionally set
        the user for usage statistics.

        Args:
        ----
            api_key (str): The API key for the model provider.

            user (str, optional): The user for usage statistics. If provided and
                equals "community", will track usage stats.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        self.user = user

        try:
            self.chat = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=0,
            )
            self.ca_chat = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=0,
            )

            # if binding happens here, tools will be available for all messages
            if self.tools:
                self.bind_tools(self.tools)

            return True

        except Exception:  # Google Genai doesn't expose specific exception types
            self._chat = None
            self._ca_chat = None
            return False

    def _primary_query(self, tools: list[Callable] | None = None, explain_tool_result: bool = False) -> tuple:
        """Run the primary query in sync mode."""
        starting_tools = self.tools if self.tools else []
        in_chat_tools = tools if tools else []
        available_tools = starting_tools + in_chat_tools

        if self.model_name in TOOL_CALLING_MODELS:
            chat = self.chat.bind_tools(available_tools)
        elif self.model_name not in TOOL_CALLING_MODELS and len(available_tools) > 0:
            self.tools_prompt = self._create_tool_prompt(available_tools)
            self.messages[-1] = self.tools_prompt
            chat = self.chat
        elif len(available_tools) == 0:
            chat = self.chat

        try:
            response = chat.invoke(self.messages)
        except Exception as e:
            return str(e), None

        # Process tool calls if present (model supports tool calling)
        if response.tool_calls:
            msg = self._process_tool_calls(
                tool_calls=response.tool_calls,
                available_tools=available_tools,
                response_content=response.content,
                explain_tool_result=explain_tool_result,
            )
        # case where the model does not support tool calling and we need manual processing
        elif self.model_name not in TOOL_CALLING_MODELS and self.tools_prompt:
            msg = response.content.replace('"""', "").replace("json", "").replace("`", "").replace("\n", "").strip()
            msg = json.loads(msg)
            msg = self._porcess_manual_tool_call(
                tool_call=msg,
                available_tools=available_tools,
                explain_tool_result=explain_tool_result,
            )
            # self.append_ai_message(msg)
        else:
            msg = response.content
            self.append_ai_message(msg)

        token_usage = response.usage_metadata["total_tokens"]

        return msg, token_usage

    def _correct_response(self, msg: str) -> str:
        """Correct the response from the Gemini API.

        Send the response to a secondary language model. Optionally split the
        response into single sentences and correct each sentence individually.
        Update usage stats.

        Args:
        ----
            msg (str): The response from the Gemini API.

        Returns:
        -------
            str: The corrected response (or OK if no correction necessary).

        """
        ca_messages = self.ca_messages.copy()
        ca_messages.append(
            HumanMessage(
                content=msg,
            ),
        )
        ca_messages.append(
            SystemMessage(
                content="If there is nothing to correct, please respond with just 'OK', and nothing else!",
            ),
        )

        response = self.ca_chat.invoke(ca_messages)

        correction = response.content
        token_usage = response.usage_metadata["total_tokens"]

        return correction, token_usage
