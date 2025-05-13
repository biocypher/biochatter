from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from biochatter.llm_connect.available_models import TOOL_CALLING_MODELS
from biochatter.llm_connect.conversation import Conversation


class GeminiConversation(Conversation):
    """Conversation class for the Google Gemini model."""

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        tools: list[Callable] = None,
        tool_call_mode: Literal["auto", "text"] = "auto",
    ) -> None:
        """Initialise the GeminiConversation class.

        Connect to Google's Gemini API and set up a conversation with the user.
        Also initialise a second conversational agent to provide corrections to
        the model output, if necessary.

        Args:
        ----
            model_name (str): The name of the model to use.

            prompts (dict): A dictionary of prompts to use for the conversation.

            correct (bool): Whether to correct the model output.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

            tools (list[Callable]): List of tool functions to use with the model.

            tool_call_mode (str): The mode to use for tool calls.
                "auto": Automatically call tools.
                "text": Only return text output of the tool call.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            tools=tools,
            tool_call_mode=tool_call_mode,
        )

        self.ca_model_name = "gemini-2.0-flash"

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the Google Gemini API.

        If the key is valid, initialise the conversational agent. Optionally set
        the user for usage statistics.

        Args:
        ----
            api_key (str): The API key for the Google Gemini API.

            user (str, optional): The user for usage statistics. If provided and
                equals "community", will track usage stats.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        self.user = user

        try:
            self.chat = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                google_api_key=api_key,
            )
            self.ca_chat = ChatGoogleGenerativeAI(
                model=self.ca_model_name,
                temperature=0,
                google_api_key=api_key,
            )

            # if binding happens here, tools will be available for all messages
            if self.tools:
                self.bind_tools(self.tools)

            return True

        except Exception:  # Google Genai doesn't expose specific exception types
            self._chat = None
            self._ca_chat = None
            return False

    def _primary_query(self, tools: list[Callable] | None = None, **kwargs) -> tuple:
        """Query the Google Gemini API with the user's message.

        Return the response using the message history (flattery system messages,
        prior conversation) as context. Correct the response if necessary.

        Args:
        ----
            tools (list[Callable]): The tools to use for the query. Tools
            passed at this step are used only for this message and not stored
            as part of the conversation object.

            **kwargs: Additional keyword arguments.

        Returns:
        -------
            tuple: A tuple containing the response from the Gemini API and
                the token usage.

        """
        # bind tools to the chat if provided in the query
        chat = self.chat.bind_tools(tools) if (tools and self.model_name in TOOL_CALLING_MODELS) else self.chat

        try:
            response = chat.invoke(self.messages)
        except Exception as e:
            return str(e), None

        # Process tool calls if present
        if response.tool_calls:
            msg = self._process_tool_calls(response.tool_calls, tools, response.content)
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
            SystemMessage(
                content="If there is nothing to correct, please respond with just 'OK', and nothing else!",
            ),
        )

        ca_messages.append(
            HumanMessage(
                content=msg,
            ),
        )

        response = self.ca_chat.invoke(ca_messages)

        correction = response.content
        token_usage = response.usage_metadata["total_tokens"]

        return correction
