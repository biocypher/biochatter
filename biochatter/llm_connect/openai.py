from collections.abc import Callable

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from biochatter._stats import get_stats
from biochatter.llm_connect.conversation import Conversation


class GptConversation(Conversation):
    """Conversation class for the OpenAI GPT model."""

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        base_url: str = None,
        update_token_usage: Callable | None = None,
    ) -> None:
        """Connect to OpenAI's GPT API and set up a conversation with the user.

        Also initialise a second conversational agent to provide corrections to
        the model output, if necessary.

        Args:
        ----
            model_name (str): The name of the model to use.

            prompts (dict): A dictionary of prompts to use for the conversation.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

            base_url (str): Optional OpenAI base_url value to use custom
                endpoint URL instead of default

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
        )
        self.base_url = base_url
        self.ca_model_name = "gpt-3.5-turbo"
        # TODO make accessible by drop-down

        self._update_token_usage = update_token_usage

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the OpenAI API.

        If the key is valid, initialise the conversational agent. Optionally set
        the user for usage statistics.

        Args:
        ----
            api_key (str): The API key for the OpenAI API.

            user (str, optional): The user for usage statistics. If provided and
                equals "community", will track usage stats.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        self.user = user

        try:
            client.models.list()
            self.chat = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=api_key,
                base_url=self.base_url,
            )
            self.ca_chat = ChatOpenAI(
                model_name=self.ca_model_name,
                temperature=0,
                openai_api_key=api_key,
                base_url=self.base_url,
            )
            if user == "community":
                self.usage_stats = get_stats(user=user)

            return True

        except openai._exceptions.AuthenticationError:
            self._chat = None
            self._ca_chat = None
            return False

    def _primary_query(self, **kwargs) -> tuple:
        """Query the OpenAI API with the user's message.

        Return the response using the message history (flattery system messages,
        prior conversation) as context. Correct the response if necessary.

        Args:
        ----
            **kwargs: Keyword arguments (not used by this basic GPT implementation,
                     but accepted for compatibility with the base Conversation interface)

        Returns:
        -------
            tuple: A tuple containing the response from the OpenAI API and the
                token usage.

        """
        try:
            response = self.chat.generate([self.messages])
        except (
            openai._exceptions.APIError,
            openai._exceptions.OpenAIError,
            openai._exceptions.ConflictError,
            openai._exceptions.NotFoundError,
            openai._exceptions.APIStatusError,
            openai._exceptions.RateLimitError,
            openai._exceptions.APITimeoutError,
            openai._exceptions.BadRequestError,
            openai._exceptions.APIConnectionError,
            openai._exceptions.AuthenticationError,
            openai._exceptions.InternalServerError,
            openai._exceptions.PermissionDeniedError,
            openai._exceptions.UnprocessableEntityError,
            openai._exceptions.APIResponseValidationError,
        ) as e:
            return str(e), None

        msg = response.generations[0][0].text
        token_usage = response.llm_output.get("token_usage")

        self._update_usage_stats(self.model_name, token_usage)

        self.append_ai_message(msg)

        return msg, token_usage

    def _correct_response(self, msg: str) -> str:
        """Correct the response from the OpenAI API.

        Send the response to a secondary language model. Optionally split the
        response into single sentences and correct each sentence individually.
        Update usage stats.

        Args:
        ----
            msg (str): The response from the OpenAI API.

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

        response = self.ca_chat.generate([ca_messages])

        correction = response.generations[0][0].text
        token_usage = response.llm_output.get("token_usage")

        self._update_usage_stats(self.ca_model_name, token_usage)

        return correction

    def _update_usage_stats(self, model: str, token_usage: dict) -> None:
        """Update redis database with token usage statistics.

        Use the usage_stats object with the increment method.

        Args:
        ----
            model (str): The model name.

            token_usage (dict): The token usage statistics.

        """
        if self.user == "community":
            # Only process integer values
            stats_dict = {f"{k}:{model}": v for k, v in token_usage.items() if isinstance(v, int | float)}
            self.usage_stats.increment(
                "usage:[date]:[user]",
                stats_dict,
            )

        if self._update_token_usage is not None:
            self._update_token_usage(self.user, model, token_usage)
