import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biochatter._stats import get_stats
from biochatter.llm_connect.conversation import Conversation


class AnthropicConversation(Conversation):
    """Conversation class for the Anthropic model."""

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
    ) -> None:
        """Connect to Anthropic's API and set up a conversation with the user.

        Also initialise a second conversational agent to provide corrections to
        the model output, if necessary.

        Args:
        ----
            model_name (str): The name of the model to use.

            prompts (dict): A dictionary of prompts to use for the conversation.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
        )

        self.ca_model_name = "claude-3-5-sonnet-20240620"
        # TODO make accessible by drop-down

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the Anthropic API.

        If the key is valid, initialise the conversational agent. Optionally set
        the user for usage statistics.

        Args:
        ----
            api_key (str): The API key for the Anthropic API.

            user (str, optional): The user for usage statistics. If provided and
                equals "community", will track usage stats.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.user = user

        try:
            client.count_tokens("Test connection")
            self.chat = ChatAnthropic(
                model_name=self.model_name,
                temperature=0,
                api_key=api_key,
            )
            self.ca_chat = ChatAnthropic(
                model_name=self.ca_model_name,
                temperature=0,
                api_key=api_key,
            )
            if user == "community":
                self.usage_stats = get_stats(user=user)

            return True

        except anthropic._exceptions.AuthenticationError:
            self._chat = None
            self._ca_chat = None
            return False

    def _primary_query(self, **kwargs) -> tuple:
        """Query the Anthropic API with the user's message.

        Return the response using the message history (flattery system messages,
        prior conversation) as context. Correct the response if necessary.

        Args:
        ----
            **kwargs: Keyword arguments (not used by this basic Anthropic implementation,
                     but accepted for compatibility with the base Conversation interface)

        Returns:
        -------
            tuple: A tuple containing the response from the Anthropic API and
                the token usage.

        """
        try:
            history = self._create_history()
            response = self.chat.generate([history])
        except (
            anthropic._exceptions.APIError,
            anthropic._exceptions.AnthropicError,
            anthropic._exceptions.ConflictError,
            anthropic._exceptions.NotFoundError,
            anthropic._exceptions.APIStatusError,
            anthropic._exceptions.RateLimitError,
            anthropic._exceptions.APITimeoutError,
            anthropic._exceptions.BadRequestError,
            anthropic._exceptions.APIConnectionError,
            anthropic._exceptions.AuthenticationError,
            anthropic._exceptions.InternalServerError,
            anthropic._exceptions.PermissionDeniedError,
            anthropic._exceptions.UnprocessableEntityError,
            anthropic._exceptions.APIResponseValidationError,
        ) as e:
            return str(e), None

        msg = response.generations[0][0].text
        token_usage = response.llm_output.get("token_usage")

        self.append_ai_message(msg)

        return msg, token_usage

    def _create_history(self) -> list:
        """Create a history of messages for the Anthropic API.

        Returns
        -------
            list: A list of messages, with the last message being the most
                recent.

        """
        history = []
        # extract text components from message contents
        msg_texts = [m.content[0]["text"] if isinstance(m.content, list) else m.content for m in self.messages]

        # check if last message is an image message
        is_image_message = False
        if isinstance(self.messages[-1].content, list):
            is_image_message = self.messages[-1].content[1]["type"] == "image_url"

        # find location of last AI message (if any)
        last_ai_message = None
        for i, m in enumerate(self.messages):
            if isinstance(m, AIMessage):
                last_ai_message = i

        # Aggregate system messages into one message at the beginning
        system_messages = [m.content for m in self.messages if isinstance(m, SystemMessage)]
        if system_messages:
            history.append(
                SystemMessage(content="\n".join(system_messages)),
            )

        # concatenate all messages before the last AI message into one message
        if last_ai_message is not None:
            history.append(
                HumanMessage(
                    content="\n".join([m for m in msg_texts[:last_ai_message]]),
                ),
            )
            # then append the last AI message
            history.append(
                AIMessage(
                    content=msg_texts[last_ai_message],
                ),
            )

            # then concatenate all messages after that
            # into one HumanMessage
            history.append(
                HumanMessage(
                    content="\n".join(
                        [m for m in msg_texts[last_ai_message + 1 :]],
                    ),
                ),
            )

        # else add human message to history (without system messages)
        else:
            last_system_message = None
            for i, m in enumerate(self.messages):
                if isinstance(m, SystemMessage):
                    last_system_message = i
            history.append(
                HumanMessage(
                    content="\n".join(
                        [m for m in msg_texts[last_system_message + 1 :]],
                    ),
                ),
            )

        # if the last message is an image message, add the image to the history
        if is_image_message:
            history[-1].content = [
                {"type": "text", "text": history[-1].content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.messages[-1].content[1]["image_url"]["url"],
                    },
                },
            ]
        return history

    def _correct_response(self, msg: str) -> str:
        """Correct the response from the Anthropic API.

        Send the response to a secondary language model. Optionally split the
        response into single sentences and correct each sentence individually.
        Update usage stats.

        Args:
        ----
            msg (str): The response from the Anthropic API.

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

        return correction
