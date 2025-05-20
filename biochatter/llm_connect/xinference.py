import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biochatter.llm_connect.conversation import Conversation


class XinferenceConversation(Conversation):
    """Conversation class for the Xinference deployment."""

    def __init__(
        self,
        base_url: str,
        prompts: dict,
        model_name: str = "auto",
        correct: bool = False,
        split_correction: bool = False,
    ) -> None:
        """Connect to an open-source LLM via the Xinference client.

        Connect to a running Xinference deployment and set up a conversation
        with the user. Also initialise a second conversational agent to
        provide corrections to the model output, if necessary.

        Args:
        ----
            base_url (str): The base URL of the Xinference instance (should not
            include the /v1 part).

            prompts (dict): A dictionary of prompts to use for the conversation.

            model_name (str): The name of the model to use. Will be mapped to
            the according uid from the list of available models. Can be set to
            "auto" to use the first available model.

            correct (bool): Whether to correct the model output.

            split_correction (bool): Whether to correct the model output by
            splitting the output into sentences and correcting each sentence
            individually.

        """
        # Shaohong: Please keep this xinference importing code here, so that,
        # we don't need to depend on xinference if we dont need it (xinference
        # is expensive to install)
        from xinference.client import Client

        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
        )
        self.client = Client(base_url=base_url)

        self.models = {}
        self.load_models()

        self.ca_model_name = model_name

        self.set_api_key()

        # TODO make accessible by drop-down

    def load_models(self) -> None:
        """Load the models from the Xinference client."""
        for id, model in self.client.list_models().items():
            model["id"] = id
            self.models[model["model_name"]] = model

    def append_system_message(self, message: str) -> None:
        """Override the system message addition.

        Xinference does not accept multiple system messages. We concatenate them
        if there are multiple.

        Args:
        ----
            message (str): The message to append.

        """
        # if there is not already a system message in self.messages
        if not any(isinstance(m, SystemMessage) for m in self.messages):
            self.messages.append(
                SystemMessage(
                    content=message,
                ),
            )
        else:
            # if there is a system message, append to the last one
            for i, msg in enumerate(self.messages):
                if isinstance(msg, SystemMessage):
                    self.messages[i].content += f"\n{message}"
                    break

    def append_ca_message(self, message: str) -> None:
        """Override the system message addition for the correcting agent.

        Xinference does not accept multiple system messages. We concatenate them
        if there are multiple.

        TODO this currently assumes that the correcting agent is the same model
        as the primary one.

        Args:
        ----
            message (str): The message to append.

        """
        # if there is not already a system message in self.messages
        if not any(isinstance(m, SystemMessage) for m in self.ca_messages):
            self.ca_messages.append(
                SystemMessage(
                    content=message,
                ),
            )
        else:
            # if there is a system message, append to the last one
            for i, msg in enumerate(self.ca_messages):
                if isinstance(msg, SystemMessage):
                    self.ca_messages[i].content += f"\n{message}"
                    break

    def _primary_query(self, **kwargs) -> tuple:
        """Query the Xinference client API.

        Use the user's message and return the response using the message history
        (flattery system messages, prior conversation) as context. Correct the
        response if necessary.

        LLaMA2 architecture does not accept separate system messages, so we
        concatenate the system message with the user message to form the prompt.
        'LLaMA enforces a strict rule that chats should alternate
        user/assistant/user/assistant, and the system message, if present,
        should be embedded into the first user message.' (from
        https://discuss.huggingface.co/t/issue-with-llama-2-chat-template-and-out-of-date-documentation/61645/3)

        Returns
        -------
            tuple: A tuple containing the response from the Xinference API
            (formatted similarly to responses from the OpenAI API) and the token
            usage.

        """
        try:
            history = self._create_history()
            # TODO this is for LLaMA2 arch, may be different for newer models
            prompt = history.pop()
            response = self.model.chat(
                prompt=prompt["content"],
                chat_history=history,
                generate_config={"max_tokens": 2048, "temperature": 0},
            )
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

        msg = response["choices"][0]["message"]["content"]
        token_usage = response["usage"]

        self._update_usage_stats(self.model_name, token_usage)

        self.append_ai_message(msg)

        return msg, token_usage

    def _create_history(self) -> list:
        """Create a history of messages from the conversation.

        Returns
        -------
            list: A list of messages from the conversation.

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

        # concatenate all messages before the last AI message into one message
        if last_ai_message:
            history.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [m for m in msg_texts[:last_ai_message]],
                    ),
                },
            )
            # then append the last AI message
            history.append(
                {
                    "role": "assistant",
                    "content": msg_texts[last_ai_message],
                },
            )

            # then concatenate all messages after that
            # into one HumanMessage
            history.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [m for m in msg_texts[last_ai_message + 1 :]],
                    ),
                },
            )

        # if there is no AI message, concatenate all messages into one user
        # message
        else:
            history.append(
                {
                    "role": "user",
                    "content": "\n".join([m for m in msg_texts[:]]),
                },
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
        """Correct the response from the Xinference API.

        Send the response to a secondary language model. Optionally split the
        response into single sentences and correct each sentence individually.
        Update usage stats.

        Args:
        ----
            msg (str): The response from the model.

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
        history = []
        for m in self.messages:
            if isinstance(m, SystemMessage):
                history.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                history.append({"role": "assistant", "content": m.content})
        prompt = history.pop()
        response = self.ca_model.chat(
            prompt=prompt["content"],
            chat_history=history,
            generate_config={"max_tokens": 2048, "temperature": 0},
        )

        correction = response["choices"][0]["message"]["content"]
        token_usage = response["usage"]

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

    def set_api_key(self) -> bool:
        """Try to get the Xinference model from the client API.

        If the model is found, initialise the conversational agent. If the model
        is not found, `get_model` will raise a RuntimeError.

        Returns
        -------
            bool: True if the model is found, False otherwise.

        """
        try:
            if self.model_name is None or self.model_name == "auto":
                self.model_name = self.list_models_by_type("chat")[0]
            self.model = self.client.get_model(
                self.models[self.model_name]["id"],
            )

            if self.ca_model_name is None or self.ca_model_name == "auto":
                self.ca_model_name = self.list_models_by_type("chat")[0]
            self.ca_model = self.client.get_model(
                self.models[self.ca_model_name]["id"],
            )
            return True

        except RuntimeError:
            self._chat = None
            self._ca_chat = None
            return False

    def list_models_by_type(self, model_type: str) -> list[str]:
        """List the models by type.

        Args:
        ----
            model_type (str): The type of model to list.

        Returns:
        -------
            list[str]: A list of model names.

        """
        names = []
        if model_type in ["embed", "embedding"]:
            for name, model in self.models.items():
                if "model_ability" in model:
                    if "embed" in model["model_ability"]:
                        names.append(name)
                elif model["model_type"] == "embedding":
                    names.append(name)
            return names
        for name, model in self.models.items():
            if "model_ability" in model:
                if model_type in model["model_ability"]:
                    names.append(name)
            elif model["model_type"] == model_type:
                names.append(name)
        return names
