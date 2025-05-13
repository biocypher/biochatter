import openai
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biochatter.llm_connect.conversation import Conversation


class OllamaConversation(Conversation):
    """Conversation class for the Ollama model."""

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the Ollama API. Not implemented.

        Args:
        ----
            api_key (str): The API key for the Ollama API.

            user (str): The user for usage statistics.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        err = "Ollama does not require an API key."
        raise NotImplementedError(err)

    def __init__(
        self,
        base_url: str,
        prompts: dict,
        model_name: str = "llama3",
        correct: bool = False,
        split_correction: bool = False,
    ) -> None:
        """Connect to an Ollama LLM via the Ollama/Langchain library.

        Set up a conversation with the user. Also initialise a second
        conversational agent to provide corrections to the model output, if
        necessary.

        Args:
        ----
            base_url (str): The base URL of the Ollama instance.

            prompts (dict): A dictionary of prompts to use for the conversation.

            model_name (str): The name of the model to use. Can be any model
                name available in your Ollama instance.

            correct (bool): Whether to correct the model output.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each sentence
                individually.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
        )
        self.model_name = model_name
        self.model = ChatOllama(
            base_url=base_url,
            model=self.model_name,
            temperature=0.0,
        )

        self.ca_model_name = "mixtral:latest"

        self.ca_model = ChatOllama(
            base_url=base_url,
            model_name=self.ca_model_name,
            temperature=0.0,
        )

    def append_system_message(self, message: str) -> None:
        """Override the system message addition.

        Ollama does not accept multiple system messages. Concatenate them if
        there are multiple.

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

        Ollama does not accept multiple system messages. Concatenate them if
        there are multiple.

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
        """Query the Ollama client API with the user's message.

        Return the response using the message history (flattery system messages,
        prior conversation) as context. Correct the response if necessary.

        Returns
        -------
            tuple: A tuple containing the response from the Ollama API
            (formatted similarly to responses from the OpenAI API) and the token
            usage.

        """
        try:
            messages = self._create_history(self.messages)
            response = self.model.invoke(
                messages,
                # ,generate_config={"max_tokens": 2048, "temperature": 0},
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
        response_dict = response.dict()
        msg = response_dict["content"]
        token_usage = response_dict["response_metadata"]["eval_count"]

        self._update_usage_stats(self.model_name, token_usage)

        self.append_ai_message(msg)

        return msg, token_usage

    def _create_history(self, messages: list) -> list:
        history = []
        for _, m in enumerate(messages):
            if isinstance(m, AIMessage):
                history.append(AIMessage(content=m.content))
            elif isinstance(m, HumanMessage):
                history.append(HumanMessage(content=m.content))
            elif isinstance(m, SystemMessage):
                history.append(SystemMessage(content=m.content))

        return history

    def _correct_response(self, msg: str) -> str:
        """Correct the response from the Ollama API.

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
        response = self.ca_model.invoke(
            chat_history=self._create_history(self.messages),
        ).dict()
        correction = response["content"]
        token_usage = response["eval_count"]

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
