import json
from collections.abc import Callable

import litellm
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

from biochatter._stats import get_stats
from biochatter.llm_connect import Conversation


class LiteLLMConversation(Conversation):
    """A unified interface for multiple LLM models using LiteLLM.

    This class implements the abstract methods from the Conversation parent class
    and provides a unified way to interact with different LLM providers through
    LiteLLM, which supports models from OpenAI, Anthropic, HuggingFace, and more.

    Attributes:
        model_name (str): The name of the model to use.
        prompts (dict): Dictionary containing various prompts used in the conversation.
        correct (bool): Whether to use a correcting agent.
        split_correction (bool): Whether to split corrections by sentence.
        rag_agents (list): List of RAG agents available for context enhancement.
        history (list): Conversation history for logging/printing.
        messages (list): Messages in the conversation.
        ca_messages (list): Messages for the correcting agent.
        api_key (str): API key for the LLM provider.
        user (str): Username for the API, if required.

    """

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        use_ragagent_selector: bool = False,
        update_token_usage: Callable | None = None,
    ) -> None:
        """Initialize a UnifiedConversation instance.

        Args:
            model_name (str): The name of the model to use.
            prompts (dict): Dictionary containing various prompts used in the conversation.
            correct (bool): Whether to use a correcting agent. Defaults to False.
            split_correction (bool): Whether to split corrections by sentence. Defaults to False.
            use_ragagent_selector (bool): Whether to use RagAgentSelector. Defaults to False.
            update_token_usage (Callable): A function to update the token usage statistics.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            use_ragagent_selector=use_ragagent_selector,
        )
        self.api_key = None
        self.user = None
        self.ca_model_name = model_name
        self._update_token_usage = update_token_usage

    def get_litellm_object(self, api_key: str, model: str) -> ChatLiteLLM:
        """Get a LiteLLM object for the specified model and API key.

        Args:
            api_key (str): The API key for the LLM provider.
            model (str): The name of the model to use.

        Returns:
            ChatLiteLLM: An instance of ChatLiteLLM configured with the specified model, temperature, max tokens and API key.

        Raises:
            ValueError: If the API key is None.
            litellm.exceptions.AuthenticationError: If there is an authentication error.
            litellm.exceptions.InvalidRequestError: If the request is invalid.
            litellm.exceptions.RateLimitError: If the rate limit is exceeded.
            litellm.exceptions.ServiceUnavailableError: If the service is unavailable.
            litellm.exceptions.APIError: If there is a general API error.
            litellm.exceptions.Timeout: If the request times out.
            litellm.exceptions.APIConnectionError: If there is a connection error.
            litellm.exceptions.InternalServerError: If there is an internal server error.
            Exception: If there is an unexpected error.

        """
        if api_key is None:
            raise ValueError("API key must not be None")

        try:
            max_tokens = self.get_model_max_tokens(model)
        except:
            max_tokens = None

        kwargs = {"temperature": 0, "max_token": max_tokens, "model_name": model}

        if self.model_name.startswith("gpt-"):
            api_key_kwarg = "openai_api_key"
        elif self.model_name.startswith("claude-"):
            api_key_kwarg = "anthropic_api_key"
        elif self.model_name.startswith("azure/"):
            api_key_kwarg = "azure_api_key"
        elif self.model_name.startswith("mistral/") or self.model_name in [
            "mistral-tiny",
            "mistral-small",
            "mistral-medium",
            "mistral-large-latest",
        ]:
            api_key_kwarg = "api_key"
        else:
            api_key_kwarg = "api_key"

        kwargs[api_key_kwarg] = api_key
        try:
            return ChatLiteLLM(**kwargs)

        except (
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.InvalidRequestError,
            litellm.exceptions.RateLimitError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.APIError,
            litellm.exceptions.Timeout,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.InternalServerError,
        ) as api_setup_error:
            raise api_setup_error
        except Exception as e:
            raise e

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the LLM provider.

        Args:
            api_key (str): The API key for the LLM provider.
            user (Union[str, None]): The username

        Returns:
            bool: True if the API key is successfully set, False otherwise.

        Raises:
            ValueError: If the model name or correction model name is not set.
            TypeError: If the LiteLLM object initialization fails.
            Exception: If there is an unexpected error.

        """
        try:
            if self.model_name is None:
                raise ValueError("Primary Model name is not set.")

            if self.ca_model_name is None:
                raise ValueError("Correction Model name is not set.")

            self.chat = self.get_litellm_object(api_key, self.model_name)
            if self.chat is None:
                raise TypeError("Failed to intialize primary agent chat object.")

            self.ca_chat = self.get_litellm_object(api_key, self.ca_model_name)
            if self.ca_chat is None:
                raise TypeError("Failed to intialize correcting agent chat object.")

            self.user = user
            if user == "community":
                self.usage_stats = get_stats(user=user)
            return True

        except (ValueError, TypeError):
            self.chat = None
            self.ca_chat = None
            return False
        except Exception:
            self.chat = None
            self.ca_chat = None
            return False

    def json_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if obj is None:
            raise ValueError("Object is None")
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        try:
            return str(obj)
        except:
            return repr(obj)

    def parse_llm_response(self, response) -> dict | None:
        """Parse the response from the LLM."""
        try:
            full_json = json.loads(json.dumps(response, default=self.json_serializable))

            if not full_json.get("generations"):
                return None

            generations = full_json["generations"]
            if not generations or not generations[0]:
                return None

            first_generation = generations[0][0]
            if not first_generation or not first_generation.get("message"):
                return None

            message = first_generation["message"]
            if not message.get("response_metadata"):
                return None

            response_metadata = message["response_metadata"]
            if not response_metadata.get("token_usage"):
                return None

            return response_metadata["token_usage"]

        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            print(f"Error parsing LLM response: {e}")
            return None

        except Exception as e:
            print(f"Unexpected error while parsing LLM response: {e}")
            return None

    def _primary_query(self, **kwargs) -> tuple:
        """Query the LLM API with the user's message.

        Return the response using the message history (flattery system messages,
        prior conversation) as context. Correct the response if necessary.

        Args:
        ----
            **kwargs: Keyword arguments (not used by this basic LiteLLM implementation,
                     but accepted for compatibility with the base Conversation interface)

        Returns:
            tuple: A tuple containing the response from the LLM API and the token usage.

        """
        try:
            response = self.chat.generate([self.messages])
        except (
            AttributeError,
            litellm.exceptions.APIError,
            litellm.exceptions.OpenAIError,
            litellm.exceptions.RateLimitError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.PermissionDeniedError,
            litellm.exceptions.UnprocessableEntityError,
            litellm.exceptions.APIResponseValidationError,
            litellm.exceptions.BudgetExceededError,
            litellm.exceptions.RejectedRequestError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.Timeout,
        ) as e:
            return e, None
        except Exception as e:
            return e, None

        msg = response.generations[0][0].text
        token_usage = self.parse_llm_response(response)

        self.append_ai_message(msg)

        self._update_usage_stats(self.model_name, token_usage)

        return msg, token_usage

    def _correct_response(self, msg: str) -> str:
        """Correct the response from the LLM.

        Args:
            msg (str): The response message to correct.

        Returns:
            str: The corrected response message.

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
        token_usage = self.parse_llm_response(response)

        self._update_usage_stats(self.ca_model_name, token_usage)

        return correction

    def _update_usage_stats(self, model: str, token_usage: dict) -> None:
        """Update the usage statistics.

        Args:
            model (str): The model name.
            token_usage (dict): The token usage information.

        """
        if self.user == "community" and model:
            stats_dict = {f"{k}:{model}": v for k, v in token_usage.items() if isinstance(v, int | float)}
            self.usage_stats.increment(
                "usage:[date]:[user]",
                stats_dict,
            )

        if self._update_token_usage is not None:
            self._update_token_usage(self.user, model, token_usage)

    def get_all_model_list(self) -> list:
        """Get a list of all available models."""
        return litellm.model_list

    def get_models_by_provider(self):
        """Get a dictionary of models grouped by their provider."""
        return litellm.models_by_provider

    def get_all_model_info(self) -> dict:
        """Get information about all available models."""
        return litellm.model_cost

    def get_model_info(self, model: str) -> dict:
        """Get information about a specific model.

        Args:
            model (str): The name of the model.

        Returns:
            dict: A dictionary containing information about the specified model.

        """
        models_info: dict = self.get_all_model_info()
        if model not in models_info:
            raise litellm.exceptions.NotFoundError(
                f"{model} model's information is not available.", model=model, llm_provider="Unknown"
            )
        return models_info[model]

    def get_model_max_tokens(self, model: str) -> int:
        """Get the maximum number of tokens for a specific model.

        Args:
            model (str): The name of the model.

        Returns:
            int: The maximum number of tokens for the specified model.

        """
        try:
            model_info = self.get_model_info(model)
            if "max_tokens" not in model_info:
                raise litellm.exceptions.NotFoundError(
                    f"Max token information for {model} is not available.", model=model, llm_provider="Unknown"
                )
            return model_info["max_tokens"]
        except litellm.exceptions.NotFoundError as e:
            raise e
