import os

from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

from biochatter.llm_connect import LangChainConversation

load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: SecretStr | None = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: str | None = None, **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", openai_api_key=openai_api_key, **kwargs)


class OpenRouterConversation(LangChainConversation):
    """Conversation class for the OpenRouter API."""

    def __init__(self, model_name: str, prompts: dict, **kwargs):
        super().__init__(model_name, "", prompts, **kwargs)

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
            self.chat = ChatOpenRouter(
                model_name=self.model_name,
                temperature=0,
            )
            self.ca_chat = ChatOpenRouter(
                model_name=self.model_name,
                temperature=0,
            )

            # if binding happens here, tools will be available for all messages
            if self.tools:
                self.bind_tools(self.tools)

            return True

        except Exception:
            self._chat = None
            self._ca_chat = None
            return False
