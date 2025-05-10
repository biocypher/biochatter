from collections.abc import Callable

import openai
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from biochatter.llm_connect.openai import GptConversation


class AzureGptConversation(GptConversation):
    """Conversation class for the Azure GPT model."""

    def __init__(
        self,
        deployment_name: str,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
        version: str | None = None,
        base_url: str | None = None,
        update_token_usage: Callable | None = None,
    ) -> None:
        """Connect to Azure's GPT API and set up a conversation with the user.

        Extends GptConversation.

        Args:
        ----
            deployment_name (str): The name of the Azure deployment to use.

            model_name (str): The name of the model to use. This is distinct
                from the deployment name.

            prompts (dict): A dictionary of prompts to use for the conversation.

            correct (bool): Whether to correct the model output.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

            version (str): The version of the Azure API to use.

            base_url (str): The base URL of the Azure API to use.

            update_token_usage (Callable): A function to update the token usage
                statistics.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            update_token_usage=update_token_usage,
        )

        self.version = version
        self.base_url = base_url
        self.deployment_name = deployment_name

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the Azure API.

        If the key is valid, initialise the conversational agent. No user stats
        on Azure.

        Args:
        ----
            api_key (str): The API key for the Azure API.

            user (str, optional): The user for usage statistics.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        try:
            self.chat = AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                openai_api_version=self.version,
                azure_endpoint=self.base_url,
                openai_api_key=api_key,
                temperature=0,
            )
            self.ca_chat = AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                openai_api_version=self.version,
                azure_endpoint=self.base_url,
                openai_api_key=api_key,
                temperature=0,
            )

            self.chat.generate([[HumanMessage(content="Hello")]])
            self.user = user if user is not None else "Azure Community"

            return True

        except openai._exceptions.AuthenticationError:
            self._chat = None
            self._ca_chat = None
            return False

    def _update_usage_stats(self, model: str, token_usage: dict) -> None:
        if self._update_token_usage is not None:
            self._update_token_usage(self.user, model, token_usage)
