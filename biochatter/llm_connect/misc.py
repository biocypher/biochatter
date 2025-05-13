from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biochatter.llm_connect.conversation import Conversation


class WasmConversation(Conversation):
    """Conversation class for the wasm model."""

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = False,
        split_correction: bool = False,
    ) -> None:
        """Initialize the WasmConversation class.

        This class is used to return the complete query as a string to be used
        in the frontend running the wasm model. It does not call the API itself,
        but updates the message history similarly to the other conversation
        classes. It overrides the `query` method from the `Conversation` class
        to return a plain string that contains the entire message for the model
        as the first element of the tuple. The second and third elements are
        `None` as there is no token usage or correction for the wasm model.

        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
        )

    def query(self, text: str) -> tuple:
        """Return the entire message history as a single string.

        This is the message that is sent to the wasm model.

        Args:
        ----
            text (str): The user query.

        Returns:
        -------
            tuple: A tuple containing the message history as a single string,
                and `None` for the second and third elements of the tuple.

        """
        self.append_user_message(text)

        self._inject_context(text)

        return (self._primary_query(), None, None)

    def _primary_query(self):
        """Concatenate all messages in the conversation.

        Build a single string from all messages in the conversation.
        Currently discards information about roles (system, user).

        Returns
        -------
            str: A single string from all messages in the conversation.

        """
        return "\n".join([m.content for m in self.messages])

    def _correct_response(self, msg: str) -> str:
        """Do not use for the wasm model."""
        return "ok"

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Do not use for the wasm model."""
        return True


class BloomConversation(Conversation):
    """Conversation class for the Bloom model."""

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        split_correction: bool,
    ) -> None:
        """Initialise the BloomConversation class.

        DEPRECATED: Superceded by XinferenceConversation.
        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            split_correction=split_correction,
        )

        self.messages = []

    def set_api_key(self, api_key: str, user: str | None = None) -> bool:
        """Set the API key for the HuggingFace API.

        If the key is valid, initialise the conversational agent.

        Args:
        ----
            api_key (str): The API key for the HuggingFace API.

            user (str): The user for usage statistics.

        Returns:
        -------
            bool: True if the API key is valid, False otherwise.

        """
        self.chat = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature": 1.0},  # "regular sampling"
            # as per https://huggingface.co/docs/api-inference/detailed_parameters
            huggingfacehub_api_token=api_key,
        )

        try:
            self.chat.generate(["Hello, I am a biomedical researcher."])
            return True
        except ValueError:
            return False

    def _cast_messages(self, messages: list) -> str:
        """Render the different roles of the chat-based conversation."""
        cast = ""
        for m in messages:
            if isinstance(m, SystemMessage):
                cast += f"System: {m.content}\n"
            elif isinstance(m, HumanMessage):
                cast += f"Human: {m.content}\n"
            elif isinstance(m, AIMessage):
                cast += f"AI: {m.content}\n"
            else:
                error_msg = f"Unknown message type: {type(m)}"
                raise TypeError(error_msg)

        return cast

    def _primary_query(self) -> tuple:
        response = self.chat.generate([self._cast_messages(self.messages)])

        msg = response.generations[0][0].text
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        self.append_ai_message(msg)

        return msg, token_usage

    def _correct_response(self, msg: str) -> str:
        return "ok"
