# ChatGSE LLM connectivity
# connect to API
# keep track of message history
# query API
# correct response
# update usage stats

try:
    import streamlit as st
except ImportError:
    st = None

from abc import ABC, abstractmethod
from typing import Optional
import openai

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import HuggingFaceHub

# To mock Client in tests, we need to import it in advance
from xinference.client import Client

import nltk
import json

from .vectorstore import DocumentEmbedder
from ._stats import get_stats

OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",  # legacy 3.5-turbo, until Sep 13, 2023
    "gpt-3.5-turbo-0613",  # updated 3.5-turbo
    "gpt-3.5-turbo-1106",  # further updated 3.5-turbo
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-1106-preview",  # gpt-4 turbo, 128k tokens
]

HUGGINGFACE_MODELS = ["bigscience/bloom"]

XINFERENCE_MODELS = ["custom-endpoint"]

TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo-0301": 4000,
    "gpt-3.5-turbo-0613": 4000,
    "gpt-3.5-turbo-1106": 16000,
    "gpt-4": 8000,
    "gpt-4-32k": 32000,
    "gpt-4-1106-preview": 128000,
    "bigscience/bloom": 1000,
    "custom-endpoint": 1,  # Reasonable value?
}


class Conversation(ABC):
    """

    Use this class to set up a connection to an LLM API. Can be used to set the
    user name and API key, append specific messages for system, user, and AI
    roles (if available), set up the general context as well as manual and
    tool-based data inputs, and finally to query the API with prompts made by
    the user.

    The conversation class is expected to have a `messages` attribute to store
    the conversation, and a `history` attribute, which is a list of messages in
    a specific format for logging / printing.

    """

    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = True,
        split_correction: bool = False,
        rag_agent: DocumentEmbedder = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.prompts = prompts
        self.correct = correct
        self.split_correction = split_correction
        self.rag_agent = rag_agent
        self.history = []
        self.messages = []
        self.ca_messages = []
        self.current_statements = []

    def set_user_name(self, user_name: str):
        self.user_name = user_name

    @abstractmethod
    def set_api_key(self, api_key: str, user: Optional[str] = None):
        pass

    def get_prompts(self):
        return self.prompts

    def set_prompts(self, prompts: dict):
        self.prompts = prompts

    def set_rag_agent(self, rag_agent: DocumentEmbedder):
        self.rag_agent = rag_agent

    def append_ai_message(self, message: str):
        self.messages.append(
            AIMessage(
                content=message,
            ),
        )

    def append_system_message(self, message: str):
        self.messages.append(
            SystemMessage(
                content=message,
            ),
        )

    def append_ca_message(self, message: str):
        self.ca_messages.append(
            SystemMessage(
                content=message,
            ),
        )

    def append_user_message(self, message: str):
        self.messages.append(
            HumanMessage(
                content=message,
            ),
        )

    def setup(self, context: str):
        """
        Set up the conversation with general prompts and a context.
        """
        for msg in self.prompts["primary_model_prompts"]:
            if msg:
                self.append_system_message(msg)

        for msg in self.prompts["correcting_agent_prompts"]:
            if msg:
                self.append_ca_message(msg)

        self.context = context
        msg = f"The topic of the research is {context}."
        self.append_system_message(msg)

    def setup_data_input_manual(self, data_input: str):
        self.data_input = data_input
        msg = f"The user has given information on the data input: {data_input}."
        self.append_system_message(msg)

    def setup_data_input_tool(self, df, input_file_name: str):
        self.data_input_tool = df

        for tool_name in self.prompts["tool_prompts"]:
            if tool_name in input_file_name:
                msg = self.prompts["tool_prompts"][tool_name].format(df=df)
                self.append_system_message(msg)

    def query(self, text: str, collection_name: Optional[str] = None):
        self.append_user_message(text)

        if self.rag_agent:
            if self.rag_agent.use_prompt:
                self._inject_context(text, collection_name)

        msg, token_usage = self._primary_query()

        if not token_usage:
            # indicates error
            return (msg, token_usage, None)

        if not self.correct:
            return (msg, token_usage, None)

        cor_msg = (
            "Correcting (using single sentences) ..."
            if self.split_correction
            else "Correcting ..."
        )

        if st:
            with st.spinner(cor_msg):
                corrections = self._correct_query(text)
        else:
            corrections = self._correct_query(text)

        if not corrections:
            return (msg, token_usage, None)

        correction = "\n".join(corrections)
        return (msg, token_usage, correction)

    def _correct_query(self, msg: str):
        corrections = []
        if self.split_correction:
            nltk.download("punkt")
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            sentences = tokenizer.tokenize(msg)
            for sentence in sentences:
                correction = self._correct_response(sentence)

                if not str(correction).lower() in ["ok", "ok."]:
                    corrections.append(correction)
        else:
            correction = self._correct_response(msg)

            if not str(correction).lower() in ["ok", "ok."]:
                corrections.append(correction)

        return corrections

    @abstractmethod
    def _primary_query(self, text: str):
        pass

    @abstractmethod
    def _correct_response(self, msg: str):
        pass

    def _inject_context(self, text: str, collection_name: Optional[str] = None):
        """
        Inject the context into the prompt from vector database similarity
        search. Finds the most similar n text fragments and adds them to the
        message history object for usage in the next prompt. Uses the document
        summarisation prompt set to inject the context. The ultimate prompt
        should include the placeholder for the statements, `{statements}` (used
        for formatting the string).

        Args:
            text (str): The user query to be used for similarity search.
        """
        if not self.rag_agent.used:
            st.info(
                "No document has been analysed yet. To use retrieval augmented "
                "generation, please analyse at least one document first."
            )
            return

        sim_msg = (
            f"Performing similarity search to inject {self.rag_agent.n_results}"
            " fragments ..."
        )

        if st:
            with st.spinner(sim_msg):
                statements = [
                    doc.page_content
                    for doc in self.rag_agent.similarity_search(
                        text,
                        self.rag_agent.n_results,
                    )
                ]
        else:
            statements = [
                doc.page_content
                for doc in self.rag_agent.similarity_search(
                    text,
                    self.rag_agent.n_results,
                )
            ]

        prompts = self.prompts["rag_agent_prompts"]
        if statements:
            self.current_statements = statements
            for i, prompt in enumerate(prompts):
                # if last prompt, format the statements into the prompt
                if i == len(prompts) - 1:
                    self.append_system_message(
                        prompt.format(statements=statements)
                    )
                else:
                    self.append_system_message(prompt)

    def get_msg_json(self):
        """
        Return a JSON representation (of a list of dicts) of the messages in
        the conversation. The keys of the dicts are the roles, the values are
        the messages.

        Returns:
            str: A JSON representation of the messages in the conversation.
        """
        d = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "ai"
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")

            d.append({role: msg.content})

        return json.dumps(d)


class XinferenceConversation(Conversation):
    def __init__(
        self,
        base_url: str,
        prompts: dict,
        model_name: str = "auto",
        correct: bool = True,
        split_correction: bool = False,
        rag_agent: DocumentEmbedder = None,
    ):
        """

        Connect to an open-source LLM via the Xinference client library and set
        up a conversation with the user.  Also initialise a second
        conversational agent to provide corrections to the model output, if
        necessary.

        Args:

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

            rag_agent (DocumentEmbedder): A RAG agent to use for retieval
            augmented generation.

        """

        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            rag_agent=rag_agent,
        )
        self.client = Client(base_url=base_url)

        self.models = {}
        self.load_models()

        self.ca_model_name = model_name

        self.set_api_key()

        # TODO make accessible by drop-down

    def load_models(self):
        for id, model in self.client.list_models().items():
            model["id"] = id
            self.models[model["model_name"]] = model

    # def list_models_by_type(self, type: str):
    #     names = []
    #     if type == 'embed' or type == 'embedding':
    #         for name, model in self.models.items():
    #             if "model_ability" in model:
    #                 if "embed" in model["model_ability"]:
    #                     names.append(name)
    #             elif model["model_type"] == "embedding":
    #                 names.append(name)
    #         return names
    #     for name, model in self.models.items():
    #         if "model_ability" in model:
    #             if type in model["model_ability"]:
    #                 names.append(name)
    #         elif model["model_type"] == type:
    #             names.append(name)
    #     return names

    def append_system_message(self, message: str):
        """
        We override the system message addition because Xinference does not
        accept multiple system messages. We concatenate them if there are
        multiple.

        Args:
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

    def append_ca_message(self, message: str):
        """

        We also override the system message addition for the correcting agent,
        likewise because Xinference does not accept multiple system messages. We
        concatenate them if there are multiple.

        TODO this currently assumes that the correcting agent is the same model
        as the primary one.

        Args:
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

    def _primary_query(self):
        """

        Query the Xinference client API with the user's message and return the
        response using the message history (flattery system messages, prior
        conversation) as context. Correct the response if necessary.

        Returns:

            tuple: A tuple containing the response from the Xinference API
            (formatted similarly to responses from the OpenAI API) and the token
            usage.

        """
        try:
            history = []
            for m in self.messages:
                if isinstance(m, SystemMessage):
                    history.append({"role": "system", "content": m.content})
                elif isinstance(m, HumanMessage):
                    history.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    history.append({"role": "assistant", "content": m.content})
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

    def _correct_response(self, msg: str):
        """

        Correct the response from the Xinference API by sending it to a
        secondary language model. Optionally split the response into single
        sentences and correct each sentence individually. Update usage stats.

        Args:
            msg (str): The response from the model.

        Returns:
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
                content="If there is nothing to correct, please respond "
                "with just 'OK', and nothing else!",
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

    def _update_usage_stats(self, model: str, token_usage: dict):
        """
        Update redis database with token usage statistics using the usage_stats
        object with the increment method.

        Args:
            model (str): The model name.

            token_usage (dict): The token usage statistics.
        """
        # if self.user == "community":
        # self.usage_stats.increment(
        #     f"usage:[date]:[user]",
        #     {f"{k}:{model}": v for k, v in token_usage.items()},
        # )

    def set_api_key(self):
        """
        Try to get the Xinference model from the client API. If the model is
        found, initialise the conversational agent. If the model is not found,
        `get_model` will raise a RuntimeError.

        Returns:
            bool: True if the model is found, False otherwise.
        """

        try:
            if self.model_name is None or self.model_name == "auto":
                self.model_name = self.list_models_by_type("chat")[0]
            self.model = self.client.get_model(
                self.models[self.model_name]["id"]
            )

            if self.ca_model_name is None or self.ca_model_name == "auto":
                self.ca_model_name = self.list_models_by_type("chat")[0]
            self.ca_model = self.client.get_model(
                self.models[self.ca_model_name]["id"]
            )
            return True

        except RuntimeError as e:
            # TODO handle error, log?
            return False

    def list_models_by_type(self, type: str):
        names = []
        if type == "embed" or type == "embedding":
            for name, model in self.models.items():
                if "model_ability" in model:
                    if "embed" in model["model_ability"]:
                        names.append(name)
                elif model["model_type"] == "embedding":
                    names.append(name)
            return names
        for name, model in self.models.items():
            if "model_ability" in model:
                if type in model["model_ability"]:
                    names.append(name)
            elif model["model_type"] == type:
                names.append(name)
        return names


class GptConversation(Conversation):
    def __init__(
        self,
        model_name: str,
        prompts: dict,
        correct: bool = True,
        split_correction: bool = False,
        rag_agent: DocumentEmbedder = None,
    ):
        """
        Connect to OpenAI's GPT API and set up a conversation with the user.
        Also initialise a second conversational agent to provide corrections to
        the model output, if necessary.

        Args:
            model_name (str): The name of the model to use.

            prompts (dict): A dictionary of prompts to use for the conversation.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

            rag_agent (DocumentEmbedder): A RAG agent to use for
                retrieval augmented generation (RAG).
        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            rag_agent=rag_agent,
        )

        self.ca_model_name = "gpt-3.5-turbo"
        # TODO make accessible by drop-down

    def set_api_key(self, api_key: str, user: str):
        """
        Set the API key for the OpenAI API. If the key is valid, initialise the
        conversational agent. Set the user for usage statistics.

        Args:
            api_key (str): The API key for the OpenAI API.

            user (str): The user for usage statistics.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        client = openai.OpenAI(
            api_key=api_key,
        )
        self.user = user

        try:
            client.models.list()
            self.chat = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=api_key,
            )
            self.ca_chat = ChatOpenAI(
                model_name=self.ca_model_name,
                temperature=0,
                openai_api_key=api_key,
            )
            if user == "community":
                self.usage_stats = get_stats(user=user)

            return True

        except openai._exceptions.AuthenticationError as e:
            return False

    def _primary_query(self):
        """
        Query the OpenAI API with the user's message and return the response
        using the message history (flattery system messages, prior conversation)
        as context. Correct the response if necessary.

        Returns:
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

    def _correct_response(self, msg: str):
        """
        Correct the response from the OpenAI API by sending it to a secondary
        language model. Optionally split the response into single sentences and
        correct each sentence individually. Update usage stats.

        Args:
            msg (str): The response from the OpenAI API.

        Returns:
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
                content="If there is nothing to correct, please respond "
                "with just 'OK', and nothing else!",
            ),
        )

        response = self.ca_chat.generate([ca_messages])

        correction = response.generations[0][0].text
        token_usage = response.llm_output.get("token_usage")

        self._update_usage_stats(self.ca_model_name, token_usage)

        return correction

    def _update_usage_stats(self, model: str, token_usage: dict):
        """
        Update redis database with token usage statistics using the usage_stats
        object with the increment method.

        Args:
            model (str): The model name.

            token_usage (dict): The token usage statistics.
        """
        if self.user == "community":
            self.usage_stats.increment(
                f"usage:[date]:[user]",
                {f"{k}:{model}": v for k, v in token_usage.items()},
            )


class AzureGptConversation(GptConversation):
    def __init__(
        self,
        deployment_name: str,
        model_name: str,
        prompts: dict,
        correct: bool = True,
        split_correction: bool = False,
        rag_agent: DocumentEmbedder = None,
        version: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Connect to Azure's GPT API and set up a conversation with the user.
        Extends GptConversation.

        Args:
            deployment_name (str): The name of the Azure deployment to use.

            model_name (str): The name of the model to use. This is distinct
                from the deployment name.

            prompts (dict): A dictionary of prompts to use for the conversation.

            split_correction (bool): Whether to correct the model output by
                splitting the output into sentences and correcting each
                sentence individually.

            rag_agent (DocumentEmbedder): A vector database connection to use for
                retrieval augmented generation (RAG).

            version (str): The version of the Azure API to use.

            base_url (str): The base URL of the Azure API to use.
        """
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            correct=correct,
            split_correction=split_correction,
            rag_agent=rag_agent,
        )

        self.version = version
        self.base_url = base_url
        self.deployment_name = deployment_name

    def set_api_key(self, api_key: str, user: Optional[str] = None):
        """
        Set the API key for the Azure API. If the key is valid, initialise the
        conversational agent. No user stats on Azure.

        Args:
            api_key (str): The API key for the Azure API.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """

        try:
            self.chat = AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                openai_api_version=self.version,
                openai_api_base=self.base_url,
                openai_api_key=api_key,
                temperature=0,
            )
            # TODO this is the same model as the primary one; refactor to be
            # able to use any model for correction
            self.ca_chat = AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                openai_api_version=self.version,
                openai_api_base=self.base_url,
                openai_api_key=api_key,
                temperature=0,
            )

            test = self.chat.generate([[HumanMessage(content="Hello")]])

            return True

        except openai._exceptions.AuthenticationError as e:
            return False

    def _update_usage_stats(self, model: str, token_usage: dict):
        """
        We do not track usage stats for Azure.
        """
        return


class BloomConversation(Conversation):
    def __init__(
        self,
        model_name: str,
        prompts: dict,
        split_correction: bool,
        rag_agent: DocumentEmbedder = None,
    ):
        super().__init__(
            model_name=model_name,
            prompts=prompts,
            split_correction=split_correction,
            rag_agent=rag_agent,
        )

        self.messages = []

    def set_api_key(self, api_key: str, user: Optional[str] = None):
        self.chat = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature": 1.0},  # "regular sampling"
            # as per https://huggingface.co/docs/api-inference/detailed_parameters
            huggingfacehub_api_token=api_key,
        )

        try:
            self.chat.generate(["Hello, I am a biomedical researcher."])
            return True
        except ValueError as e:
            return False

    def _cast_messages(self, messages):
        """
        Render the different roles of the chat-based conversation as plain text.
        """
        cast = ""
        for m in messages:
            if isinstance(m, SystemMessage):
                cast += f"System: {m.content}\n"
            elif isinstance(m, HumanMessage):
                cast += f"Human: {m.content}\n"
            elif isinstance(m, AIMessage):
                cast += f"AI: {m.content}\n"
            else:
                raise ValueError(f"Unknown message type: {type(m)}")

        return cast

    def _primary_query(self):
        response = self.chat.generate([self._cast_messages(self.messages)])

        msg = response.generations[0][0].text
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        self.append_ai_message(msg)

        return msg, token_usage

    def _correct_response(self, msg: str):
        return "ok"
