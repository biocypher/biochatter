from typing import Optional


class RagAgentModeEnum:
    VectorStore = "vectorstore"
    KG = "kg"


class RagAgent:
    def __init__(
        self,
        mode: str,
        model_name: str,
        connection_args: dict,
        n_results: Optional[int] = 3,
        use_prompt: Optional[bool] = False,
        schema_config_or_info_dict: Optional[dict] = None,
        conversation_factory: Optional[callable] = None,
        embedding_func: Optional[object] = None,
        documentids_workspace: Optional[list[str]] = None,
    ) -> None:
        """
        Create a RAG agent that can return results from a database or vector
        store using a query engine.

        Args:
            mode (str): The mode of the agent. Either "kg" or "vectorstore".

            model_name (str): The name of the model to use.

            connection_args (dict): A dictionary of arguments to connect to the
                database or vector store. Contains database name (in case of
                multiple DBs in one DBMS), host, port, user, and password.

            n_results: the number of results to return for method
                generate_response

            use_prompt (bool): Whether to use the prompt for the query. If
                False, will not retrieve any results and return an empty list.

            schema_config_or_info_dict (dict): A dictionary of schema
                information for the database. Required if mode is "kg".

            conversation_factory (callable): A function used to create a
                conversation for creating the KG query. Required if mode is
                "kg".

            embedding_func (object): An embedding function. Required if mode is
                "vectorstore".

            documentids_workspace (Optional[List[str]], optional): a list of
                document IDs that defines the scope within which similarity
                search occurs. Defaults to None, which means the operations will
                be performed across all documents in the database.

        """
        self.mode = mode
        self.model_name = model_name
        self.use_prompt = use_prompt
        self.n_results = n_results
        self.documentids_workspace = documentids_workspace
        self.last_response = []
        if self.mode == RagAgentModeEnum.KG:
            from .database_agent import DatabaseAgent

            if not schema_config_or_info_dict:
                raise ValueError("Please provide a schema config or info dict.")
            self.schema_config_or_info_dict = schema_config_or_info_dict

            self.agent = DatabaseAgent(
                model_name=model_name,
                connection_args=connection_args,
                schema_config_or_info_dict=self.schema_config_or_info_dict,
                conversation_factory=conversation_factory,
            )

            self.agent.connect()

            self.query_func = self.agent.get_query_results

        elif self.mode == RagAgentModeEnum.VectorStore:
            from .vectorstore_agent import VectorDatabaseAgentMilvus

            if not embedding_func:
                raise ValueError("Please provide an embedding function.")

            self.agent = VectorDatabaseAgentMilvus(
                embedding_func=embedding_func,
                connection_args=connection_args,
            )

            self.agent.connect()

            self.query_func = self.agent.similarity_search
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg' or 'vectorstore'."
            )

    def generate_responses(self, user_question: str) -> list[tuple]:
        """
        Run the query function according to the mode and return the results in a
        uniform format (list of tuples, where the first element is the text for
        RAG and the second element is the metadata).

        Args:
            user_question (str): The user question.

        Returns:
            results (list[tuple]): A list of tuples containing the results.

        Todo:
            Which metadata are returned?
        """
        self.last_response = []
        if not self.use_prompt:
            return []
        if self.mode == RagAgentModeEnum.KG:
            results = self.query_func(user_question, self.n_results)
            response = [
                (
                    result.page_content,
                    result.metadata,
                )
                for result in results
            ]
        elif self.mode == RagAgentModeEnum.VectorStore:
            results = self.query_func(
                user_question,
                self.n_results,
                doc_ids=self.documentids_workspace,
            )
            response = [
                (
                    result.page_content,
                    result.metadata,
                )
                for result in results
            ]
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg' or 'vectorstore'."
            )
        self.last_response = response
        return response
