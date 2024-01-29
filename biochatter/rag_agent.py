from typing import Optional


class RagAgent:
    def __init__(
        self,
        mode: str,
        model_name: str,
        connection_args: dict,
        schema_config_or_info_dict: Optional[dict] = None,
        embedding_func: Optional[object] = None,
        embedding_collection_name: Optional[str] = None,
        metadata_collection_name: Optional[str] = None,
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

            schema_config_or_info_dict (dict): A dictionary of schema
                information for the database. Required if mode is "kg".

            embedding_func (object): An embedding function. Required if mode is
                "vectorstore".

            embedding_collection_name (str): The name of the embedding
                collection. Required if mode is "vectorstore".

            metadata_collection_name (str): The name of the metadata
                collection. Required if mode is "vectorstore".
        """
        self.mode = mode
        self.model_name = model_name
        if self.mode == "kg":
            from .database_agent import DatabaseAgent

            if not schema_config_or_info_dict:
                raise ValueError("Please provide a schema config or info dict.")
            self.schema_config_or_info_dict = schema_config_or_info_dict

            self.agent = DatabaseAgent(
                model_name=model_name,
                connection_args=connection_args,
                schema_config_or_info_dict=self.schema_config_or_info_dict,
            )

            self.agent.connect()

            self.query_func = self.agent.get_query_results

        elif self.mode == "vectorstore":
            from .vectorstore_agent import VectorDatabaseAgentMilvus

            if not embedding_func:
                raise ValueError("Please provide an embedding function.")
            if not embedding_collection_name:
                raise ValueError("Please provide an embedding collection name.")
            if not metadata_collection_name:
                raise ValueError("Please provide a metadata collection name.")

            self.agent = VectorDatabaseAgentMilvus(
                embedding_func=embedding_func,
                connection_args=connection_args,
                embedding_collection_name=embedding_collection_name,
                metadata_collection_name=metadata_collection_name,
            )

            self.agent.connect()

            self.query_func = self.agent.similarity_search
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg' or 'vectorstore'."
            )

    def generate_responses(self, user_question: str, k: int = 3) -> list[tuple]:
        """
        Run the query function according to the mode and return the results in a
        uniform format (list of tuples, where the first element is the text for
        RAG and the second element is the metadata).

        Args:
            user_question (str): The user question.

            k (int): The number of results to return.

        Returns:
            results (list[tuple]): A list of tuples containing the results.

        Todo:
            Which metadata are returned?
        """
        results = self.query_func(user_question, k)
        if self.mode == "kg":
            return [
                (
                    result.page_content,
                    result.metadata,
                )
                for result in results
            ]
        elif self.mode == "vectorstore":
            return [
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
