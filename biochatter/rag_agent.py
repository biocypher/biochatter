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
        self.mode = mode
        self.model_name = model_name
        if self.mode == "kg":
            from .prompts import BioCypherPromptEngine

            if not schema_config_or_info_dict:
                raise ValueError("Please provide a schema config or info dict.")
            self.schema_config_or_info_dict = schema_config_or_info_dict

            self.query_func = BioCypherPromptEngine(
                model_name=model_name,
                schema_config_or_info_dict=self.schema_config_or_info_dict,
            )
        elif self.mode == "vectorstore":
            from .vectorstore_host import VectorDatabaseHostMilvus

            if not embedding_func:
                raise ValueError("Please provide an embedding function.")
            if not embedding_collection_name:
                raise ValueError("Please provide an embedding collection name.")
            if not metadata_collection_name:
                raise ValueError("Please provide a metadata collection name.")

            self.query_func = VectorDatabaseHostMilvus(
                embedding_func=embedding_func,
                connection_args=connection_args,
                embedding_collection_name=embedding_collection_name,
                metadata_collection_name=metadata_collection_name,
            )
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg' or 'vectorstore'."
            )

    def generate_responses(self, user_question):
        return self.query_func(user_question)
