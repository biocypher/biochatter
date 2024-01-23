from .prompts import BioCypherPromptEngine
import neo4j_utils as nu


class DatabaseAgent:
    def __init__(
        self,
        model_name: str,
        connection_args: dict,
        schema_config_or_info_dict: dict,
    ) -> None:
        """
        Create a DatabaseAgent analogous to the VectorDatabaseHostMilvus class,
        which can return results from a database using a query engine. Currently
        limited to Neo4j for development.

        Args:
            connection_args (dict): A dictionary of arguments to connect to the
                database. Contains database name, URI, user, and password.
        """
        self.prompt_engine = BioCypherPromptEngine(
            model_name=model_name,
            schema_config_or_info_dict=schema_config_or_info_dict,
        )
        self.connection_args = connection_args

    def connect(self) -> None:
        """
        Connect to the database and authenticate.
        """
        db_name = self.connection_args.get("db_name")
        uri = f"{self.connection_args.get('host')}:{self.connection_args.get('port')}"
        user = self.connection_args.get("user")
        password = self.connection_args.get("password")
        self.driver = nu.Driver(
            db_name=db_name or "neo4j",
            uri=uri,
            user=user,
            password=password,
        )

    def get_query_results(self, query: str, k: int = 3) -> list:
        """
        Generate a query using the prompt engine and return the results.
        Replicates vector database similarity search API.

        Args:
            query (str): A query string.

            k (int): The number of results to return.

        Returns:
            results (list): A list of dictionaries containing the results.
        """
        cypher_query = self.prompt_engine.generate_query(query)
        # TODO some logic if it fails?
        results = self.driver.query(query=cypher_query)
        # TODO result formatting according to k
        return results