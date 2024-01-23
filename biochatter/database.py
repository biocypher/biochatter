from .prompts import BioCypherPromptEngine
import neo4j_utils as nu


class DatabaseAgent:
    def __init__(self, connection_args):
        """
        Create a DatabaseAgent orthogonal to the VectorDatabaseHostMilvus class,
        which can return results from a database using a query engine. Currently
        limited to Neo4j for development.

        Args:
            connection_args (dict): A dictionary of arguments to connect to the
                database. Contains database name, URI, user, and password.
        """
        self.prompt_engine = BioCypherPromptEngine()
        self.connection_args = connection_args

    def connect(self):
        """
        Connect to the database and authenticate.
        """
        db_name = self.connection_args.get("db_name")
        db_uri = self.connection_args.get("db_uri")
        db_user = self.connection_args.get("db_user")
        db_password = self.connection_args.get("db_password")
        self.driver = nu.Driver(
            db_name=db_name or "neo4j",
            db_uri=db_uri,
            db_user=db_user,
            db_password=db_password,
        )

    def get_query_results(self, query: str, k: int = 3):
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
