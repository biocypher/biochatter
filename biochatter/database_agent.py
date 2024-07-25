from collections.abc import Callable
import json

from langchain.schema import Document
import neo4j_utils as nu

from .prompts import BioCypherPromptEngine
from .kg_langgraph_agent import KGQueryReflexionAgent


class DatabaseAgent:
    def __init__(
        self,
        model_name: str,
        connection_args: dict,
        schema_config_or_info_dict: dict,
        conversation_factory: Callable,
    ) -> None:
        """
        Create a DatabaseAgent analogous to the VectorDatabaseAgentMilvus class,
        which can return results from a database using a query engine. Currently
        limited to Neo4j for development.

        Args:
            connection_args (dict): A dictionary of arguments to connect to the
                database. Contains database name, URI, user, and password.

            conversation_factory (Callable): A function to create a conversation
                for creating the KG query.
        """
        self.conversation_factory = conversation_factory
        self.prompt_engine = BioCypherPromptEngine(
            model_name=model_name,
            schema_config_or_info_dict=schema_config_or_info_dict,
            conversation_factory=conversation_factory,
        )
        self.connection_args = connection_args
        self.driver = None

    def connect(self) -> None:
        """
        Connect to the database and authenticate.
        """
        db_name = self.connection_args.get("db_name")
        uri = f"{self.connection_args.get('host')}:{self.connection_args.get('port')}"
        uri = uri if uri.startswith("bolt://") else "bolt://" + uri
        user = self.connection_args.get("user")
        password = self.connection_args.get("password")
        self.driver = nu.Driver(
            db_name=db_name or "neo4j",
            db_uri=uri,
            user=user,
            password=password,
        )

    def is_connected(self) -> bool:
        return not self.driver is None

    def _generate_query(self, query: str):
        agent = KGQueryReflexionAgent(
            self.conversation_factory,
            self.connection_args,
        )
        query_prompt = self.prompt_engine.generate_query_prompt(query)
        cypher_query = agent.execute(query, query_prompt)
        return cypher_query

    def get_query_results(self, query: str, k: int = 3) -> list[Document]:
        """
        Generate a query using the prompt engine and return the results.
        Replicates vector database similarity search API. Results are returned
        as a list of Document objects to align with the vector database agent.

        Args:
            query (str): A query string.

            k (int): The number of results to return.

        Returns:
            list[Document]: A list of Document objects. The page content values
                are the literal dictionaries returned by the query, the metadata
                values are the cypher query used to generate the results, for
                now.
        """
        cypher_query = self._generate_query(
            query
        )  # self.prompt_engine.generate_query(query)
        # TODO some logic if it fails?
        results = self.driver.query(query=cypher_query)

        documents = []
        # return first k results
        # returned nodes can have any formatting, and can also be empty or fewer
        # than k
        if results is None or len(results) == 0 or results[0] is None:
            return []
        for result in results[0]:
            documents.append(
                Document(
                    page_content=json.dumps(result),
                    metadata={
                        "cypher_query": cypher_query,
                    },
                )
            )
            if len(documents) == k:
                break

        return documents

    def get_description(self):
        try:
            result = self.driver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

            if result[0]:
                schema_info_node = result[0][0]["n"]
                MAX_SCHEMA_INFO_LENGTH = 1000
                schema_dict_content = schema_info_node["schema_info"][:MAX_SCHEMA_INFO_LENGTH] # limit to 1000 characters
                return (f"the graph database contains the following nodes and edges: \n\n"
                        f"{schema_dict_content}")
        except Exception:
            pass # failed to inquire shcema info
        
        nodes_query = "MATCH (n) RETURN DISTINCT labels(n) LIMIT 300"
        node_results = self.driver.query(query=nodes_query)
        edges_query = "MATCH (n) RETURN DISTINCT type(n) LIMIT 300"
        edge_results = self.driver.query(query=edges_query)
        return (f"The graph database contains the following nodes and edges: \n"
                f"nodes: \n{node_results}"
                f"edges: \n{edge_results}")
