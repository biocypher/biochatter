import json
from collections.abc import Callable

import neo4j_utils as nu
from langchain.schema import Document

from .constants import MAX_AGENT_DESC_LENGTH
from .kg_langgraph_agent import KGQueryReflexionAgent
from .prompts import BioCypherPromptEngine


class DatabaseAgent:
    def __init__(
        self,
        model_name: str,
        connection_args: dict,
        schema_config_or_info_dict: dict,
        conversation_factory: Callable,
        use_reflexion: bool,
    ) -> None:
        """Create a DatabaseAgent analogous to the VectorDatabaseAgentMilvus class,
        which can return results from a database using a query engine. Currently
        limited to Neo4j for development.

        Args:
        ----
            connection_args (dict): A dictionary of arguments to connect to the
                database. Contains database name, URI, user, and password.

            conversation_factory (Callable): A function to create a conversation
                for creating the KG query.

            use_reflexion (bool): Whether to use the ReflexionAgent to generate
                the query.

        """
        self.conversation_factory = conversation_factory
        self.prompt_engine = BioCypherPromptEngine(
            model_name=model_name,
            schema_config_or_info_dict=schema_config_or_info_dict,
            conversation_factory=conversation_factory,
        )
        self.connection_args = connection_args
        self.driver = None
        self.use_reflexion = use_reflexion

    def connect(self) -> None:
        """Connect to the database and authenticate."""
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
        return self.driver is not None

    def _generate_query(self, query: str):
        if self.use_reflexion:
            agent = KGQueryReflexionAgent(
                self.conversation_factory,
                self.connection_args,
            )
            query_prompt = self.prompt_engine.generate_query_prompt(query)
            agent_result = agent.execute(query, query_prompt)
            tool_result = [agent_result.tool_result] if agent_result.tool_result is not None else None
            return agent_result.answer, tool_result
        else:
            query = self.prompt_engine.generate_query(query)
            results = self.driver.query(query=query)
            return query, results

    def _build_response(
        self,
        results: list[dict],
        cypher_query: str,
        results_num: int | None = 3,
    ) -> list[Document]:
        if len(results) == 0:
            return [
                Document(
                    page_content=(
                        "I didn't find any result in knowledge graph, "
                        f"but here is the query I used: {cypher_query}. "
                        "You can ask user to refine the question. "
                        "Note: please ensure to include the query in a code "
                        "block in your response so that the user can refine "
                        "their question effectively."
                    ),
                    metadata={"cypher_query": cypher_query},
                ),
            ]

        clipped_results = results[:results_num] if results_num > 0 else results
        results_dump = json.dumps(clipped_results)

        return [
            Document(
                page_content=(
                    "The results retrieved from knowledge graph are: "
                    f"{results_dump}. "
                    f"The query used is: {cypher_query}. "
                    "Note: please ensure to include the query in a code block "
                    "in your response so that the user can refine "
                    "their question effectively."
                ),
                metadata={"cypher_query": cypher_query},
            ),
        ]

    def get_query_results(self, query: str, k: int = 3) -> list[Document]:
        """Generate a query using the prompt engine and return the results.
        Replicates vector database similarity search API. Results are returned
        as a list of Document objects to align with the vector database agent.

        Args:
        ----
            query (str): A query string.

            k (int): The number of results to return.

        Returns:
        -------
            List[Document]: A list of Document objects. The page content values
                are the literal dictionaries returned by the query, the metadata
                values are the cypher query used to generate the results, for
                now.

        """
        (cypher_query, tool_result) = self._generate_query(
            query,
        )  # self.prompt_engine.generate_query(query)
        # TODO some logic if it fails?
        if tool_result is not None:
            # If _generate_query() already returned tool_result, we won't connect
            # to graph database to query result any more
            results = tool_result
        else:
            results = self.driver.query(query=cypher_query)

        # return first k results
        # returned nodes can have any formatting, and can also be empty or fewer
        # than k
        if results is None or len(results) == 0 or results[0] is None:
            return []
        return self._build_response(
            results=results[0],
            cypher_query=cypher_query,
            results_num=k,
        )

    def get_description(self):
        result = self.driver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

        if result[0]:
            schema_info_node = result[0][0]["n"]
            schema_dict_content = schema_info_node["schema_info"][:MAX_AGENT_DESC_LENGTH]  # limit to 1000 characters
            return f"the graph database contains the following nodes and edges: \n\n{schema_dict_content}"

        # schema_info is not found in database
        nodes_query = "MATCH (n) RETURN DISTINCT labels(n) LIMIT 300"
        node_results = self.driver.query(query=nodes_query)
        edges_query = "MATCH (n) RETURN DISTINCT type(n) LIMIT 300"
        edge_results = self.driver.query(query=edges_query)
        desc = (
            f"The graph database contains the following nodes and edges: \n"
            f"nodes: \n{node_results}"
            f"edges: \n{edge_results}"
        )
        return desc[:MAX_AGENT_DESC_LENGTH]
