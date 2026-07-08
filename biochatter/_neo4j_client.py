"""Thin Neo4j driver wrapper for knowledge-graph query execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver as Neo4jDriverProtocol
    from neo4j.work.summary import ResultSummary


def bolt_uri_from_connection_args(connection_args: dict) -> str:
    """Build a bolt URI from BioChatter-style connection arguments."""
    host = connection_args.get("host", "localhost")
    port = connection_args.get("port", 7687)
    uri = f"{host}:{port}"
    return uri if uri.startswith("bolt://") else f"bolt://{uri}"


class Neo4jClient:
    """Minimal Neo4j client"""

    def __init__(
        self,
        db_name: str,
        db_uri: str,
        db_user: str | None = None,
        db_passwd: str | None = None,
    ) -> None:
        from neo4j import GraphDatabase

        self._db_name = db_name
        self._driver: Neo4jDriverProtocol = GraphDatabase.driver(
            db_uri,
            auth=(db_user, db_passwd),
        )

    def query(
        self,
        query: str,
        parameters: dict | None = None,
        **kwargs,
    ) -> tuple[list[dict] | None, ResultSummary | None]:
        del kwargs
        with self._driver.session(database=self._db_name) as session:
            result = session.run(query, parameters or {})
            return result.data(), result.consume()

    def close(self) -> None:
        self._driver.close()
