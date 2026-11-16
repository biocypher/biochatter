# Copyright (c) 2026, BioChatter contributors
# SPDX-License-Identifier: Apache-2.0
"""Run a local Knowledge Graph RAG demo against the external POLE dataset.

Prerequisites:
1. Clone and build the external POLE knowledge graph:
   https://github.com
   (Set `file_format: csv` in `pole/config/biocypher_config.yaml` if using
   BioCypher >= 0.17.0, then import into a local Neo4j instance.)
2. Set environment variables:
   export POLE_REPO_PATH="/path/to/pole"
   export NEO4J_PASSWORD="your_neo4j_password"
   # Optional (if using OpenAI instead of local Ollama):
   # export OPENAI_API_KEY="sk-..."
3. Ensure Ollama is running locally (`ollama pull llama3.1:8b`), or set
   `OPENAI_API_KEY` to use `GptConversation`.
"""

import logging
import os
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from biochatter.llm_connect import Conversation, GptConversation, OllamaConversation
from biochatter.prompts import BioCypherPromptEngine

logger = logging.getLogger(__name__)


class PoleDemoConfigError(RuntimeError):
    """Raised when required POLE demo environment variables are missing."""


def create_conversation() -> Conversation:
    """Return a local Ollama conversation, or OpenAI if configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        convo = GptConversation(model_name="gpt-4o-mini", prompts={})
        convo.set_api_key(api_key=api_key)
        return convo

    return OllamaConversation(
        model_name=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        prompts={},
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def main() -> None:
    """Run POLE Cypher query generation and summarise Neo4j results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pole_repo = os.getenv("POLE_REPO_PATH")
    if not pole_repo:
        raise PoleDemoConfigError

    schema_path = Path(pole_repo) / "config" / "schema_config.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        raise PoleDemoConfigError

    prompt_engine = BioCypherPromptEngine(
        schema_config_or_info_path=str(schema_path),
        conversation_factory=create_conversation,
    )

    questions = [
        "How many crimes happened in total?",
        "Where happened most crimes?",
        "Who were the top 3 offenders?",
    ]

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        for question in questions:
            logger.info("=" * 60)
            logger.info("Question: %s", question)

            cypher_query = prompt_engine.generate_query(
                question=question,
                query_language="Cypher",
            )
            logger.info("Generated Cypher:\n%s", cypher_query)

            try:
                with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                    result = session.run(cypher_query)
                    records = [record.data() for record in result]
            except Neo4jError:
                logger.exception("Cypher execution failed for question: %s", question)
                continue

            if records:
                convo = create_conversation()
                final_answer, _, _ = convo.query(
                    "Based on the following retrieved knowledge graph results, "
                    "answer the user's question concisely.\n\n"
                    f"Question: {question}\n"
                    f"Cypher query used: {cypher_query}\n"
                    f"Results: {records}"
                )
                logger.info("Final Answer: %s", final_answer)
            else:
                logger.info("No results returned from knowledge graph.")


if __name__ == "__main__":
    main()
