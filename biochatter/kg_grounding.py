"""
kg_grounding.py
---------------
Entity grounding module for BioChatter KG queries.

Takes a raw user question and resolves biomedical entity mentions
to their canonical names in the Neo4j knowledge graph, using a
frontier model (Gemini) with search tools.

Usage:
    from biochatter.kg_grounding import ground_entities_in_question

    grounded_question, grounded_entities = ground_entities_in_question(
        question="which genes are upregulated in AF cardiomyocytes?",
        connection_args={
            "host": "localhost",
            "port": "7687",
            "db_name": "neo4j",
            "user": "neo4j",
            "password": "neo4j1234",
        }
    )
"""

import logging
import os
import time

import neo4j_utils as nu
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────
GROUNDING_MODEL = "models/gemini-2.5-flash"
MAX_GROUNDING_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 15


def _make_driver(connection_args: dict):
    """Create a Neo4j driver from connection args."""
    host = connection_args.get("host", "localhost")
    port = connection_args.get("port", "7687")
    db_uri = host if host.startswith("bolt://") else f"bolt://{host}:{port}"
    return nu.Driver(
        db_name=connection_args.get("db_name") or "neo4j",
        db_uri=db_uri,
        user=connection_args.get("user"),
        password=connection_args.get("password"),
    )


def _make_tools(driver):
    """
    Create the five search tools for the grounding node.
    Each tool queries Neo4j for a specific entity type.
    The frontier model calls these tools to verify entities exist.
    """

    def search_disease(term: str) -> str:
        """
        Search the knowledge graph for a disease matching the given term.
        Always expand abbreviations before calling this tool.
        For example: search for 'atrial fibrillation' not 'AF',
        search for 'myocardial infarction' not 'heart attack' or 'MI'.
        """
        try:
            results = driver.query(
                """
                MATCH (d:Disease)
                WHERE toLower(d.name) CONTAINS toLower($term)
                RETURN d.name AS name
                LIMIT 5
                """,
                parameters={"term": term},
            )
            if results and results[0]:
                names = [r["name"] for r in results[0] if r.get("name")]
                if names:
                    logger.info(f"search_disease('{term}') -> '{names[0]}'")
                    return names[0]
            all_results = driver.query(
                "MATCH (d:Disease) RETURN d.name AS name"
            )
            all_names = [r["name"] for r in all_results[0] if r.get("name")]
            return f"no match found for '{term}'. Available diseases: {all_names}"
        except Exception as e:
            logger.error(f"search_disease error: {e}")
            return f"search failed: {e}"

    def search_cell_type(term: str) -> str:
        """
        Search the knowledge graph for a cell type matching the given term.
        Use singular form: search for 'cardiomyocyte' not 'cardiomyocytes'.
        """
        try:
            results = driver.query(
                """
                MATCH (c:Cell_type)
                WHERE toLower(c.name) CONTAINS toLower($term)
                RETURN c.name AS name
                LIMIT 5
                """,
                parameters={"term": term},
            )
            if results and results[0]:
                names = [r["name"] for r in results[0] if r.get("name")]
                if names:
                    logger.info(f"search_cell_type('{term}') -> '{names[0]}'")
                    return names[0]
            all_results = driver.query(
                "MATCH (c:Cell_type) RETURN c.name AS name LIMIT 30"
            )
            all_names = [r["name"] for r in all_results[0] if r.get("name")]
            return f"no match found for '{term}'. Available cell types: {all_names}"
        except Exception as e:
            logger.error(f"search_cell_type error: {e}")
            return f"search failed: {e}"

    def search_gene(term: str) -> str:
        """
        Search the knowledge graph for a gene by its HGNC symbol.
        Use the standard symbol: search for 'MYH7' not 'myosin heavy chain'.
        """
        try:
            results = driver.query(
                """
                MATCH (g:Gene)
                WHERE toUpper(g.symbol) = toUpper($term)
                RETURN g.symbol AS symbol
                LIMIT 1
                """,
                parameters={"term": term},
            )
            if results and results[0]:
                symbols = [r["symbol"] for r in results[0] if r.get("symbol")]
                if symbols:
                    logger.info(f"search_gene('{term}') -> '{symbols[0]}'")
                    return symbols[0]
            return f"no gene found matching '{term}'"
        except Exception as e:
            logger.error(f"search_gene error: {e}")
            return f"search failed: {e}"

    def search_tissue(term: str) -> str:
        """
        Search the knowledge graph for a tissue matching the given term.
        For example: search for 'heart' or 'liver' or 'lung'.
        """
        try:
            results = driver.query(
                """
                MATCH (t:Tissue)
                WHERE toLower(t.name) CONTAINS toLower($term)
                RETURN t.name AS name
                LIMIT 5
                """,
                parameters={"term": term},
            )
            if results and results[0]:
                names = [r["name"] for r in results[0] if r.get("name")]
                if names:
                    logger.info(f"search_tissue('{term}') -> '{names[0]}'")
                    return names[0]
            all_results = driver.query(
                "MATCH (t:Tissue) RETURN t.name AS name"
            )
            all_names = [r["name"] for r in all_results[0] if r.get("name")]
            return f"no match found for '{term}'. Available tissues: {all_names}"
        except Exception as e:
            logger.error(f"search_tissue error: {e}")
            return f"search failed: {e}"

    def search_species(term: str) -> str:
        """
        Search the knowledge graph for a species matching the given term.
        For example: search for 'human' or 'mouse' or 'rat'.
        """
        try:
            results = driver.query(
                """
                MATCH (s:Species)
                WHERE toLower(s.name) CONTAINS toLower($term)
                RETURN s.name AS name
                LIMIT 5
                """,
                parameters={"term": term},
            )
            if results and results[0]:
                names = [r["name"] for r in results[0] if r.get("name")]
                if names:
                    logger.info(f"search_species('{term}') -> '{names[0]}'")
                    return names[0]
            all_results = driver.query(
                "MATCH (s:Species) RETURN s.name AS name"
            )
            all_names = [r["name"] for r in all_results[0] if r.get("name")]
            return f"no match found for '{term}'. Available species: {all_names}"
        except Exception as e:
            logger.error(f"search_species error: {e}")
            return f"search failed: {e}"

    return [
        search_disease,
        search_cell_type,
        search_gene,
        search_tissue,
        search_species,
    ]


def _parse_grounding_response(text: str) -> dict:
    """Parse the structured text response from the frontier model."""
    grounded = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("DISEASE:"):
            val = line.split(":", 1)[1].strip()
            grounded["disease"] = None if "not found" in val.lower() else val
        elif line.upper().startswith("CELL_TYPE:"):
            val = line.split(":", 1)[1].strip()
            grounded["cell_type"] = None if "not found" in val.lower() else val
        elif line.upper().startswith("GENE:"):
            val = line.split(":", 1)[1].strip()
            grounded["gene"] = None if "not found" in val.lower() else val
        elif line.upper().startswith("TISSUE:"):
            val = line.split(":", 1)[1].strip()
            grounded["tissue"] = None if "not found" in val.lower() else val
        elif line.upper().startswith("SPECIES:"):
            val = line.split(":", 1)[1].strip()
            grounded["species"] = None if "not found" in val.lower() else val
    return grounded


def _rewrite_question(question: str, grounded: dict) -> str:
    """Append grounded entity context to the question."""
    canonical = []
    if grounded.get("disease"):
        canonical.append(f"disease: {grounded['disease']}")
    if grounded.get("cell_type"):
        canonical.append(f"cell type: {grounded['cell_type']}")
    if grounded.get("gene"):
        canonical.append(f"gene: {grounded['gene']}")
    if grounded.get("tissue"):
        canonical.append(f"tissue: {grounded['tissue']}")
    if grounded.get("species"):
        canonical.append(f"species: {grounded['species']}")

    if not canonical:
        return question

    return (
        f"{question}\n\n"
        f"[Grounded entities for this query: {', '.join(canonical)}. "
        f"Use these exact names when writing the Cypher query.]"
    )


def ground_entities_in_question(
    question: str,
    connection_args: dict,
    api_key: str | None = None,
    model: str = GROUNDING_MODEL,
    max_attempts: int = MAX_GROUNDING_ATTEMPTS,
) -> tuple[str, dict]:
    """
    Main entry point. Ground biomedical entities in the question
    against the Neo4j knowledge graph.

    Args:
        question:        raw user question
        connection_args: Neo4j connection dict with host, port,
                         db_name, user, password
        api_key:         Gemini API key, falls back to GOOGLE_API_KEY env var
        model:           Gemini model to use
        max_attempts:    max retry attempts if grounding fails

    Returns:
        tuple of (grounded_question, grounded_entities_dict)
        If grounding fails, returns (original_question, {})
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.warning(
            "No Gemini API key found. Skipping grounding, "
            "returning original question."
        )
        return question, {}

    try:
        client = genai.Client(api_key=key)
        driver = _make_driver(connection_args)
        tools = _make_tools(driver)
    except Exception as e:
        logger.error(f"Grounding setup failed: {e}")
        return question, {}

    system_instruction = (
        "You are helping a cardiomyopathy researcher query a "
        "biomedical knowledge graph.\n"
        "Instructions:\n"
        "1. Read the question carefully.\n"
        "2. Identify ALL biomedical entities mentioned: "
        "diseases, cell types, genes, tissues, species.\n"
        "3. Expand ALL abbreviations and synonyms to full canonical "
        "names using your biomedical knowledge. "
        "For example: AF -> atrial fibrillation, "
        "MI -> myocardial infarction, "
        "heart attack -> myocardial infarction, "
        "cardiomyocytes -> cardiomyocyte.\n"
        "4. Call the appropriate search tool for each entity to verify "
        "it exists in the graph. Always call tools before responding.\n"
        "5. Use the EXACT values returned by the tools in your response.\n"
        "6. Respond ONLY in this format, one per line:\n"
        "DISEASE: <exact tool result or 'not found'>\n"
        "CELL_TYPE: <exact tool result or 'not found'>\n"
        "GENE: <exact tool result or 'not found'>\n"
        "TISSUE: <exact tool result or 'not found'>\n"
        "SPECIES: <exact tool result or 'not found'>"
    )

    grounded = {}

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            logger.info(
                f"Grounding retry {attempt}, "
                f"waiting {RETRY_WAIT_SECONDS}s..."
            )
            time.sleep(RETRY_WAIT_SECONDS)

        try:
            response = client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=system_instruction,
                ),
            )

            result_text = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    result_text = part.text

            if result_text:
                grounded = _parse_grounding_response(result_text)
                if any(grounded.values()):
                    logger.info(f"Grounding succeeded: {grounded}")
                    return _rewrite_question(question, grounded), grounded

        except Exception as e:
            logger.error(f"Grounding attempt {attempt} failed: {e}")

    logger.warning(
        "Grounding failed after all attempts. "
        "Passing original question to reflexion agent."
    )
    return question, {}
