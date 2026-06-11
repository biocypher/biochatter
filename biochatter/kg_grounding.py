"""
kg_grounding.py
---------------
Entity grounding module for BioChatter KG queries.

Uses a three-agent pipeline:

    Agent 1 — EntityIdentifierAgent:
        LLM identifies all biomedical entities in the question
        and their types. No tool calls. Returns structured list.

    Agent 2 — TierSelectorAgent:
        LLM decides which grounding tier to use for each entity.
        No tool calls. Returns tier decision.
        tier1: common terms, abbreviations, synonyms, misspellings
        tier2: rare/specific terms needing ontology resolution
        tier3: unknown/ambiguous terms

    Agent 3 — GroundingExecutorAgent:
        Pure execution — no AFC, no tool calling unreliability.
        tier1: LLM expands term → Neo4j name match
        tier2: ontomcp → ontology ID → Neo4j ID match
        tier3: Neo4j candidates → LLM picks closest

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
        },
        schema=schema_dict,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time

import neo4j_utils as nu
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
GROUNDING_MODEL = "models/gemini-2.5-flash"
MAX_GROUNDING_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 15
ONTOMCP_SHARD_DIR = os.getenv(
    "ONTOMCP_SHARD_DIR",
    os.path.expanduser("~/Desktop/code/ontomcp/shards"),
)
SKIP_ENTITY_TYPES = {"de_result", "study_details", "schema_info"}
SHARD_MAP = {
    "disease": "mondo",
    "cell_type": "cl",
    "tissue": "uberon",
    "gene": "hgnc",
}


# ── Neo4j driver ───────────────────────────────────────────────────────────────

def _make_driver(connection_args: dict):
    """Create a Neo4j driver from connection args dict."""
    host = connection_args.get("host", "localhost")
    port = connection_args.get("port", "7687")
    db_uri = host if host.startswith("bolt://") else f"bolt://{host}:{port}"
    return nu.Driver(
        db_name=connection_args.get("db_name") or "neo4j",
        db_uri=db_uri,
        user=connection_args.get("user"),
        password=connection_args.get("password"),
    )


# ── ontomcp setup ──────────────────────────────────────────────────────────────

def _make_ontomcp_manager():
    """
    Initialise the ontomcp shard manager.
    Returns (manager, find_terms_fn) or (None, None) if not available.
    """
    try:
        from ontomcp.retrieval.shard_manager import ShardManager
        from ontomcp.retrieval.searcher import find_terms as _find_terms

        if not os.path.exists(ONTOMCP_SHARD_DIR):
            logger.warning(
                f"ontomcp shard directory not found: {ONTOMCP_SHARD_DIR}. "
                "Tier 2 grounding will be skipped."
            )
            return None, None

        manager = ShardManager(ONTOMCP_SHARD_DIR)
        available = manager.list_ids()
        if not available:
            logger.warning("No ontomcp shards found. Tier 2 will be skipped.")
            return None, None

        logger.info(f"ontomcp shards available: {available}")
        return manager, _find_terms

    except ImportError:
        logger.warning("ontomcp not installed. Tier 2 grounding will be skipped.")
        return None, None


# ── schema parsing ─────────────────────────────────────────────────────────────

def _get_searchable_entity_types(schema: dict) -> list[str]:
    """
    Extract searchable entity types from a BioCypher schema dict.
    Returns list of entity type names that are nodes with string properties.
    """
    entity_types = []
    for entity_name, config in schema.items():
        if not isinstance(config, dict):
            continue
        if config.get("represented_as") != "node":
            continue
        if not config.get("present_in_knowledge_graph", False):
            continue
        if entity_name.lower() in SKIP_ENTITY_TYPES:
            continue
        properties = config.get("properties", {})
        if not any(t == "str" for t in properties.values()):
            continue
        entity_types.append(entity_name)
    return entity_types


# ── Agent 1: EntityIdentifierAgent ────────────────────────────────────────────

def _identify_entities(
    question: str,
    entity_types: list[str],
    client,
    model: str,
) -> list[tuple[str, str]]:
    """
    Agent 1: identify biomedical entities in the question.
    Returns list of (term, entity_type) tuples.
    Pure LLM call, no tools.
    """
    entity_list = ", ".join(entity_types)
    prompt = (
        f"Identify ALL biomedical entities in this question: '{question}'\n\n"
        f"Entity types to look for: {entity_list}\n\n"
        f"Rules:\n"
        f"- Only return entities explicitly mentioned in the question\n"
        f"- Do NOT include 'gene' unless a specific gene symbol is mentioned (e.g. MYH7)\n"
        f"- Keep the original term as written in the question\n\n"
        f"Respond with JSON only, no other text:\n"
        f"[{{\"term\": \"<term as written>\", \"entity_type\": \"<type>\"}}]"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.candidates[0].content.parts[0].text.strip()
        # strip markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        entities = json.loads(text)
        result = [(e["term"], e["entity_type"]) for e in entities
                  if e.get("entity_type") in entity_types]
        logger.info(f"[Agent 1] identified entities: {result}")
        return result
    except Exception as e:
        logger.error(f"[Agent 1] failed: {e}")
        return []


# ── Agent 2: TierSelectorAgent ─────────────────────────────────────────────────

def _select_tier(
    term: str,
    entity_type: str,
    client,
    model: str,
) -> str:
    """
    Agent 2: select grounding tier for an entity.
    Returns 'tier1', 'tier2', or 'tier3'.
    Pure LLM call, no tools.
    """
    prompt = (
        f"Select the grounding tier for this biomedical entity:\n"
        f"Term: '{term}'\n"
        f"Type: {entity_type}\n\n"
        f"Tiers:\n"
        f"tier1: common terms, abbreviations (AF, MI, HF), synonyms (heart attack), "
        f"plurals (cardiomyocytes), slight misspellings, species variants (homo sapiens). "
        f"Use when an LLM would recognise and expand the term.\n"
        f"tier2: rare or highly specific biomedical terms requiring authoritative ontology "
        f"resolution. Only for full terms, not abbreviations.\n"
        f"tier3: completely unknown, ambiguous, or severely misspelled terms.\n\n"
        f"Respond with ONLY one of: tier1, tier2, tier3"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        tier = response.candidates[0].content.parts[0].text.strip().lower()
        if tier not in ("tier1", "tier2", "tier3"):
            tier = "tier1"
        logger.info(f"[Agent 2] '{term}' ({entity_type}) -> {tier}")
        return tier
    except Exception as e:
        logger.error(f"[Agent 2] failed: {e}")
        return "tier1"


# ── Agent 3: GroundingExecutorAgent ───────────────────────────────────────────

def _execute_tier1(
    term: str,
    entity_type: str,
    driver,
    client,
    model: str,
) -> str | None:
    """
    Tier 1: LLM expands term → Neo4j name match.
    Simple LLM call (no tools) to expand, then pure Python Neo4j query.
    """
    # step 1: LLM expands the term
    prompt = (
        f"Expand this biomedical term to its full canonical form.\n"
        f"Term: '{term}'\n"
        f"Type: {entity_type}\n\n"
        f"Examples:\n"
        f"AF -> atrial fibrillation\n"
        f"MI -> myocardial infarction\n"
        f"HF -> heart failure\n"
        f"cardiomyocytes -> cardiomyocyte\n"
        f"homo sapiens -> human\n"
        f"heart attack -> myocardial infarction\n\n"
        f"Respond with ONLY the expanded term, nothing else."
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        expanded = response.candidates[0].content.parts[0].text.strip()
        logger.info(f"[Tier 1] '{term}' expanded to '{expanded}'")
    except Exception as e:
        logger.warning(f"[Tier 1] expansion failed: {e}")
        expanded = term

    # step 2: Neo4j name match
    neo4j_label = entity_type[0].upper() + entity_type[1:]
    prop_name = "symbol" if entity_type == "gene" else "name"
    try:
        results = driver.query(
            f"MATCH (n:{neo4j_label}) "
            f"WHERE toLower(n.{prop_name}) CONTAINS toLower($term) "
            f"RETURN n.{prop_name} AS value LIMIT 5",
            parameters={"term": expanded},
        )
        if results and results[0]:
            values = [r["value"] for r in results[0] if r.get("value")]
            if values:
                logger.info(f"[Tier 1] '{expanded}' -> '{values[0]}'")
                return values[0]
    except Exception as e:
        logger.warning(f"[Tier 1] Neo4j query failed: {e}")

    return None


def _execute_tier2(
    term: str,
    entity_type: str,
    driver,
    ontomcp_manager,
    find_terms_fn,
) -> str | None:
    """
    Tier 2: ontomcp → ontology ID → Neo4j match.
    Pure Python, no LLM.
    """
    shard = SHARD_MAP.get(entity_type, "")
    if not shard or ontomcp_manager is None or find_terms_fn is None:
        return None

    neo4j_label = entity_type[0].upper() + entity_type[1:]
    prop_name = "symbol" if entity_type == "gene" else "name"

    try:
        matches, _ = find_terms_fn(
            term,
            ontomcp_manager,
            ontologies=[shard],
            top_k=5,
            threshold=0.3,
            model_name="all-MiniLM-L6-v2",
        )
        for match in matches:
            ontology_id = match.term_id
            canonical_name = match.label

            # try ID match
            try:
                results = driver.query(
                    f"MATCH (n:{neo4j_label}) "
                    f"WHERE n.id CONTAINS $ontology_id "
                    f"RETURN n.{prop_name} AS value LIMIT 1",
                    parameters={"ontology_id": ontology_id},
                )
                if results and results[0]:
                    values = [r["value"] for r in results[0] if r.get("value")]
                    if values:
                        logger.info(f"[Tier 2a] '{term}' -> {ontology_id} -> '{values[0]}'")
                        return values[0]
            except Exception:
                pass

            # try canonical name match
            try:
                results = driver.query(
                    f"MATCH (n:{neo4j_label}) "
                    f"WHERE toLower(n.{prop_name}) CONTAINS toLower($name) "
                    f"RETURN n.{prop_name} AS value LIMIT 1",
                    parameters={"name": canonical_name},
                )
                if results and results[0]:
                    values = [r["value"] for r in results[0] if r.get("value")]
                    if values:
                        logger.info(f"[Tier 2b] '{term}' -> '{canonical_name}' -> '{values[0]}'")
                        return values[0]
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"[Tier 2] ontomcp failed: {e}")

    return None


def _execute_tier3(
    term: str,
    entity_type: str,
    driver,
    client,
    model: str,
) -> str | None:
    """
    Tier 3: return all Neo4j candidates → LLM picks closest.
    Pure Python to get candidates, simple LLM call to pick.
    """
    neo4j_label = entity_type[0].upper() + entity_type[1:]
    prop_name = "symbol" if entity_type == "gene" else "name"

    try:
        all_results = driver.query(
            f"MATCH (n:{neo4j_label}) RETURN n.{prop_name} AS value LIMIT 50"
        )
        if not all_results or not all_results[0]:
            return None

        all_values = [r["value"] for r in all_results[0] if r.get("value")]
        if not all_values:
            return None

        logger.info(f"[Tier 3] '{term}' -> {len(all_values)} candidates")

        # LLM picks closest match from candidates
        prompt = (
            f"Pick the closest match for '{term}' ({entity_type}) "
            f"from this list: {all_values}\n\n"
            f"Respond with ONLY the exact value from the list, nothing else.\n"
            f"If nothing is close, respond with 'not found'."
        )
        response = client.models.generate_content(model=model, contents=prompt)
        picked = response.candidates[0].content.parts[0].text.strip()
        if picked.lower() == "not found" or picked not in all_values:
            return None
        logger.info(f"[Tier 3] picked '{picked}'")
        return picked

    except Exception as e:
        logger.warning(f"[Tier 3] failed: {e}")
        return None


def _ground_entity(
    term: str,
    entity_type: str,
    tier: str,
    driver,
    ontomcp_manager,
    find_terms_fn,
    client,
    model: str,
) -> str | None:
    """
    Agent 3: execute grounding for one entity using the selected tier.
    Falls back to next tier if selected tier fails.
    """
    if tier == "tier1":
        result = _execute_tier1(term, entity_type, driver, client, model)
        if result:
            return result
        logger.info(f"[Agent 3] Tier 1 failed for '{term}', trying Tier 3")
        return _execute_tier3(term, entity_type, driver, client, model)

    elif tier == "tier2":
        result = _execute_tier2(term, entity_type, driver, ontomcp_manager, find_terms_fn)
        if result:
            return result
        logger.info(f"[Agent 3] Tier 2 failed for '{term}', trying Tier 3")
        return _execute_tier3(term, entity_type, driver, client, model)

    else:  # tier3
        return _execute_tier3(term, entity_type, driver, client, model)


# ── response utils ─────────────────────────────────────────────────────────────

def _rewrite_question(question: str, grounded: dict) -> str:
    """Append grounded entity context to the question."""
    canonical = [
        f"{entity}: {value}"
        for entity, value in grounded.items()
        if value is not None
    ]
    if not canonical:
        return question
    return (
        f"{question}\n\n"
        f"[Grounded entities: {', '.join(canonical)}. "
        f"Use these exact values when writing the Cypher query.]"
    )


# ── main entry point ───────────────────────────────────────────────────────────

def ground_entities_in_question(
    question: str,
    connection_args: dict,
    schema: dict | None = None,
    api_key: str | None = None,
    model: str = GROUNDING_MODEL,
    max_attempts: int = MAX_GROUNDING_ATTEMPTS,
) -> tuple[str, dict]:
    """
    Ground biomedical entities in a user question against the Neo4j graph.

    Uses a three-agent pipeline:
        Agent 1: identify entities (LLM, no tools)
        Agent 2: select tier per entity (LLM, no tools)
        Agent 3: execute grounding (pure Python + simple LLM calls, no AFC)

    Args:
        question:        raw user question
        connection_args: Neo4j connection dict {host, port, db_name, user, password}
        schema:          BioCypher schema dict for dynamic entity type discovery
        api_key:         Gemini API key, falls back to GOOGLE_API_KEY env var
        model:           Gemini model to use
        max_attempts:    max retry attempts if grounding fails

    Returns:
        tuple of (grounded_question, grounded_entities_dict)
        If grounding fails, returns (original_question, {})
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.warning("No Gemini API key found. Skipping grounding.")
        return question, {}

    try:
        client = genai.Client(api_key=key)
        driver = _make_driver(connection_args)
        ontomcp_manager, find_terms_fn = _make_ontomcp_manager()
    except Exception as e:
        logger.error(f"Grounding setup failed: {e}")
        return question, {}

    # get entity types from schema or use defaults
    if schema:
        entity_types = _get_searchable_entity_types(schema)
        logger.info(f"Entity types from schema: {entity_types}")
    else:
        entity_types = ["disease", "cell_type", "gene", "tissue", "species"]
        logger.warning("No schema provided, using default entity types.")

    if not entity_types:
        logger.warning("No searchable entity types found.")
        return question, {}

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            logger.info(f"Grounding retry {attempt}, waiting {RETRY_WAIT_SECONDS}s...")
            time.sleep(RETRY_WAIT_SECONDS)

        try:
            # ── Agent 1: identify entities ──────────────────────────────────
            entities = _identify_entities(question, entity_types, client, model)
            if not entities:
                logger.warning("[Agent 1] no entities found")
                continue

            # ── Agent 2: select tier per entity ─────────────────────────────
            entity_tiers = []
            for term, entity_type in entities:
                tier = _select_tier(term, entity_type, client, model)
                entity_tiers.append((term, entity_type, tier))

            # ── Agent 3: execute grounding ───────────────────────────────────
            grounded = {}
            for term, entity_type, tier in entity_tiers:
                result = _ground_entity(
                    term, entity_type, tier,
                    driver, ontomcp_manager, find_terms_fn,
                    client, model,
                )
                if result:
                    grounded[entity_type] = result

            if grounded:
                logger.info(f"Grounding succeeded: {grounded}")
                return _rewrite_question(question, grounded), grounded

        except Exception as e:
            logger.error(f"Grounding attempt {attempt} failed: {e}")

    logger.warning("Grounding failed. Returning original question.")
    return question, {}
