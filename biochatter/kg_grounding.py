"""
kg_grounding.py
---------------
Entity and property value grounding for BioChatter KG queries.

Terms come directly from BioCypherPromptEngine._select_entities() and
_select_properties() (extended to return term+type / term+property pairs
in the same LLM call they already make — no separate identification agent,
no redundant LLM call to redo work BioCypherPromptEngine already did).

Entity grounding tier ladder, ordered by cost:
    Tier 0   — exact match against KG values (free, no LLM)
    Tier 0.5 — fuzzy match against KG values (free, no LLM)
    Tier 1   — LLM expansion, then exact/fuzzy match again (one LLM call)
    Tier 2   — ontomcp -> ontology ID -> KG ID match (no LLM)
    Tier 3   — all KG candidates -> LLM picks closest (last resort, one LLM call)

Tier selection is schema-driven, never based on entity type names:
    - entity types with no searchable string property go straight to Tier 2
    - entity types with a searchable string property try Tier 0 -> 0.5 -> 1 -> 2 -> 3

Property value grounding (separate, shorter ladder):
    Tier 0   — exact substring match against allowed values (free)
    Tier 0.5 — fuzzy match against allowed values (free)
    Tier 1   — LLM maps the question to one of the allowed values (one LLM call)
    Only attempted for properties whose schema type is "str" and whose
    distinct value count is small enough to be a controlled vocabulary
    (not a continuous numeric property).

All KG access goes through a KGAdapter, so grounding logic never contains
DBMS-specific query syntax. All LLM access goes through the same
conversation_factory BioCypherPromptEngine uses, so grounding is not tied
to any specific model provider.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 85  # rapidfuzz ratio, 0-100
MAX_CATEGORICAL_VALUES = 20
ONTOMCP_SHARD_DIR = os.getenv("ONTOMCP_SHARD_DIR")


# ── KG adapter interface ───────────────────────────────────────────────────────

class KGAdapter:
    """Abstract interface for KG access. One implementation per DBMS."""

    def get_all_values(self, entity_type: str, search_property: str) -> list[str]:
        """Return all values of search_property for nodes of entity_type."""
        raise NotImplementedError

    def get_node_value_by_id_substring(
        self, entity_type: str, search_property: str, ontology_id: str,
    ) -> str | None:
        """Return search_property value of a node whose id contains ontology_id."""
        raise NotImplementedError

    def get_distinct_property_values(self, entity_type: str, prop_name: str) -> list:
        """Return distinct values of a property across nodes of entity_type."""
        raise NotImplementedError


class Neo4jAdapter(KGAdapter):
    """KGAdapter implementation for Neo4j via neo4j_utils.Driver."""

    def __init__(self, driver):
        self.driver = driver

    def _label(self, entity_type: str) -> str:
        # PascalCase, matching BioCypherPromptEngine.entities keys
        return entity_type[0].upper() + entity_type[1:] if entity_type else entity_type

    def get_all_values(self, entity_type: str, search_property: str) -> list[str]:
        if not search_property:
            return []
        label = self._label(entity_type)
        try:
            results = self.driver.query(
                f"MATCH (n:{label}) WHERE n.{search_property} IS NOT NULL "
                f"RETURN n.{search_property} AS value"
            )
            if not results or not results[0]:
                return []
            return [r["value"] for r in results[0] if r.get("value") is not None]
        except Exception as e:
            logger.warning(f"[Neo4jAdapter] get_all_values failed for {entity_type}.{search_property}: {e}")
            return []

    def get_node_value_by_id_substring(
        self, entity_type: str, search_property: str, ontology_id: str,
    ) -> str | None:
        label = self._label(entity_type)
        try:
            if search_property:
                results = self.driver.query(
                    f"MATCH (n:{label}) WHERE toLower(n.id) CONTAINS toLower($ontology_id) "
                    f"RETURN n.{search_property} AS value LIMIT 1",
                    parameters={"ontology_id": ontology_id},
                )
            else:
                results = self.driver.query(
                    f"MATCH (n:{label}) WHERE toLower(n.id) CONTAINS toLower($ontology_id) "
                    f"RETURN n.id AS value LIMIT 1",
                    parameters={"ontology_id": ontology_id},
                )
            if not results or not results[0]:
                return None
            value = results[0][0].get("value")
            return value
        except Exception as e:
            logger.warning(f"[Neo4jAdapter] get_node_value_by_id_substring failed for {entity_type}: {e}")
            return None

    def get_distinct_property_values(self, entity_type: str, prop_name: str) -> list:
        label = self._label(entity_type)
        try:
            results = self.driver.query(
                f"MATCH (n:{label}) WHERE n.{prop_name} IS NOT NULL "
                f"RETURN DISTINCT n.{prop_name} AS value"
            )
            if not results or not results[0]:
                return []
            return [r["value"] for r in results[0] if r.get("value") is not None]
        except Exception as e:
            logger.warning(f"[Neo4jAdapter] get_distinct_property_values failed for {entity_type}.{prop_name}: {e}")
            return []


def _make_neo4j_driver(connection_args: dict):
    """Create a neo4j_utils Driver from connection args dict."""
    import neo4j_utils as nu

    host = connection_args.get("host", "localhost")
    port = connection_args.get("port", "7687")
    db_uri = host if host.startswith(("bolt://", "neo4j://")) else f"bolt://{host}:{port}"
    user = connection_args.get("user") or None
    password = connection_args.get("password") or None
    return nu.Driver(
        db_name=connection_args.get("db_name") or "neo4j",
        db_uri=db_uri,
        db_user=user,
        db_passwd=password,
    )


def make_adapter(connection_args: dict, dbms: str = "neo4j") -> KGAdapter:
    """Factory for KGAdapter implementations. Extend with more DBMS as needed."""
    if dbms == "neo4j":
        driver = _make_neo4j_driver(connection_args)
        return Neo4jAdapter(driver)
    raise NotImplementedError(f"No KGAdapter implementation for dbms='{dbms}'")


# ── ontomcp setup ──────────────────────────────────────────────────────────────

def _make_ontomcp_manager():
    """Initialise the ontomcp shard manager. Returns (manager, find_terms_fn) or (None, None)."""
    if not ONTOMCP_SHARD_DIR:
        logger.warning("ONTOMCP_SHARD_DIR not set. Tier 2 grounding will be skipped.")
        return None, None

    try:
        from ontomcp.retrieval.searcher import find_terms as _find_terms
        from ontomcp.retrieval.shard_manager import ShardManager

        if not os.path.exists(ONTOMCP_SHARD_DIR):
            logger.warning(f"ontomcp shard directory not found: {ONTOMCP_SHARD_DIR}. Tier 2 will be skipped.")
            return None, None

        manager = ShardManager(ONTOMCP_SHARD_DIR)
        if not manager.list_ids():
            logger.warning("No ontomcp shards found. Tier 2 will be skipped.")
            return None, None

        return manager, _find_terms
    except ImportError:
        logger.warning("ontomcp not installed. Tier 2 grounding will be skipped.")
        return None, None


# ── schema-driven config (no entity-type-name hardcoding) ────────────────────

def get_entity_grounding_config(entity_type: str, schema: dict) -> dict:
    """
    Derive grounding strategy for an entity type purely from structural
    schema fields (properties, preferred_id) that exist for any entity
    type, regardless of its name. Never branches on entity_type's name.
    """
    config = schema.get(entity_type, {})
    properties = config.get("properties", {}) or {}
    preferred_id = config.get("preferred_id")

    string_properties = [k for k, v in properties.items() if v == "str"]

    return {
        "search_property": string_properties[0] if string_properties else None,
        "ontology": preferred_id,
        "has_searchable_name": len(string_properties) > 0,
        "has_ontology": preferred_id is not None,
    }


def get_property_grounding_config(entity_type: str, prop_name: str, schema: dict) -> dict:
    """Derive whether a property is in scope for value grounding, from its declared schema type."""
    prop_type = (schema.get(entity_type, {}).get("properties", {}) or {}).get(prop_name)
    return {
        "is_categorical_candidate": prop_type == "str",
    }


# ── generic matching helpers (no LLM) ──────────────────────────────────────────

def _exact_match(term: str, candidates: list) -> str | None:
    """Case-insensitive exact match of term against a list of candidate values."""
    term_l = term.lower()
    for c in candidates:
        if str(c).lower() == term_l:
            return c
    return None


def _fuzzy_best_match(term: str, candidates: list) -> tuple[str | None, float]:
    """Return (best_candidate, score 0-100) for term against a list of candidates."""
    if not candidates:
        return None, 0.0
    best = None
    best_score = 0.0
    term_l = term.lower()
    for c in candidates:
        score = fuzz.ratio(term_l, str(c).lower())
        if score > best_score:
            best_score = score
            best = c
    return best, best_score


# ── LLM helpers (provider-agnostic via conversation_factory) ─────────────────

def _llm_expand_term(
    term: str,
    entity_type: str,
    candidate_values: list,
    conversation_factory: Callable,
) -> str:
    """
    Tier 1 — LLM expands a term (abbreviation, synonym, shorthand) to the
    biomedical form most likely to match one of the actual candidate
    values held in the KG for this entity type. The candidate values are
    given explicitly so the LLM expands toward what the graph actually
    contains, not toward an arbitrary textbook canonical form.
    """
    conversation = conversation_factory()

    sample = candidate_values[:20] if candidate_values else []

    conversation.append_system_message(
        f"You have access to a knowledge graph that contains entities of "
        f"type {entity_type}. Some example values stored for this entity "
        f"type in the graph are: {sample}. A user has referred to one of "
        "these entities using an abbreviation, synonym, or shorthand term "
        "that may not exactly match a value in the graph. Your task is to "
        "expand the term to the full biomedical form most likely to "
        "correspond to a value in the graph, following the style of the "
        "example values shown. Only return the expanded term, "
        "comma-separated alternatives are not allowed. Do not return the "
        "original term unchanged, an explanation, or any value not "
        "plausible for this entity type.",
    )

    try:
        msg, token_usage, correction = conversation.query(term)
    except Exception as e:
        logger.warning(f"[Tier 1] LLM expansion failed for '{term}' ({entity_type}): {e}")
        return term

    return msg.strip() if msg else term


def _llm_pick_closest(
    term: str,
    entity_type: str,
    candidates: list,
    conversation_factory: Callable,
) -> str | None:
    """
    Tier 3 — last resort. LLM is given every candidate value the KG
    actually holds for this entity type and must pick exactly one of
    them, or explicitly decline.
    """
    conversation = conversation_factory()

    conversation.append_system_message(
        f"You have access to a knowledge graph that contains exactly "
        f"these {len(candidates)} values for entities of type "
        f"{entity_type}: {candidates}. A user has referred to one of "
        "these entities using a term that does not exactly or closely "
        "match any value in this list. Your task is to select the "
        "single value from the list that most plausibly corresponds to "
        "the user's term. Only return that exact value, copied "
        "character-for-character from the list. Do not return a value "
        "that is not in the list. If no value in the list is a "
        "reasonable match, return 'not found' instead.",
    )

    try:
        msg, token_usage, correction = conversation.query(term)
    except Exception as e:
        logger.warning(f"[Tier 3] LLM pick failed for '{term}' ({entity_type}): {e}")
        return None

    picked = msg.strip() if msg else ""
    if picked.lower() == "not found" or picked not in candidates:
        return None
    return picked


def _llm_map_property_value(
    question: str,
    prop_name: str,
    entity_or_relationship: str,
    allowed_values: list,
    conversation_factory: Callable,
) -> str | None:
    """
    Tier 1 for property values. The LLM is given the exact closed set of
    values the property can take in the KG and must map the user's
    informal wording in the question onto exactly one of them, or
    explicitly state that the question does not filter on this property.
    """
    conversation = conversation_factory()

    conversation.append_system_message(
        f"You have access to a knowledge graph in which the property "
        f"'{prop_name}' of {entity_or_relationship} only ever takes one "
        f"of these {len(allowed_values)} values: {allowed_values}. The "
        f"user's question may refer to '{prop_name}' using different "
        "wording than these exact values. Your task is to determine "
        f"which of these values, if any, the question is referring to "
        f"for '{prop_name}', for subsequent use as a filter in a query. "
        "Only return that exact value, copied character-for-character "
        "from the list. Do not return a value that is not in the list. "
        f"If the question does not filter on '{prop_name}' at all, "
        "return 'none' instead.",
    )

    try:
        msg, token_usage, correction = conversation.query(question)
    except Exception as e:
        logger.warning(f"[Property Tier 1] LLM mapping failed for '{prop_name}': {e}")
        return None

    picked = msg.strip() if msg else "none"
    if picked.lower() == "none" or picked not in [str(v) for v in allowed_values]:
        return None
    return picked


# ── ontomcp lookup (Tier 2, no LLM) ────────────────────────────────────────────

def _ontomcp_lookup(
    term: str,
    ontology: str,
    ontomcp_manager,
    find_terms_fn,
    top_k: int = 10,
) -> list[tuple[str, str]]:
    """
    Return a ranked list of (ontology_id, canonical_label) candidates for
    the term. The caller is responsible for trying each candidate against
    the KG in order, since the embedding-search top match is not always
    the entity actually present in any given graph (e.g. a more specific
    subtype may rank above the general disease entry that the KG holds).
    """
    if not ontology or ontomcp_manager is None or find_terms_fn is None:
        return []
    try:
        matches, _ = find_terms_fn(
            term, ontomcp_manager, ontologies=[ontology],
            top_k=top_k, threshold=0.3, model_name="all-MiniLM-L6-v2",
        )
        return [(m.term_id, m.label) for m in matches]
    except Exception as e:
        logger.warning(f"[Tier 2] ontomcp lookup failed for '{term}' (ontology={ontology}): {e}")
        return []


# ── entity grounding: tier ladder ──────────────────────────────────────────────

def _ground_one_entity(
    term: str,
    entity_type: str,
    schema: dict,
    adapter: KGAdapter,
    ontomcp_manager,
    find_terms_fn,
    conversation_factory: Callable,
) -> str | None:
    """
    Run the full cost-ordered tier ladder for one entity term.
    Schema-driven throughout: never branches on entity_type's literal
    name, only on its structural config (searchable name? has ontology?).
    """
    config = get_entity_grounding_config(entity_type, schema)

    # entity type has no string property at all (e.g. Biotope's Disease,
    # which only stores an ontology id) -> name-based tiers are not
    # possible, only the ontology path can work
    if not config["has_searchable_name"]:
        if config["has_ontology"]:
            # ontomcp's embedding search fails on raw abbreviations and
            # shorthand (empirically confirmed: "AF" matches unrelated
            # terms at very low scores). Expand the term to its likely
            # scientific/canonical form first, then look that up — the
            # expansion is what makes the ontology search work at all.
            expanded = _llm_expand_term(term, entity_type, [], conversation_factory)

            candidates = _ontomcp_lookup(expanded, config["ontology"], ontomcp_manager, find_terms_fn)
            if not candidates and expanded.lower() != term.lower():
                # expansion may occasionally hurt rather than help —
                # fall back to the raw term as a second attempt
                candidates = _ontomcp_lookup(term, config["ontology"], ontomcp_manager, find_terms_fn)

            # the embedding search's top-ranked candidate is not always
            # the entity actually present in this particular KG (e.g. a
            # more specific subtype can rank above the general entry the
            # graph holds) — try each ranked candidate in order until one
            # is found in the KG, rather than only trying rank 1
            for rank, (ontology_id, label) in enumerate(candidates, start=1):
                value = adapter.get_node_value_by_id_substring(entity_type, None, ontology_id)
                if value:
                    logger.info(
                        f"[Tier 2] '{term}' -> '{expanded}' -> {ontology_id} -> '{value}' "
                        f"(no searchable name property on {entity_type}; "
                        f"rank {rank} of {len(candidates)} ontomcp matches)"
                    )
                    return value
        logger.info(f"[no match] '{term}' ({entity_type}) — no searchable name property and ontology lookup failed")
        return None

    search_property = config["search_property"]
    all_values = adapter.get_all_values(entity_type, search_property)

    # Tier 0 — exact match
    match = _exact_match(term, all_values)
    if match:
        logger.info(f"[Tier 0] '{term}' -> '{match}' (exact match, {entity_type}.{search_property})")
        return match

    # Tier 0.5 — fuzzy match
    match, score = _fuzzy_best_match(term, all_values)
    if match and score >= FUZZY_THRESHOLD:
        logger.info(f"[Tier 0.5] '{term}' -> '{match}' (fuzzy, score={score:.0f}, {entity_type}.{search_property})")
        return match

    # Tier 1 — LLM expansion grounded in real candidate values, then match again
    expanded = _llm_expand_term(term, entity_type, all_values, conversation_factory)
    if expanded.lower() != term.lower():
        match = _exact_match(expanded, all_values)
        if match:
            logger.info(f"[Tier 1] '{term}' -> '{expanded}' -> '{match}' (exact match after expansion)")
            return match
        match, score = _fuzzy_best_match(expanded, all_values)
        if match and score >= FUZZY_THRESHOLD:
            logger.info(f"[Tier 1] '{term}' -> '{expanded}' -> '{match}' (fuzzy after expansion, score={score:.0f})")
            return match

    # Tier 2 — ontomcp, only if this entity type has a usable ontology.
    # Try ranked candidates in order, not just the top match — the same
    # reasoning as the no-searchable-name branch above.
    if config["has_ontology"]:
        candidates = _ontomcp_lookup(term, config["ontology"], ontomcp_manager, find_terms_fn)
        for ontology_id, canonical_label in candidates:
            match = _exact_match(canonical_label, all_values)
            if not match:
                match, score = _fuzzy_best_match(canonical_label, all_values)
                match = match if score >= FUZZY_THRESHOLD else None
            if match:
                logger.info(f"[Tier 2] '{term}' -> ontomcp '{canonical_label}' ({ontology_id}) -> '{match}'")
                return match

    # Tier 3 — last resort, LLM sees every candidate and picks
    if all_values:
        picked = _llm_pick_closest(term, entity_type, all_values, conversation_factory)
        if picked:
            logger.info(f"[Tier 3] '{term}' -> '{picked}' (LLM picked from {len(all_values)} candidates)")
            return picked

    logger.info(f"[no match] '{term}' ({entity_type}) — exhausted all tiers")
    return None


# ── entity grounding: entry point ──────────────────────────────────────────────

def ground_entities(
    question: str,
    selected_entity_terms: list[tuple[str, str]],
    connection_args: dict,
    schema: dict,
    conversation_factory: Callable,
    dbms: str = "neo4j",
) -> tuple[str, dict]:
    """
    Ground entity terms in the question against the KG.

    Args:
        question:              raw (or partially grounded) user question
        selected_entity_terms: [(term, entity_type), ...] captured by
                               BioCypherPromptEngine._select_entities()
                               in the same LLM call that selects types —
                               no separate term-identification call here
        connection_args:       KG connection dict
        schema:                BioCypherPromptEngine.entities (parsed schema)
        conversation_factory:  same factory BioCypherPromptEngine uses,
                               so grounding works with any LLM provider
        dbms:                  which KGAdapter implementation to use

    Returns:
        (grounded_question, grounded_entities_dict)
    """
    if not selected_entity_terms:
        logger.warning("No entity terms provided. Skipping entity grounding.")
        return question, {}

    try:
        adapter = make_adapter(connection_args, dbms=dbms)
    except Exception as e:
        logger.error(f"Entity grounding adapter setup failed: {e}")
        return question, {}

    ontomcp_manager, find_terms_fn = _make_ontomcp_manager()

    grounded = {}
    for term, entity_type in selected_entity_terms:
        result = _ground_one_entity(
            term, entity_type, schema, adapter,
            ontomcp_manager, find_terms_fn, conversation_factory,
        )
        if result:
            grounded[term] = result

    grounded_question = _rewrite_question(question, grounded)
    return grounded_question, grounded


# ── property value grounding: tier ladder ──────────────────────────────────────

def _ground_one_property_value(
    question: str,
    prop_name: str,
    entity_or_relationship: str,
    allowed_values: list,
    conversation_factory: Callable,
) -> str | None:
    """Cost-ordered tier ladder for one property's value, given the user's question."""
    q_lower = question.lower()

    # Tier 0 — does any allowed value already appear verbatim in the question?
    for value in allowed_values:
        if str(value).lower() in q_lower:
            logger.info(f"[Property Tier 0] '{prop_name}' -> '{value}' (exact substring match in question)")
            return value

    # Tier 0.5 — fuzzy match each word in the question against allowed values
    best_overall, best_score = None, 0.0
    for word in q_lower.split():
        match, score = _fuzzy_best_match(word, [str(v) for v in allowed_values])
        if score > best_score:
            best_overall, best_score = match, score
    if best_overall and best_score >= FUZZY_THRESHOLD:
        logger.info(f"[Property Tier 0.5] '{prop_name}' -> '{best_overall}' (fuzzy, score={best_score:.0f})")
        return best_overall

    # Tier 1 — LLM semantic mapping against the closed set of allowed values
    canonical = _llm_map_property_value(
        question, prop_name, entity_or_relationship, allowed_values, conversation_factory,
    )
    if canonical:
        logger.info(f"[Property Tier 1] '{prop_name}' -> '{canonical}' (LLM semantic match)")
    return canonical


def ground_property_values(
    question: str,
    selected_properties: dict,
    connection_args: dict,
    schema: dict,
    conversation_factory: Callable,
    dbms: str = "neo4j",
    max_categorical_values: int = MAX_CATEGORICAL_VALUES,
) -> str:
    """
    Ground property filter terms in the question against KG property values.

    Only attempted for properties whose schema type is "str" — bool, int,
    and float properties are skipped via schema check, not by hardcoding
    property names. Properties with too many distinct values to be a
    meaningful controlled vocabulary (continuous data) are also skipped.

    Args:
        question:              grounded question (after entity grounding)
        selected_properties:   from BioCypherPromptEngine._select_properties()
                               e.g. {"De_result": ["direction"]}
        connection_args:       KG connection dict
        schema:                BioCypherPromptEngine.entities/relationships
        conversation_factory:  same factory BioCypherPromptEngine uses
        dbms:                  which KGAdapter implementation to use
        max_categorical_values: skip properties with more distinct values than this

    Returns:
        question with property filter terms replaced by canonical KG values
    """
    try:
        adapter = make_adapter(connection_args, dbms=dbms)
    except Exception as e:
        logger.error(f"Property grounding adapter setup failed: {e}")
        return question

    substitutions = {}

    for entity_or_relationship, props in selected_properties.items():
        for prop_name in props:
            type_config = get_property_grounding_config(entity_or_relationship, prop_name, schema)
            if not type_config["is_categorical_candidate"]:
                logger.info(f"Skipping '{prop_name}' — not a str-typed property (schema-driven skip)")
                continue

            allowed_values = adapter.get_distinct_property_values(entity_or_relationship, prop_name)
            if not allowed_values:
                continue
            if len(allowed_values) > max_categorical_values:
                logger.info(f"Skipping '{prop_name}' — {len(allowed_values)} distinct values (continuous)")
                continue

            canonical = _ground_one_property_value(
                question, prop_name, entity_or_relationship, allowed_values, conversation_factory,
            )
            if canonical:
                substitutions[prop_name] = canonical

    return _rewrite_question_with_properties(question, substitutions)


# ── question rewriting ─────────────────────────────────────────────────────────

def _rewrite_question(question: str, grounded: dict) -> str:
    """Replace original terms in question with canonical KG values."""
    rewritten = question
    for original_term, canonical_value in grounded.items():
        if canonical_value is not None:
            rewritten = rewritten.replace(original_term, str(canonical_value))
    return rewritten


def _rewrite_question_with_properties(question: str, substitutions: dict) -> str:
    """Append grounded property value context to the question."""
    if not substitutions:
        return question
    canonical = [f"{prop}: {value}" for prop, value in substitutions.items()]
    return (
        f"{question}\n\n"
        f"[Grounded property values: {', '.join(canonical)}. "
        f"Use these exact values in the query WHERE clause.]"
    )
