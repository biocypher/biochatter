"""Templates for BioCypher prompts.

This module contains all the prompt templates used in the BioCypher prompt engine.
Templates are stored as format strings and can be populated using the str.format() method.
"""

ENTITY_SELECTION_PROMPT = """You have access to a knowledge graph that contains these entity types: {entities}. Your task is to select the entity types that are relevant to the user's question for subsequent use in a query. Only return the entity types, comma-separated, without any additional text. Do not return entity names, relationships, or properties."""

RELATIONSHIP_SELECTION_PROMPT = """You have access to a knowledge graph that contains these entities: {selected_entities}. Your task is to select the relationships that are relevant to the user's question for subsequent use in a query. Only return the relationships without their sources or targets, comma-separated, and without any additional text. Here are the possible relationships and their source and target entities: {relationships}."""

PROPERTY_SELECTION_PROMPT = """You have access to a knowledge graph that contains entities and relationships. They have the following properties. Entities:{entity_properties}, Relationships: {relationship_properties}. Your task is to select the properties that are relevant to the user's question for subsequent use in a query. Only return the entities and relationships with their relevant properties in compact JSON format, without any additional text. Return the entities/relationships as top-level dictionary keys, and their properties as dictionary values. Do not return properties that are not relevant to the question."""

QUERY_GENERATION_BASE = """Generate a database query in {query_language} that answers the user's question. You can use the following entities: {entities}, relationships: {relationships}, and properties: {properties}."""

QUERY_GENERATION_DIRECTIONS = """Given the following valid combinations of source, relationship, and target: {directions}, generate a {query_language} query using one of these combinations."""

QUERY_GENERATION_SUFFIX = (
    """Only return the query, without any additional text, symbols or characters --- just the query statement."""
)
