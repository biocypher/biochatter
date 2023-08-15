from biochatter.prompts import BioCypherPrompt
import pytest


@pytest.fixture
def ps():
    return BioCypherPrompt(schema_config_path="test/test_schema_config.yaml")


def test_biocypher_prompts(ps):
    assert list(ps.entities.keys()) == [
        "Protein",
        "Pathway",
        "Gene",
        "Disease",
    ]
    assert list(ps.relationships.keys()) == [
        "PostTranslationalInteraction",
        "Phosphorylation",
        "GeneToDiseaseAssociation",
    ]

    assert "name" in ps.entities.get("Protein").get("properties")
    assert (
        ps.relationships.get("Phosphorylation").get("represented_as") == "edge"
    )


def test_entity_selection(ps):
    """

    The first step is to identify the KG components that are relevant to the
    question. Requires only a question as input, entities and relationships are
    inferred from the schema.

    "You have access to a knowledge graph that contains {entities} and
    {relationships}. Your task is to select the ones that are relevant to
    the user's question for subsequent use in a query. Only return the
    entities and relationships, comma-separated, without any additional
    text."

    TODO: a couple more representative cases

    """
    success = ps.select_entities(
        question="Which genes are associated with mucoviscidosis?"
    )
    assert success
    assert ps.selected_entities == ["Gene", "Disease"]
    assert ps.selected_relationships == ["GeneToDiseaseAssociation"]
    assert ps.selected_relationship_labels == ["PERTURBED_IN_DISEASE"]


def test_property_selection(ps):
    """

    The second step is to identify the properties of the relevant KG components
    to enable constraining the query. Requires the pre-selected entities and
    relationships as input. The question is reused from the entity selection
    (but can optionally be provided), and the properties are inferred from the
    schema.

    "You have access to a knowledge graph that contains {entities} and
    {relationships}. They have the following properties: {property dictionary}.
    Your task is to select the properties that are relevant to the user's
    question for subsequent use in a query. Only return the entities and
    relationships and relevant properties in dictionary format, similar to the
    example above, without any additional text."

    """
    success = ps.select_properties(
        question="Which genes are associated with mucoviscidosis?",
        entities=["Gene", "Disease"],
        relationships=["GeneToDiseaseAssociation"],
    )
    assert success
    assert "Disease" in ps.selected_properties.keys()
    assert "name" in ps.selected_properties.get("Disease")


def test_query_generation(ps):
    """

    The third step is to generate a query that will retrieve the information
    that is relevant to the user's question from the KG. Requires a question,
    the pre-selected entities and relationships, and the pre-selected properties
    as input. The query language can be inferred from the dedicated extended
    biocypher schema info file, but also given as a parameter in case only the
    schema configuration is available.

    "You have access to a knowledge graph that contains {entities} and
    {relationships} in a {database language} database. Your task is to generate
    a query that will retrieve the information that is relevant to the user's
    question. Only return the query, without any additional text."

    TODO: need to know verbose name of disease. should be an additional mapping
    step to get the disease name used in the database. should probably be
    non-LLM-based, since the list of possible matches could be very long. Use a
    combination of GILDA and fuzzy search? dedicated synonym KG?

    TODO: special case relationship as node

    """
    assert ps.generate_query(
        question="Which genes are associated with mucoviscidosis?",
        entities=["Gene", "Disease"],
        relationships=["PERTURBED_IN_DISEASE"],
        properties={"Disease": ["name"]},
        database_language="Cypher",
    ) == (
        "MATCH (n:Gene)-[:PERTURBED_IN_DISEASE]->(m:Disease) "
        "WHERE m.name = 'mucoviscidosis' RETURN n"
    )
