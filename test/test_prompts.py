from biochatter.prompts import BioCypherPromptEngine
import pytest

## THIS IS LARGELY BENCHMARK MATERIAL, TO BE MOCKED FOR UNIT TESTING


@pytest.fixture
def prompt_engine():
    return BioCypherPromptEngine(
        schema_config_or_info_path="test/test_schema_info.yaml"
    )


def test_biocypher_prompts(prompt_engine):
    assert list(prompt_engine.entities.keys()) == [
        "Protein",
        "Gene",
        "Disease",
    ]
    assert list(prompt_engine.relationships.keys()) == [
        "Phosphorylation",
        "GeneToPhenotypeAssociation",
        "GeneToDiseaseAssociation",
    ]

    assert "name" in prompt_engine.entities.get("Protein").get("properties")
    assert (
        prompt_engine.relationships.get("Phosphorylation").get("represented_as")
        == "edge"
    )


def test_entity_selection(prompt_engine):
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
    success = prompt_engine._select_entities(
        question="Which genes are associated with mucoviscidosis?"
    )
    assert success
    assert prompt_engine.selected_entities == ["Gene", "Disease"]


def test_relationship_selection(prompt_engine):
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Gene", "Disease"]
    success = prompt_engine._select_relationships()
    assert success

    assert prompt_engine.selected_relationships == [
        "GeneToPhenotypeAssociation"
    ]
    assert "PERTURBED" in prompt_engine.selected_relationship_labels.keys()
    assert "source" in prompt_engine.selected_relationship_labels.get(
        "PERTURBED"
    )
    assert "target" in prompt_engine.selected_relationship_labels.get(
        "PERTURBED"
    )
    assert "Disease" in prompt_engine.selected_relationship_labels.get(
        "PERTURBED"
    ).get("source")
    assert "Protein" in prompt_engine.selected_relationship_labels.get(
        "PERTURBED"
    ).get("target")


def test_relationship_selection_with_incomplete_entities(prompt_engine):
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Disease"]
    success = prompt_engine._select_relationships()
    assert success

    # TODO convert into benchmark to be independent of model call, mock to
    # assert the selection logic before the model call


def test_property_selection(prompt_engine):
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
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Gene", "Disease"]
    prompt_engine.selected_relationships = ["GeneToPhenotypeAssociation"]
    success = prompt_engine._select_properties()
    assert success
    assert "Disease" in prompt_engine.selected_properties.keys()
    assert "name" in prompt_engine.selected_properties.get("Disease")


def test_query_generation(prompt_engine):
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
    query = prompt_engine._generate_query(
        question="Which genes are associated with mucoviscidosis?",
        entities=["Gene", "Disease"],
        relationships={
            "PERTURBED": {"source": "Disease", "target": ["Protein", "Gene"]}
        },
        properties={"Disease": ["name", "ICD10", "DSM5"]},
        query_language="Cypher",
    )

    assert "MATCH" in query
    assert "RETURN" in query
    assert "Gene" in query
    assert "Disease" in query
    assert "mucoviscidosis" in query
    assert (
        "-[:PERTURBED]->(g:Gene)" in query or "(g:Gene)<-[:PERTURBED]-" in query
    )
    assert "WHERE" in query or "{name:" in query


def test_end_to_end_query_generation(prompt_engine):
    query = prompt_engine.generate_query(
        question="Which genes are associated with mucoviscidosis?",
        query_language="Cypher",
    )

    assert "MATCH" in query
    assert "RETURN" in query
    assert "Gene" in query
    assert "Disease" in query
    assert "mucoviscidosis" in query
    assert (
        "-[:PERTURBED]->(g:Gene)" in query or "(g:Gene)<-[:PERTURBED]-" in query
    )
    assert "WHERE" in query or "{name:" in query
