from unittest.mock import Mock, patch
import os

import pytest

from biochatter.prompts import BioCypherPromptEngine
from biochatter.llm_connect import GptConversation

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
        "CellType",
    ]
    assert list(prompt_engine.relationships.keys()) == [
        "PostTranslationalInteraction",
        "Phosphorylation",
        "GeneToPhenotypeAssociation",
        "GeneToDiseaseAssociation",
        "GeneExpressedInCellType",
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

    """
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        system_msg = "You have access to a knowledge graph that contains these entity types: Protein, Gene, Disease, CellType. Your task is to select the entity types that are relevant to the user's question for subsequent use in a query. Only return the entity types, comma-separated, without any additional text. Do not return entity names, relationships, or properties."
        mock_conversation.return_value.query.return_value = [
            "Gene,Disease",
            Mock(),
            None,
        ]
        mock_append_system_message = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_message
        )
        success = prompt_engine._select_entities(
            question="Which genes are associated with mucoviscidosis?",
            conversation=mock_conversation.return_value,
        )
        mock_append_system_message.assert_called_once_with((system_msg))
        assert success
        assert prompt_engine.selected_entities == ["Gene", "Disease"]


def test_relationship_selection(prompt_engine):
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Gene", "Disease"]
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        mock_conversation.return_value.query.return_value = [
            "GeneToPhenotypeAssociation",
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        success = prompt_engine._select_relationships(
            conversation=mock_conversation.return_value
        )
        assert success
        mock_append_system_messages.assert_called_once_with(
            "You have access to a knowledge graph that contains these "
            "entities: Gene, Disease. Your task is to select the relationships "
            "that are relevant to the user's question for subsequent use in a "
            "query. Only return the relationships without their sources or "
            "targets, comma-separated, and without any additional text. Here "
            "are the possible relationships and their source and target "
            'entities: [["GeneToPhenotypeAssociation", ["Disease", "Protein"]],'
            ' ["GeneToPhenotypeAssociation", ["Disease", "Gene"]]].'
        )

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
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        mock_conversation.return_value.query.return_value = [
            "GeneToDiseaseAssociation",
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        success = prompt_engine._select_relationships(
            conversation=mock_conversation.return_value
        )
        assert success
        mock_append_system_messages.assert_called_once_with(
            "You have access to a knowledge graph that contains these "
            "entities: Disease. Your task is to select the relationships that "
            "are relevant to the user's question for subsequent use in a "
            "query. Only return the relationships without their sources or "
            "targets, comma-separated, and without any additional text. Here "
            "are the possible relationships and their source and target "
            'entities: [["GeneToPhenotypeAssociation", ["Disease", "Protein"]],'
            ' ["GeneToPhenotypeAssociation", ["Disease", "Gene"]], '
            '["GeneToDiseaseAssociation", ["Protein", "Disease"]]].'
        )

        # TODO convert into benchmark to be independent of model call, mock to
        # assert the selection logic before the model call

        # technically, these are wrong selections, but the test is also arbitrarily
        # wrong; in practice, this would be bad labelling.
        assert (
            "GeneToDiseaseAssociation" in prompt_engine.selected_relationships
        )

        assert "Protein" in prompt_engine.selected_entities


def test_relationship_selection_does_not_create_none_entities(prompt_engine):
    prompt_engine.question = "Which proteins interact post-translationally?"
    prompt_engine.selected_entities = ["Protein"]
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        mock_conversation.return_value.query.return_value = [
            "PostTranslationalInteraction",
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        success = prompt_engine._select_relationships(
            conversation=mock_conversation.return_value
        )
        assert success
        assert None not in prompt_engine.selected_entities


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
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        resultMsg = """
        {
            "Disease":{
                "name":"mucoviscidosis"
            },
            "GeneToPhenotypeAssociation":{
                "score":null,
                "source":null,
                "evidence":null
            }
        }"""
        mock_conversation.return_value.query.return_value = [
            resultMsg,
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        success = prompt_engine._select_properties(
            conversation=mock_conversation.return_value
        )
        assert success
        mock_append_system_messages.assert_called_once_with(
            "You have access to a knowledge graph that contains entities and relationships. They have the following properties. Entities:{'Gene': ['id', 'name', 'taxon'], 'Disease': ['name', 'ICD10', 'DSM5']}, Relationships: {'GeneToPhenotypeAssociation': ['score', 'source', 'evidence']}. Your task is to select the properties that are relevant to the user's question for subsequent use in a query. Only return the entities and relationships with their relevant properties in JSON format, without any additional text. Return the entities/relationships as top-level dictionary keys, and their properties as dictionary values. Do not return properties that are not relevant to the question."
        )
        assert "Disease" in prompt_engine.selected_properties.keys()
        assert "name" in prompt_engine.selected_properties.get("Disease")


def test_cypher_query_generation(prompt_engine):
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
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        resultMsg = """
        MATCH (d:Disease {name: 'mucoviscidosis'})-[:PERTURBED]->(g:Gene)
        RETURN g.name AS AssociatedGenes
        """
        mock_conversation.return_value.query.return_value = [
            resultMsg,
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        query = prompt_engine._generate_query(
            question="Which genes are associated with mucoviscidosis?",
            entities=["Gene", "Disease"],
            relationships={
                "PERTURBED": {
                    "source": "Disease",
                    "target": ["Protein", "Gene"],
                }
            },
            properties={"Disease": ["name", "ICD10", "DSM5"]},
            query_language="Cypher",
            conversation=mock_conversation.return_value,
        )

        mock_append_system_messages.assert_called_once_with(
            "Generate a database query in Cypher that answers the user's "
            "question. You can use the following entities: "
            "['Gene', 'Disease'], relationships: ['PERTURBED'], and "
            "properties: {'Disease': ['name', 'ICD10', 'DSM5']}. Given the "
            "following valid combinations of source, relationship, and target: "
            "'(:Disease)-(:PERTURBED)->(:Protein)', "
            "'(:Disease)-(:PERTURBED)->(:Gene)', generate a Cypher query using "
            "one of these combinations. Only return the query, without any "
            "additional text."
        )
        assert "MATCH" in query
        assert "RETURN" in query
        assert "Gene" in query
        assert "Disease" in query
        assert "mucoviscidosis" in query
        assert (
            "-[:PERTURBED]->(g:Gene)" in query
            or "(g:Gene)<-[:PERTURBED]-" in query
        )
        assert "WHERE" in query or "{name:" in query


def test_sql_query_generation(prompt_engine):
    """
    Same as before but test the generation of an SQL query.

    Note: not as representative, because normally in a relational DB there would
    be a joining table linking genes and diseases - that info
    is not provided from the BioCypher case.
    Still keeping it for the sake of testing the method.

    The mocked output here is generated giving extra information to
    the LLM, i.e. that there is a joining table named "gene_disease"
    """
    with patch("biochatter.prompts.Conversation") as mock_conversation:
        resultMsg = """
        SELECT g.name AS AssociatedGenes
        FROM Gene_Disease gd
        JOIN Gene g ON gd.gene_id = g.id
        JOIN Disease d ON gd.disease_id = d.id
        WHERE d.name = 'mucoviscidosis';
        """
        mock_conversation.return_value.query.return_value = [
            resultMsg,
            Mock(),
            None,
        ]
        mock_append_system_messages = Mock()
        mock_conversation.return_value.append_system_message = (
            mock_append_system_messages
        )
        query = prompt_engine._generate_query(
            question="Which genes are associated with mucoviscidosis?",
            entities=["Gene", "Disease"],
            relationships={
                "PERTURBED": {
                    "source": "Disease",
                    "target": ["Protein", "Gene"],
                }
            },
            properties={"Disease": ["name", "ICD10", "DSM5"]},
            query_language="SQL",
            conversation=mock_conversation.return_value,
        )

        mock_append_system_messages.assert_called_once_with(
            "Generate a database query in SQL that answers the user's "
            "question. You can use the following entities: "
            "['Gene', 'Disease'], relationships: ['PERTURBED'], and "
            "properties: {'Disease': ['name', 'ICD10', 'DSM5']}. Given the "
            "following valid combinations of source, relationship, and target: "
            "'(:Disease)-(:PERTURBED)->(:Protein)', "
            "'(:Disease)-(:PERTURBED)->(:Gene)', generate a SQL query using "
            "one of these combinations. Only return the query, without any "
            "additional text."
        )
        assert "SELECT" in query
        assert "Gene" in query
        assert "Disease" in query
        assert "mucoviscidosis" in query
        assert "WHERE" in query or "JOIN" in query


@pytest.mark.skip(reason="temporarily skip")
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
