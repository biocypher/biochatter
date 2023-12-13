import re
from unittest.mock import patch, Mock

import pytest

from biochatter.query_interaction import BioCypherQueryHandler


_kg = {'entities': {'Protein': {'represented_as': 'node', 'is_relationship': False, 'present_in_knowledge_graph': True, 'preferred_id': 'uniprot', 'input_label': 'protein', 'db_collection_name': 'proteins', 'properties': {'name': 'str', 'score': 'float', 'taxon': 'int', 'genes': 'str[]'}}, 'Gene': {'represented_as': 'node', 'is_relationship': False, 'present_in_knowledge_graph': True, 'preferred_id': 'hgnc', 'input_label': ['hgnc', 'ensg'], 'exclude_properties': 'accession'}, 'Disease': {'represented_as': 'node', 'is_relationship': False, 'present_in_knowledge_graph': True, 'preferred_id': 'doid', 'input_label': 'Disease', 'properties': {'name': 'str', 'ICD10': 'str', 'DSM5': 'str'}}}, 'properties': {'Disease': {'name': {}, 'ICD10': {}, 'DSM5': {}}, 'Protein': {'name': {}, 'score': {}, 'taxon': {}, 'genes': {}}, 'GeneToPhenotypeAssociation': {'score': {}, 'source': {}, 'evidence': {}}}, 'relationships': {'Phosphorylation': {'is_a': 'post translational interaction', 'is_relationship': True, 'present_in_knowledge_graph': True, 'represented_as': 'edge', 'input_label': 'phosphorylation', 'source': 'Protein', 'target': 'Protein'}, 'GeneToPhenotypeAssociation': {'represented_as': 'edge', 'is_relationship': True, 'present_in_knowledge_graph': True, 'label_as_edge': 'PERTURBED', 'input_label': ['protein_disease', 'gene_disease'], 'source': 'Disease', 'target': ['Protein', 'Gene'], 'exclude_properties': 'accession', 'properties': {'score': 'float', 'source': 'str', 'evidence': 'str'}}, 'GeneToDiseaseAssociation': {'represented_as': 'edge', 'is_relationship': True, 'present_in_knowledge_graph': True, 'label_as_edge': 'PERTURBED_IN', 'input_label': 'gene_phenotype', 'source': 'Protein', 'target': 'Disease', 'properties': {'score': 'float', 'source': 'str', 'evidence': 'str'}}}}
_kg_selected = {'entities': ['Gene', 'Disease', 'Protein'], 'properties': {'Disease': {'name': {}, 'ICD10': {}, 'DSM5': {}}, 'Protein': {'name': {}, 'score': {}, 'taxon': {}, 'genes': {}}, 'GeneToPhenotypeAssociation': {'score': {}, 'source': {}, 'evidence': {}}}, 'relationships': {'PERTURBED': {'source': 'Disease', 'target': ['Protein', 'Gene']}}}

@pytest.fixture
# TODO merge the 2 possible mock queries; there is probably some pytest functionality to do it nicely
def query_handler():
    query = "MATCH (d:Disease {name: 'mucoviscidosis'})-[:PERTURBED]->(g:Gene)\nRETURN g.name AS AssociatedGenes"
    query_lang = "Cypher"
    question = "Which genes are associated with mucoviscidosis?"
    return BioCypherQueryHandler(query, query_lang, _kg, _kg_selected, question)


@pytest.fixture
def query_with_limit_handler():
    query = "MATCH (d:Disease {name: 'mucoviscidosis'})-[:PERTURBED]->(g:Gene)\n" \
            "RETURN g.name AS AssociatedGenes LIMIT 25"
    query_lang = "Cypher"
    question = "Which genes are associated with mucoviscidosis?"
    return BioCypherQueryHandler(query, query_lang, _kg, _kg_selected, question)


def test_explain(query_handler):
    with patch("biochatter.query_interaction.GptConversation") as mock_gptconv:
        resultMsg = "**Some explanation**"

        mock_gptconv.return_value.query.return_value = [resultMsg, Mock(), None]
        mock_append_system_messages = Mock()
        mock_gptconv.return_value.append_system_message = (
            mock_append_system_messages
        )
        explanation = query_handler.explain_query()

        mock_append_system_messages.assert_called_once_with(
            f"You are an expert in {query_handler.query_lang} and will assist in explaining a query.\n"
            f"The query answers the following user question: '{query_handler.question}'."
            "It will be used to query a knowledge graph that contains (among others)"
            f" the following entities: {query_handler.kg_selected['entities']}, "
            f"relationships: {list(query_handler.kg_selected['relationships'].keys())}, and "
            f"properties: {query_handler.kg_selected['properties']}. "
            "Only return the explanation, without any additional text."
        )
    # TODO what other checks can we do? The explanation text is not predictable
    #  (except if we would add desired formatting in the future)


# for updating; assert that it returns largely the same?
def test_add_limit(query_handler):
    update_request = "Return maximum 10 results"
    updated_query = update_limit(query_handler, update_request)
    assert query_handler.query in updated_query
    assert "LIMIT 10" in updated_query


def test_update_limit(query_with_limit_handler):
    update_request = "Return maximum 10 results"
    updated_query = update_limit(query_with_limit_handler, update_request)
    assert updated_query == re.sub(r"LIMIT [\d]+", "LIMIT 10", query_with_limit_handler.query)


def update_limit(query_handler, update_request):
    with patch("biochatter.query_interaction.GptConversation") as mock_gptconv:
        resultMsg = """
        MATCH (d:Disease {name: 'mucoviscidosis'})-[:PERTURBED]->(g:Gene)\nRETURN g.name AS AssociatedGenes LIMIT 10
        """
        mock_gptconv.return_value.query.return_value = [resultMsg, Mock(), None]
        mock_append_system_messages = Mock()
        mock_gptconv.return_value.append_system_message = (
            mock_append_system_messages
        )
        updated_query = query_handler.update_query(update_request)

        mock_append_system_messages.assert_called_once_with(
            f"You are an expert in {query_handler.query_lang} and will assist in updating a query.\n"
            f"The original query answers the following user question: '{query_handler.question}'."
            f"This is the original query: '{query_handler.query}'."
            f"It will be used to query a knowledge graph that has the following entities: "
            f"{query_handler.kg['entities']}, relationships: {list(query_handler.kg['relationships'].keys())}, and "
            f"properties: {query_handler.kg['properties']}. "
            "Update the query to reflect the user's request." 
            "Only return the updated query, without any additional text."
        )
    return updated_query.strip()


def test_return_all_properties():
    assert True


def test_return_other_property():
    assert True


def test_update_sorting():
    assert True


def test_update_filtering():
    assert True

