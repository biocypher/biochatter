from biochatter.prompts import BioCypherPromptEngine
import pytest
from .conftest import calculate_test_score


MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4",
]


@pytest.fixture(scope="module", params=MODEL_NAMES)
def prompt_engine(request):
    model_name = request.param
    return BioCypherPromptEngine(
        schema_config_or_info_path="test/test_schema_info.yaml",
        model_name=model_name,
    )


def test_entity_selection(prompt_engine):
    success = prompt_engine._select_entities(
        question="Which genes are associated with mucoviscidosis?"
    )
    assert success

    score = []
    score.append("Gene" in prompt_engine.selected_entities)
    score.append("Disease" in prompt_engine.selected_entities)

    with open("benchmark/results/biocypher_query_generation.csv", "a") as f:
        f.write(
            f"{prompt_engine.model_name},entities,{calculate_test_score(score)}\n"
        )


def test_relationship_selection(prompt_engine):
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Gene", "Disease"]
    success = prompt_engine._select_relationships()
    assert success

    score = []
    score.append(
        prompt_engine.selected_relationships == ["GeneToPhenotypeAssociation"]
    )
    score.append(
        "PERTURBED" in prompt_engine.selected_relationship_labels.keys()
    )
    score.append(
        "source" in prompt_engine.selected_relationship_labels.get("PERTURBED")
    )
    score.append(
        "target" in prompt_engine.selected_relationship_labels.get("PERTURBED")
    )
    score.append(
        "Disease"
        in prompt_engine.selected_relationship_labels.get("PERTURBED").get(
            "source"
        )
    )
    score.append(
        "Protein"
        in prompt_engine.selected_relationship_labels.get("PERTURBED").get(
            "target"
        )
    )

    with open("benchmark/results/biocypher_query_generation.csv", "a") as f:
        f.write(
            f"{prompt_engine.model_name},relationships,{calculate_test_score(score)}\n"
        )


def test_property_selection(prompt_engine):
    prompt_engine.question = "Which genes are associated with mucoviscidosis?"
    prompt_engine.selected_entities = ["Gene", "Disease"]
    prompt_engine.selected_relationships = ["GeneToPhenotypeAssociation"]
    success = prompt_engine._select_properties()
    assert success

    score = []
    score.append("Disease" in prompt_engine.selected_properties.keys())
    score.append("name" in prompt_engine.selected_properties.get("Disease"))

    with open("benchmark/results/biocypher_query_generation.csv", "a") as f:
        f.write(
            f"{prompt_engine.model_name},properties,{calculate_test_score(score)}\n"
        )


def test_query_generation(prompt_engine):
    query = prompt_engine._generate_query(
        question="Which genes are associated with mucoviscidosis?",
        entities=["Gene", "Disease"],
        relationships={
            "PERTURBED": {"source": "Disease", "target": ["Protein", "Gene"]}
        },
        properties={"Disease": ["name", "ICD10", "DSM5"]},
        query_language="Cypher",
    )

    score = []
    score.append("MATCH" in query)
    score.append("RETURN" in query)
    score.append("Gene" in query)
    score.append("Disease" in query)
    score.append("mucoviscidosis" in query)
    score.append(
        (
            "-[:PERTURBED]->(g:Gene)" in query
            or "(g:Gene)<-[:PERTURBED]-" in query
        )
    )
    score.append("WHERE" in query or "{name:" in query)

    with open("benchmark/results/biocypher_query_generation.csv", "a") as f:
        f.write(
            f"{prompt_engine.model_name},cypher query,{calculate_test_score(score)}\n"
        )


def test_end_to_end_query_generation(prompt_engine):
    query = prompt_engine.generate_query(
        question="Which genes are associated with mucoviscidosis?",
        query_language="Cypher",
    )

    score = []
    score.append("MATCH" in query)
    score.append("RETURN" in query)
    score.append("Gene" in query)
    score.append("Disease" in query)
    score.append("mucoviscidosis" in query)
    score.append(
        (
            "-[:PERTURBED]->(g:Gene)" in query
            or "(g:Gene)<-[:PERTURBED]-" in query
        )
    )
    score.append("WHERE" in query or "{name:" in query)

    with open("benchmark/results/biocypher_query_generation.csv", "a") as f:
        f.write(
            f"{prompt_engine.model_name},end-to-end,{calculate_test_score(score)}\n"
        )
