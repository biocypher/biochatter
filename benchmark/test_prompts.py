from biochatter.prompts import BioCypherPromptEngine
import pytest


MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4",
]


@pytest.fixture
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def ps(model_name):
    return BioCypherPromptEngine(
        schema_config_or_info_path="test/test_schema_info.yaml",
        model_name=model_name,
    )


def calculate_test_score(vector: list[bool]):
    score = sum(list)
    max = len(list)
    return f"{score}/{max}"


def test_entity_selection(ps):
    success = ps._select_entities(
        question="Which genes are associated with mucoviscidosis?"
    )
    assert success

    score = []
    score.append("Gene" in ps.selected_entities)
    score.append("Disease" in ps.selected_entities)

    with open("benchmark/biocypher_query_results.csv", "a") as f:
        f.write(f"{ps.model_name},entities,{calculate_test_score(score)}\n")


def test_relationship_selection(ps):
    ps.question = "Which genes are associated with mucoviscidosis?"
    ps.selected_entities = ["Gene", "Disease"]
    success = ps._select_relationships()
    assert success

    score = []
    score.append(ps.selected_relationships == ["GeneToPhenotypeAssociation"])
    score.append("PERTURBED" in ps.selected_relationship_labels.keys())
    score.append("source" in ps.selected_relationship_labels.get("PERTURBED"))
    score.append("target" in ps.selected_relationship_labels.get("PERTURBED"))
    score.append(
        "Disease"
        in ps.selected_relationship_labels.get("PERTURBED").get("source")
    )
    score.append(
        "Protein"
        in ps.selected_relationship_labels.get("PERTURBED").get("target")
    )

    with open("benchmark/biocypher_query_results.csv", "a") as f:
        f.write(
            f"{ps.model_name},relationships,{calculate_test_score(score)}\n"
        )


def test_property_selection(ps):
    ps.question = "Which genes are associated with mucoviscidosis?"
    ps.selected_entities = ["Gene", "Disease"]
    ps.selected_relationships = ["GeneToPhenotypeAssociation"]
    success = ps._select_properties()
    assert success

    score = []
    score.append("Disease" in ps.selected_properties.keys())
    score.append("name" in ps.selected_properties.get("Disease"))

    with open("benchmark/biocypher_query_results.csv", "a") as f:
        f.write(f"{ps.model_name},properties,{calculate_test_score(score)}\n")


def test_query_generation(ps):
    query = ps._generate_query(
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

    with open("benchmark/biocypher_query_results.csv", "a") as f:
        f.write(f"{ps.model_name},cypher query,{calculate_test_score(score)}\n")


def test_end_to_end_query_generation(ps):
    query = ps.generate_query(
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

    with open("benchmark/biocypher_query_results.csv", "a") as f:
        f.write(f"{ps.model_name},end-to-end,{calculate_test_score(score)}\n")
