import pytest

from biochatter.prompts import BioCypherPromptEngine
from .conftest import calculate_test_score, RESULT_FILES

FILE_PATH = next(
    (
        s
        for s in RESULT_FILES
        if "biocypher" in s and "query" in s and "generation" in s
    ),
    None,
)

MODEL_NAMES = [
    "gpt-3.5-turbo",
    # "gpt-4",
]


@pytest.fixture(scope="module", params=MODEL_NAMES)
def prompt_engine(request):
    def setup_prompt_engine(kg_schema_path):
        model_name = request.param
        return BioCypherPromptEngine(
            schema_config_or_info_path=kg_schema_path,
            model_name=model_name,
        )

    return setup_prompt_engine


def test_entity_selection(prompt_engine, test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    prompt_engine = prompt_engine(kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}")

    success = prompt_engine._select_entities(
        question=prompt
    )
    assert success

    score = []
    for expected_entity in expected_entities:
        score.append(expected_entity in prompt_engine.selected_entities)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},entities,{calculate_test_score(score)}\n"
        )


def test_relationship_selection(prompt_engine, test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    expected_relationships = test_data_biocypher_query_generation[3]
    expected_relationship_labels = test_data_biocypher_query_generation[4]

    prompt_engine = prompt_engine(kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}")

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    success = prompt_engine._select_relationships()
    assert success

    score = []
    for expected_relationship in expected_relationships:
        score.append(expected_relationship in prompt_engine.selected_relationships)

    for expected_relationship_label_key in expected_relationship_labels.keys():
        score.append(expected_relationship_label_key in prompt_engine.selected_relationship_labels.keys())

        for expected_relationship_label_value in expected_relationship_labels[expected_relationship_label_key]:
            score.append(expected_relationship_label_value in prompt_engine.selected_relationship_labels[
                expected_relationship_label_key])

    # TODO: make it more generic to be able to compare arbitrarily nested structures
    # score.append(
    #    "Disease"
    #    in prompt_engine.selected_relationship_labels.get("PERTURBED").get(
    #        "source"
    #    )
    # )
    # score.append(
    #    "Protein"
    #    in prompt_engine.selected_relationship_labels.get("PERTURBED").get(
    #        "target"
    #    )
    # )

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},relationships,{calculate_test_score(score)}\n"
        )


def test_property_selection(prompt_engine, test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    expected_relationships = test_data_biocypher_query_generation[3]
    expected_properties = test_data_biocypher_query_generation[5]

    prompt_engine = prompt_engine(kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}")
    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    prompt_engine.selected_relationships = expected_relationships
    success = prompt_engine._select_properties()
    assert success

    score = []
    for expected_property_key in expected_properties.keys():
        score.append(expected_property_key in prompt_engine.selected_properties.keys())

        for expected_property_value in expected_properties[expected_property_key]:
            score.append(expected_property_value in prompt_engine.selected_properties[expected_property_key])

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},properties,{calculate_test_score(score)}\n"
        )


def test_query_generation(prompt_engine, test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    expected_relationship_labels = test_data_biocypher_query_generation[4]
    expected_properties = test_data_biocypher_query_generation[5]
    expected_parts_of_query = test_data_biocypher_query_generation[6]

    prompt_engine = prompt_engine(kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}")
    query = prompt_engine._generate_query(
        question=prompt,
        entities=expected_entities,
        relationships=expected_relationship_labels,
        properties=expected_properties,
        query_language="Cypher",
    )

    score = []
    for expected_part_of_query in expected_parts_of_query:
        print(expected_part_of_query)
        if isinstance(expected_part_of_query, tuple):
            score.append(expected_part_of_query[0] in query or expected_part_of_query[1] in query)
        else:
            score.append(expected_part_of_query in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},cypher query,{calculate_test_score(score)}\n"
        )


def test_end_to_end_query_generation(prompt_engine, test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_parts_of_query = test_data_biocypher_query_generation[6]

    prompt_engine = prompt_engine(kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}")

    query = prompt_engine.generate_query(
        question=prompt,
        query_language="Cypher",
    )

    score = []
    for expected_part_of_query in expected_parts_of_query:
        print(expected_part_of_query)
        if isinstance(expected_part_of_query, tuple):
            score.append(expected_part_of_query[0] in query or expected_part_of_query[1] in query)
        else:
            score.append(expected_part_of_query in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},end-to-end,{calculate_test_score(score)}\n"
        )
