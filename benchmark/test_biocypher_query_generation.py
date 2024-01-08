import inspect

import pandas as pd
import pytest

from biochatter.prompts import BioCypherPromptEngine
from .conftest import calculate_test_score, RESULT_FILES

TASK = "biocypher_query_generation"

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


def benchmark_already_executed(
    task: str,
    subtask: str,
    model_name: str,
    result_files: dict[str, pd.DataFrame],
) -> bool:
    """
    Checks if the benchmark task and subtask for the model_name have already
    been executed.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"
        subtask (str): The benchmark subtask, e.g. "entities"
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        result_files (dict[str, pd.DataFrame]): The result files

    Returns:

        bool: True if the benchmark task and subtask for the model_name has
            already been run, False otherwise
    """
    task_results = result_files[f"benchmark/results/{task}.csv"]
    task_results_subset = (task_results["model"] == model_name) & (
        task_results["subtask"] == subtask
    )
    return task_results_subset.any()


def write_results_to_file(model_name: str, subtask: str, score: str):
    results = pd.read_csv(FILE_PATH, header=0)
    new_row = pd.DataFrame([[model_name, subtask, score]], columns=results.columns)
    results = pd.concat([results, new_row], ignore_index=True).sort_values(by='subtask')
    results.to_csv(FILE_PATH, index=False)


def get_test_data(test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    expected_relationships = test_data_biocypher_query_generation[3]
    expected_relationship_labels = test_data_biocypher_query_generation[4]
    expected_properties = test_data_biocypher_query_generation[5]
    expected_parts_of_query = test_data_biocypher_query_generation[6]
    test_case_index = test_data_biocypher_query_generation[7]
    return kg_schema_file_name, prompt, expected_entities, expected_relationships, expected_relationship_labels, expected_properties, expected_parts_of_query, test_case_index


def setup_test(kg_schema_file_name, prompt_engine, result_files, subtask):
    prompt_engine = prompt_engine(
        kg_schema_path=f"./benchmark/data/biocypher_query_generation/{kg_schema_file_name}"
    )
    if benchmark_already_executed(
            TASK, subtask, prompt_engine.model_name, result_files
    ):
        pytest.skip(
            f"benchmark {TASK}: {subtask} with {prompt_engine.model_name} already executed"
        )
    return prompt_engine


def test_entity_selection(
    prompt_engine, test_data_biocypher_query_generation, result_files
):
    kg_schema_file_name, prompt, expected_entities, _, _, _, _, test_case_index = get_test_data(
        test_data_biocypher_query_generation)
    subtask = inspect.currentframe().f_code.co_name + "_" + str(test_case_index)
    prompt_engine = setup_test(kg_schema_file_name, prompt_engine, result_files, subtask)

    success = prompt_engine._select_entities(question=prompt)
    assert success

    score = []
    for expected_entity in expected_entities:
        score.append(expected_entity in prompt_engine.selected_entities)

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_relationship_selection(
    prompt_engine, test_data_biocypher_query_generation, result_files
):
    kg_schema_file_name, prompt, expected_entities, _, expected_relationship_labels, _, _, test_case_index = get_test_data(
        test_data_biocypher_query_generation)
    subtask = inspect.currentframe().f_code.co_name + "_" + str(test_case_index)
    prompt_engine = setup_test(kg_schema_file_name, prompt_engine, result_files, subtask)

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    success = prompt_engine._select_relationships()
    assert success

    score = []
    for expected_relationship in expected_relationship_labels:
        score.append(
            expected_relationship in prompt_engine.selected_relationships
        )

    for expected_relationship_label_key in expected_relationship_labels.keys():
        score.append(
            expected_relationship_label_key
            in prompt_engine.selected_relationship_labels.keys()
        )

        for expected_relationship_label_value in expected_relationship_labels[
            expected_relationship_label_key
        ]:
            score.append(
                expected_relationship_label_value
                in prompt_engine.selected_relationship_labels[
                    expected_relationship_label_key
                ]
            )

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

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_property_selection(
    prompt_engine, test_data_biocypher_query_generation, result_files
):
    kg_schema_file_name, prompt, expected_entities, expected_relationships, _, expected_properties, _, test_case_index = get_test_data(
        test_data_biocypher_query_generation)
    subtask = inspect.currentframe().f_code.co_name + "_" + str(test_case_index)
    prompt_engine = setup_test(kg_schema_file_name, prompt_engine, result_files, subtask)

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    prompt_engine.selected_relationships = expected_relationships
    success = prompt_engine._select_properties()
    assert success

    score = []
    for expected_property_key in expected_properties.keys():
        score.append(
            expected_property_key in prompt_engine.selected_properties.keys()
        )

        for expected_property_value in expected_properties[
            expected_property_key
        ]:
            score.append(
                expected_property_value
                in prompt_engine.selected_properties[expected_property_key]
            )

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_query_generation(
    prompt_engine, test_data_biocypher_query_generation, result_files
):
    kg_schema_file_name, prompt, expected_entities, _, expected_relationship_labels, expected_properties, expected_parts_of_query, test_case_index = get_test_data(
        test_data_biocypher_query_generation)
    subtask = inspect.currentframe().f_code.co_name + "_" + str(test_case_index)
    prompt_engine = setup_test(kg_schema_file_name, prompt_engine, result_files, subtask)

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
            score.append(
                expected_part_of_query[0] in query
                or expected_part_of_query[1] in query
            )
        else:
            score.append(expected_part_of_query in query)

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_end_to_end_query_generation(
    prompt_engine, test_data_biocypher_query_generation, result_files
):
    kg_schema_file_name, prompt, _, _, _, _, expected_parts_of_query, test_case_index = get_test_data(
        test_data_biocypher_query_generation)
    subtask = inspect.currentframe().f_code.co_name + "_" + str(test_case_index)
    prompt_engine = setup_test(kg_schema_file_name, prompt_engine, result_files, subtask)

    query = prompt_engine.generate_query(
        question=prompt,
        query_language="Cypher",
    )

    score = []
    for expected_part_of_query in expected_parts_of_query:
        print(expected_part_of_query)
        if isinstance(expected_part_of_query, tuple):
            score.append(
                expected_part_of_query[0] in query
                or expected_part_of_query[1] in query
            )
        else:
            score.append(expected_part_of_query in query)

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )
