import re
import inspect

import pytest

from biochatter.prompts import BioCypherPromptEngine
from .conftest import calculate_test_score
from .benchmark_utils import (
    get_result_file_path,
    write_results_to_file,
    benchmark_already_executed,
)


def get_test_data(test_data_biocypher_query_generation: list) -> tuple:
    """Helper function to unpack the test data from the test_data_biocypher_query_generation fixture.

    Args:
        test_data_biocypher_query_generation (list): The test data from the test_data_biocypher_query_generation fixture

    Returns:
        tuple: The unpacked test data
    """
    return (
        test_data_biocypher_query_generation["kg_path"],
        test_data_biocypher_query_generation["prompt"],
        test_data_biocypher_query_generation["entities"],
        test_data_biocypher_query_generation["relationships"],
        test_data_biocypher_query_generation["relationship_labels"],
        test_data_biocypher_query_generation["properties"],
        test_data_biocypher_query_generation["parts_of_query"],
        test_data_biocypher_query_generation["test_case_purpose"],
        test_data_biocypher_query_generation["index"],
    )


def skip_if_already_run(
    model_name: str,
    task: str,
    subtask: str,
) -> None:
    """Helper function to check if the test case is already executed.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        result_files (dict[str, pd.DataFrame]): The result files
        task (str): The benchmark task, e.g. "biocypher_query_generation"
        subtask (str): The benchmark subtask test case, e.g. "0_single_word"
    """
    if benchmark_already_executed(model_name, task, subtask):
        pytest.skip(
            f"benchmark {task}: {subtask} with {model_name} already executed"
        )


def get_prompt_engine(
    kg_schema_file_name: str,
    create_prompt_engine,
) -> BioCypherPromptEngine:
    """Helper function to create the prompt engine for the test.

    Args:
        kg_schema_file_name (str): The KG schema file name
        create_prompt_engine: The function to create the BioCypherPromptEngine

    Returns:
        BioCypherPromptEngine: The prompt engine for the test
    """
    return create_prompt_engine(
        kg_schema_path=f"./benchmark/data/{kg_schema_file_name}"
    )


def test_entity_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    conversation,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        expected_entities,
        _,
        _,
        _,
        _,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    subtask = f"{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name=model_name, task=task, subtask=subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    def run_test():
        success = prompt_engine._select_entities(
            question=prompt, conversation=conversation
        )
        assert success

        score = []
        for expected_entity in expected_entities:
            score.append(expected_entity in prompt_engine.selected_entities)
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        get_result_file_path(task),
    )


def test_relationship_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        expected_entities,
        _,
        expected_relationship_labels,
        _,
        _,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities

    # TODO: more generic, for nested structures

    def run_test():
        success = prompt_engine._select_relationships(conversation=conversation)
        assert success

        score = []
        for expected_relationship in expected_relationship_labels:
            score.append(
                expected_relationship in prompt_engine.selected_relationships
            )

        for (
            expected_relationship_label_key
        ) in expected_relationship_labels.keys():
            score.append(
                expected_relationship_label_key
                in prompt_engine.selected_relationship_labels.keys()
            )

            for (
                expected_relationship_label_value
            ) in expected_relationship_labels[expected_relationship_label_key]:
                score.append(
                    expected_relationship_label_value
                    in prompt_engine.selected_relationship_labels[
                        expected_relationship_label_key
                    ]
                )
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        FILE_PATH,
    )


def test_property_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        expected_entities,
        expected_relationships,
        _,
        expected_properties,
        _,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    prompt_engine.selected_relationships = expected_relationships

    def run_test():
        success = prompt_engine._select_properties(conversation=conversation)
        assert success

        score = []
        for expected_property_key in expected_properties.keys():
            try:
                score.append(
                    expected_property_key
                    in prompt_engine.selected_properties.keys()
                )
            except KeyError:
                score.append(0)

            for expected_property_value in expected_properties[
                expected_property_key
            ]:
                try:
                    score.append(
                        expected_property_value
                        in prompt_engine.selected_properties[
                            expected_property_key
                        ]
                    )
                except KeyError:
                    score.append(0)
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        FILE_PATH,
    )


def test_query_generation(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        expected_entities,
        _,
        expected_relationship_labels,
        expected_properties,
        expected_parts_of_query,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    def run_test():
        query = prompt_engine._generate_query(
            question=prompt,
            entities=expected_entities,
            relationships=expected_relationship_labels,
            properties=expected_properties,
            query_language="Cypher",
            conversation=conversation,
        )

        score = []
        for expected_part_of_query in expected_parts_of_query:
            if isinstance(expected_part_of_query, tuple):
                score.append(
                    expected_part_of_query[0] in query
                    or expected_part_of_query[1] in query
                )
            else:
                score.append(
                    (re.search(expected_part_of_query, query) is not None)
                )
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        FILE_PATH,
    )


def test_end_to_end_query_generation(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        _,
        _,
        _,
        _,
        expected_parts_of_query,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    def run_test():
        query = prompt_engine.generate_query(
            question=prompt,
            query_language="Cypher",
        )

        score = []
        for expected_part_of_query in expected_parts_of_query:
            if isinstance(expected_part_of_query, tuple):
                score.append(
                    expected_part_of_query[0] in query
                    or expected_part_of_query[1] in query
                )
            else:
                score.append(
                    (re.search(expected_part_of_query, query) is not None)
                )
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        FILE_PATH,
    )


######
# test hallucination: are all properties available in the KG schema?
# in selected properties, also in the actual property used in the query
######


def map_entities_to_labels(entity_list):
    entity_mapping = {}
    for entity in entity_list:
        match = re.match(r"(\w+):(\w+)", entity)
        if match:
            label, entity_type = match.groups()
            entity_mapping[label] = entity_type

    return entity_mapping


def map_dot_properties_to_labels(property_list):
    property_mapping = {}
    for property in property_list:
        match = re.match(r"(\w+)\.(\w+)", property)
        if match:
            label, property_type = match.groups()
            property_mapping[label] = property_type

    return property_mapping


def map_bracket_properties_to_labels(property_list):
    property_mapping = {}
    for property in property_list:
        match = re.search(r"\((\w+):\w+ \{(\w+):", property)
        if match:
            label, property_type = match.groups()
            property_mapping[label] = property_type

    return property_mapping


def join_dictionaries(dict1: dict, dict2: dict) -> dict:
    result_dict = {}
    for key in dict1:
        if key in dict2:
            result_dict[dict1[key]] = dict2[key]

    return result_dict


def get_used_property_from_query(query):
    property_mapping = dict()

    # first get all properties used in 'dot' format

    property_regex_dot = r"[a-zA-Z]+\.\S+ |[a-zA-Z]+\..+$"
    used_properties = re.findall(property_regex_dot, query)
    used_properties = [i.strip() for i in used_properties]
    # map variable name to properties used
    property_mapping_add = map_dot_properties_to_labels(used_properties)
    property_mapping.update(property_mapping_add)

    # get properties used in curly brackets
    if "{" in query:
        property_regex_bracket = r"\(\w+:\w+ \{\w+: "
        used_properties = re.findall(property_regex_bracket, query)
        used_properties = [i.strip() for i in used_properties]
        # map variable name to properties used
        property_mapping_add = map_bracket_properties_to_labels(used_properties)
        property_mapping.update(property_mapping_add)

    # get all entities or relationships involved in the query
    entity_regex = r"[a-zA-Z]+:\w+"
    used_entities = re.findall(entity_regex, query)
    used_entities = [i.strip() for i in used_entities]

    # map variable name to entity or relationship labels
    entity_mapping = map_entities_to_labels(used_entities)

    # get all the entity and respective properties used in the cypher query
    used_entity_property = join_dictionaries(entity_mapping, property_mapping)

    return entity_mapping, property_mapping, used_entity_property


def test_property_exists(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
    multiple_testing,
):
    (
        kg_schema_file_name,
        prompt,
        expected_entities,
        _,
        expected_relationship_labels,
        expected_properties,
        expected_parts_of_query,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_biocypher_query_generation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)
    prompt_engine = get_prompt_engine(kg_schema_file_name, prompt_engine)

    def run_test():
        query = prompt_engine._generate_query(
            question=prompt,
            entities=expected_entities,
            relationships=expected_relationship_labels,
            properties=expected_properties,
            query_language="Cypher",
            conversation=conversation,
        )

        score = []

        (
            entity_mapping,
            property_mapping,
            used_entity_property,
        ) = get_used_property_from_query(query)

        for entity, property in used_entity_property.items():
            if (
                entity in prompt_engine.entities.keys()
                and "properties" in prompt_engine.entities[entity]
            ):
                # check property used is in available properties for entities
                avail_property_entity = list(
                    prompt_engine.entities[entity]["properties"].keys()
                )
                score.append(property in avail_property_entity)
            elif (
                entity in prompt_engine.relationships.keys()
                and "properties" in prompt_engine.relationships[entity]
            ):
                # check property used is in available properties for relationships
                avail_property_entity = list(
                    prompt_engine.relationships[entity]["properties"].keys()
                )
                score.append(property in avail_property_entity)
            else:
                # no properties of the entity or relationship exist, simply made up
                score.append(0)
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        subtask,
        f"{mean_score}/{max};{n_iterations}",
        FILE_PATH,
    )
