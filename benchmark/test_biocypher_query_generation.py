import os
import re
import inspect

import pytest

import pandas as pd

from biochatter.prompts import BioCypherPromptEngine
from biochatter.llm_connect import GptConversation, XinferenceConversation
from .conftest import RESULT_FILES, calculate_test_score

TASK = "biocypher_query_generation"

# find right file to write to
# TODO should we use SQLite? An online database (REDIS)?
FILE_PATH = next(
    (
        s
        for s in RESULT_FILES
        if "biocypher" in s and "query" in s and "generation" in s
    ),
    None,
)

# set model matrix
# TODO should probably go to conftest.py
OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    # "gpt-4",
]

XINFERENCE_MODEL_NAMES = [
    # "llama2-hf",
    # "llama2-chat-hf",
]

BENCHMARKED_MODELS = OPENAI_MODEL_NAMES + XINFERENCE_MODEL_NAMES

BENCHMARK_URL = "http://llm.biocypher.org"


@pytest.fixture(scope="module", params=BENCHMARKED_MODELS)
def prompt_engine(request):
    def setup_prompt_engine(kg_schema_path):
        model_name = request.param
        return BioCypherPromptEngine(
            schema_config_or_info_path=kg_schema_path,
            model_name=model_name,
        )

    return setup_prompt_engine


@pytest.fixture(scope="function", params=BENCHMARKED_MODELS)
def conversation(request):
    model_name = request.param
    if model_name in OPENAI_MODEL_NAMES:
        conversation = GptConversation(
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            os.getenv("OPENAI_API_KEY"), user="benchmark_user"
        )
    elif model_name in XINFERENCE_MODEL_NAMES:
        # TODO here we probably need to start the right model on the server
        conversation = XinferenceConversation(
            base_url=BENCHMARK_URL,
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key()

    return conversation


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
    new_row = pd.DataFrame(
        [[model_name, subtask, score]], columns=results.columns
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by="subtask"
    )
    results.to_csv(FILE_PATH, index=False)


def get_test_data(test_data_biocypher_query_generation):
    kg_schema_file_name = test_data_biocypher_query_generation[0]
    prompt = test_data_biocypher_query_generation[1]
    expected_entities = test_data_biocypher_query_generation[2]
    expected_relationships = test_data_biocypher_query_generation[3]
    expected_relationship_labels = test_data_biocypher_query_generation[4]
    expected_properties = test_data_biocypher_query_generation[5]
    expected_parts_of_query = test_data_biocypher_query_generation[6]
    test_case_purpose = test_data_biocypher_query_generation[7]
    test_case_index = test_data_biocypher_query_generation[8]
    return (
        kg_schema_file_name,
        prompt,
        expected_entities,
        expected_relationships,
        expected_relationship_labels,
        expected_properties,
        expected_parts_of_query,
        test_case_purpose,
        test_case_index,
    )


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
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )

    success = prompt_engine._select_entities(
        question=prompt, conversation=conversation
    )
    assert success

    score = []
    for expected_entity in expected_entities:
        score.append(expected_entity in prompt_engine.selected_entities)

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_relationship_selection(
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    success = prompt_engine._select_relationships(conversation=conversation)
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

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_property_selection(
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )

    prompt_engine.question = prompt
    prompt_engine.selected_entities = expected_entities
    prompt_engine.selected_relationships = expected_relationships
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
                    in prompt_engine.selected_properties[expected_property_key]
                )
            except KeyError:
                score.append(0)

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_query_generation(
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )
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
        print(expected_part_of_query)
        if isinstance(expected_part_of_query, tuple):
            score.append(
                expected_part_of_query[0] in query
                or expected_part_of_query[1] in query
            )
        else:
            score.append((re.search(expected_part_of_query, query) is not None))

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )


def test_end_to_end_query_generation(
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )
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
            score.append((re.search(expected_part_of_query, query) is not None))

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
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


def join_dictionaries(dict1, dict2):
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
    prompt_engine,
    test_data_biocypher_query_generation,
    result_files,
    conversation,
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
    prompt_engine = setup_test(
        kg_schema_file_name, prompt_engine, result_files, subtask
    )

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

    write_results_to_file(
        prompt_engine.model_name, subtask, calculate_test_score(score)
    )
