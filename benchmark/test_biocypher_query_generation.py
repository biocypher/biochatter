import re
import json
import inspect

import pytest

from biochatter.prompts import BioCypherPromptEngine
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def test_naive_query_generation_using_schema(
    model_name,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    schema = kg_schemas[yaml_data["input"]["kg_schema"]]

    def run_test():
        conversation.reset()  # needs to be reset for each test

        conversation.append_system_message(
            "You are a database expert. Please write a Cypher query to "
            "retrieve information for the user. The schema of the graph is "
            "defined as follows: "
        )
        conversation.append_system_message(json.dumps(schema, indent=2))
        conversation.append_system_message(
            "Only return the query, nothing else."
        )

        query, _, _ = conversation.query(yaml_data["input"]["prompt"])

        score = []
        for expected_part_of_query in yaml_data["expected"]["parts_of_query"]:
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
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def get_prompt_engine(
    kg_schema_dict: dict,
    create_prompt_engine,
) -> BioCypherPromptEngine:
    """Helper function to create the prompt engine for the test.

    Args:
        kg_schema_dict (dict): The KG schema
        create_prompt_engine: The function to create the BioCypherPromptEngine

    Returns:
        BioCypherPromptEngine: The prompt engine for the test
    """
    return create_prompt_engine(kg_schema_dict=kg_schema_dict)


def test_entity_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        success = prompt_engine._select_entities(
            question=yaml_data["input"]["prompt"],
            conversation=conversation,
        )
        assert success

        score = []
        for expected_entity in yaml_data["expected"]["entities"]:
            score.append(expected_entity in prompt_engine.selected_entities)
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def test_relationship_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    if not yaml_data["expected"]["relationships"]:
        pytest.skip("No relationships to test")
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    prompt_engine.question = yaml_data["input"]["prompt"]
    prompt_engine.selected_entities = yaml_data["expected"]["entities"]

    # TODO: more generic, for nested structures

    def run_test():
        conversation.reset()  # needs to be reset for each test
        success = prompt_engine._select_relationships(conversation=conversation)
        assert success

        score = []
        for expected_relationship_label_key in yaml_data["expected"][
            "relationship_labels"
        ].keys():
            score.append(
                expected_relationship_label_key
                in prompt_engine.selected_relationship_labels.keys()
            )

            for expected_relationship_label_value in yaml_data["expected"][
                "relationship_labels"
            ][expected_relationship_label_key]:
                try:
                    score.append(
                        expected_relationship_label_value
                        in prompt_engine.selected_relationship_labels[
                            expected_relationship_label_key
                        ]
                    )
                except KeyError:
                    score.append(False)
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def test_property_selection(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    prompt_engine.question = yaml_data["input"]["prompt"]
    prompt_engine.selected_entities = yaml_data["expected"]["entities"]
    prompt_engine.selected_relationships = yaml_data["expected"][
        "relationships"
    ]

    def run_test():
        conversation.reset()  # needs to be reset for each test
        success = prompt_engine._select_properties(conversation=conversation)

        if success:
            score = []
            for expected_property_key in yaml_data["expected"][
                "properties"
            ].keys():
                try:
                    score.append(
                        expected_property_key
                        in prompt_engine.selected_properties.keys()
                    )
                except KeyError:
                    score.append(False)

                for expected_property_value in yaml_data["expected"][
                    "properties"
                ][expected_property_key]:
                    try:
                        score.append(
                            expected_property_value
                            in prompt_engine.selected_properties[
                                expected_property_key
                            ]
                        )
                    except KeyError:
                        score.append(False)
        else:
            total_properties = len(
                yaml_data["expected"]["properties"].keys()
            ) + sum(
                len(v) for v in yaml_data["expected"]["properties"].values()
            )
            score = [False] * total_properties

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def test_query_generation(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        query = prompt_engine._generate_query(
            question=yaml_data["input"]["prompt"],
            entities=yaml_data["expected"]["entities"],
            relationships=yaml_data["expected"]["relationship_labels"],
            properties=yaml_data["expected"]["properties"],
            query_language="Cypher",
            conversation=conversation,
        )

        score = []
        for expected_part_of_query in yaml_data["expected"]["parts_of_query"]:
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
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def test_end_to_end_query_generation(
    model_name,
    prompt_engine,
    test_data_biocypher_query_generation,
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        try:
            query = prompt_engine.generate_query(
                question=yaml_data["input"]["prompt"],
                query_language="Cypher",
            )
            score = []
            for expected_part_of_query in yaml_data["expected"][
                "parts_of_query"
            ]:
                if isinstance(expected_part_of_query, tuple):
                    score.append(
                        expected_part_of_query[0] in query
                        or expected_part_of_query[1] in query
                    )
                else:
                    score.append(
                        (re.search(expected_part_of_query, query) is not None)
                    )
        except ValueError as e:
            score = [False for _ in yaml_data["expected"]["parts_of_query"]]

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
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
    kg_schemas,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_biocypher_query_generation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    prompt_engine = get_prompt_engine(
        kg_schemas[yaml_data["input"]["kg_schema"]], prompt_engine
    )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        query = prompt_engine._generate_query(
            question=yaml_data["input"]["prompt"],
            entities=yaml_data["expected"]["entities"],
            relationships=yaml_data["expected"]["relationship_labels"],
            properties=yaml_data["expected"]["properties"],
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
                score.append(False)

        # if score is shorter than the least expected number of properties, add
        # False values until the length is reached
        score += [False] * (len(yaml_data["expected"]["entities"]) - len(score))
        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        prompt_engine.model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


@pytest.mark.skip(reason="Helper function for testing regex patterns")
def test_regex(test_data_biocypher_query_generation):
    yaml_data = test_data_biocypher_query_generation
    query = 'MATCH (g:Gene)-[:GENE_EXPRESSED_IN_CELL_TYPE]->(c:CellType) WHERE c.cell_type_name = "fibroblast" RETURN g.id, g.name, c.cell_type_name, c.expression_level ORDER BY c.expression_level DESC'
    score = []
    for expected_part_of_query in yaml_data["expected"]["parts_of_query"]:
        if isinstance(expected_part_of_query, tuple):
            score.append(
                expected_part_of_query[0] in query
                or expected_part_of_query[1] in query
            )
        else:
            score.append((re.search(expected_part_of_query, query) is not None))

        assert True
