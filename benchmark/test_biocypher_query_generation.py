from biochatter.prompts import BioCypherPromptEngine
import pytest
from .conftest import calculate_test_score, RESULT_FILES
import re


FILE_PATH = next(
    (
        s
        for s in RESULT_FILES
        if "biocypher" in s and "query" in s and "generation" in s
    ),
    None,
)

OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4",
]

XINFERENCE_MODEL_NAMES = [
    "llama2-hf",
    "llama2-chat-hf",
]

BENCHMARKED_MODELS = OPENAI_MODEL_NAMES + XINFERENCE_MODEL_NAMES


@pytest.fixture(scope="module", params=BENCHMARKED_MODELS)
def prompt_engine(request):
    model_name = request.param
    if model_name in OPENAI_MODEL_NAMES:
        return BioCypherPromptEngine(
            schema_config_or_info_path="test/test_schema_info.yaml",
            model_name=model_name,
        )
    elif model_name in XINFERENCE_MODEL_NAMES:
        # TODO implement starting a model on xinference server and connecting
        # from the prompt engine
        pass


def test_entity_selection(prompt_engine):
    success = prompt_engine._select_entities(
        question="Which genes are associated with mucoviscidosis?"
    )
    assert success

    score = []
    score.append("Gene" in prompt_engine.selected_entities)
    score.append("Disease" in prompt_engine.selected_entities)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},entities_single,{calculate_test_score(score)}\n"
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

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},relationships_single,{calculate_test_score(score)}\n"
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

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},properties_single,{calculate_test_score(score)}\n"
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
    cypher_regex = r"MATCH \([a-zA-Z]*:Gene\)<-\[[a-zA-Z]*:PERTURBED\]-\([a-zA-Z]*:Disease.*\)|MATCH \([a-zA-Z]*:Disease.*\)-\[[a-zA-Z]*:PERTURBED\]->\([a-zA-Z]*:Gene\)"
    score.append((re.search(cypher_regex, query) is not None))
    score.append("WHERE" in query or "{name:" in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},cypher query_single,{calculate_test_score(score)}\n"
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
    cypher_regex = r"MATCH \([a-zA-Z]*:Gene\)<-\[[a-zA-Z]*:PERTURBED\]-\([a-zA-Z]*:Disease.*\)|MATCH \([a-zA-Z]*:Disease.*\)-\[[a-zA-Z]*:PERTURBED\]->\([a-zA-Z]*:Gene\)"
    score.append((re.search(cypher_regex, query) is not None))
    score.append("WHERE" in query or "{name:" in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},end-to-end_single,{calculate_test_score(score)}\n"
        )


######
# test multiple word entities
######


# entity selection doesn't have any issue
def test_entity_selection_multi_word(prompt_engine):
    success = prompt_engine._select_entities(
        question="Which genes are expressed in fibroblast?"
    )
    assert success

    score = []
    score.append("Gene" in prompt_engine.selected_entities)
    score.append("CellType" in prompt_engine.selected_entities)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},entities_multi,{calculate_test_score(score)}\n"
        )


# relationship selection doesn't have any issue
def test_relationship_selection_multi_word(prompt_engine):
    prompt_engine.question = "Which genes are expressed in fibroblast?"
    prompt_engine.selected_entities = ["Gene", "CellType"]
    success = prompt_engine._select_relationships()
    assert success

    score = []
    score.append(
        prompt_engine.selected_relationships == ["GeneExpressedInCellType"]
    )
    score.append(
        "GENE_EXPRESSED_IN_CELL_TYPE"
        in prompt_engine.selected_relationship_labels.keys()
    )

    score.append(
        "source"
        in prompt_engine.selected_relationship_labels.get(
            "GENE_EXPRESSED_IN_CELL_TYPE"
        )
    )
    score.append(
        "target"
        in prompt_engine.selected_relationship_labels.get(
            "GENE_EXPRESSED_IN_CELL_TYPE"
        )
    )
    score.append(
        ## and or or?
        "Gene"
        in prompt_engine.selected_relationship_labels.get(
            "GENE_EXPRESSED_IN_CELL_TYPE"
        ).get("source")
    )
    score.append(
        "CellType"
        in prompt_engine.selected_relationship_labels.get(
            "GENE_EXPRESSED_IN_CELL_TYPE"
        ).get("target")
    )

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},relationships_multi,{calculate_test_score(score), prompt_engine.selected_relationships, prompt_engine.selected_relationship_labels}\n"
        )


# select properties DOES have issue


def test_property_selection_multi_word(prompt_engine):
    prompt_engine.question = "Which genes are expressed in fibroblast?"
    prompt_engine.selected_entities = ["Gene", "CellType"]
    prompt_engine.selected_relationships = ["GeneExpressedInCellType"]
    success = prompt_engine._select_properties()
    assert success

    score = []
    score.append("CellType" in prompt_engine.selected_properties.keys())
    if "CellType" in prompt_engine.selected_properties.keys():
        # only run if CellType is actually a selected property
        score.append(
            "cell_type_name"
            in prompt_engine.selected_properties.get("CellType")
        )
        # require a list of relevant properties
        score.append(
            isinstance(prompt_engine.selected_properties.get("CellType"), list)
        )
    else:
        score.append(0)
        score.append(0)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},properties_multi,{calculate_test_score(score), prompt_engine.selected_properties}\n"
        )


def test_query_generation_multi_word(prompt_engine):
    query = prompt_engine._generate_query(
        question="Which genes are expressed in fibroblast?",
        entities=["Gene", "CellType"],
        relationships={
            "GENE_EXPRESSED_IN_CELL_TYPE": {
                "source": "Gene",
                "target": "CellType",
            },
        },
        properties={
            "CellType": ["cell_type_name"],
        },
        query_language="Cypher",
    )

    score = []
    score.append("MATCH" in query)
    score.append("RETURN" in query)
    score.append("Gene" in query)
    score.append("CellType" in query)
    score.append("fibroblast" in query)

    # make sure direction is right
    cypher_regex = r"MATCH \([a-zA-Z]*:Gene\)-\[[a-zA-Z]*:GENE_EXPRESSED_IN_CELL_TYPE\]->\([a-zA-Z]*:CellType.*|MATCH \([a-zA-Z]*:CellType.*<-\[[a-zA-Z]*:GENE_EXPRESSED_IN_CELL_TYPE\]-\([a-zA-Z]*:Gene\)"
    score.append((re.search(cypher_regex, query) is not None))
    # make sure conditioned
    score.append("WHERE" in query or "{cell_type_name:" in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},cypher query_single,{calculate_test_score(score),query}\n"
        )


def test_end_to_end_query_generation_multi_word(prompt_engine):
    query = prompt_engine.generate_query(
        question="Which genes are expressed in fibroblast?",
        query_language="Cypher",
    )

    score = []
    score.append("MATCH" in query)
    score.append("RETURN" in query)
    score.append("Gene" in query)
    score.append("CellType" in query)
    score.append("fibroblast" in query)

    # make sure direction is right
    cypher_regex = r"MATCH \([a-zA-Z]*:Gene\)-\[[a-zA-Z]*:GENE_EXPRESSED_IN_CELL_TYPE\]->\([a-zA-Z]*:CellType.*|MATCH \([a-zA-Z]*:CellType.*<-\[[a-zA-Z]*:GENE_EXPRESSED_IN_CELL_TYPE\]-\([a-zA-Z]*:Gene\)"
    score.append((re.search(cypher_regex, query) is not None))
    # make sure conditioned
    score.append("WHERE" in query or "{cell_type_name:" in query)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},end-to-end_multi,{calculate_test_score(score),query}\n"
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

    # get peroperties used in curly brackets
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


def test_property_exists(prompt_engine):
    query = prompt_engine.generate_query(
        question="What are the hgnc ids of the genes are expressed in fibroblast?",
        query_language="Cypher",
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
            # check property used is in available propertis for entities
            avail_property_entity = list(
                prompt_engine.entities[entity]["properties"].keys()
            )
            score.append(property in avail_property_entity)

        elif (
            entity in prompt_engine.relationships.keys()
            and "properties" in prompt_engine.relationships[entity]
        ):
            # check property used is in available propertis for relationships
            avail_property_entity = list(
                prompt_engine.relationships[entity]["properties"].keys()
            )
            score.append(property in avail_property_entity)
        else:
            # no properties of the entity or relationship exist, simply made up
            score.append(0)

    with open(FILE_PATH, "a") as f:
        f.write(
            f"{prompt_engine.model_name},property hallucination check,{entity_mapping}, {property_mapping}, {used_entity_property}, {calculate_test_score(score)}, {query}\n"
        )
