from typing import Literal

import guidance
from guidance import assistant, select, user

GLOBAL_PROMPT = """Your are a helpful assistant and your task is to help users formulate queries that will inform a RAG operation. You will be exposed to a question from the user and the schema describing a knowledge graph. You will receive as context the original question by the user and a task to perform before writing the final query"""
STRUCT_ENTITY_SELECTION_PROMPT = """You have access to a knowledge graph that contains these entity types: {entities}. Your task is to select the entity types that are relevant to the user's question for subsequent use in a query."""
STRUCT_RELATIONSHIP_SELECTION_PROMPT = """You have access to a knowledge graph that contains these entities: {selected_entities}. Your task is to select the relationships that are relevant to the user's question for subsequent use in a query. Only return the relationships without their sources or targets. Here are the possible relationships and their source and target entities: {relationships}."""
STRUCT_PROPERTY_SELECTION_PROMPT = """You have access to a knowledge graph that contains entities and relationships. They have the following properties. Entities:{entity_properties}, Relationships: {relationship_properties}. Your task is to select the properties that are relevant to the user's question for subsequent use in a query. Only return the entities and relationships with their relevant properties."""


@guidance
def list_selection(
    lm, question: str, prompt: str, elements: list, element_type: Literal["entities", "relationships"]
) -> list:
    """Select entities from a list of entities based on a prompt."""
    # user prompt
    with user():
        lm += GLOBAL_PROMPT
        lm += f"\n\n<question>{question}</question>"
        lm += f"\n\n<task>{prompt}</task>"

    # selected number of entities
    number = list(range(len(elements)))
    number = [str(i) for i in number]

    # selected entities
    with assistant():
        lm += f"I have identified {select(number, name='number_of_' + element_type)} {element_type} that are relevant to the user's question:\n"

        # select the entities
        for i in range(int(lm[f"number_of_{element_type}"])):
            lm += f'''{i + 1}. "{select(elements, name="selected_" + element_type + "_" + str(i))}"''' + "\n"

    return lm


@guidance(stateless=False)
def dictionary_selection(lm, question: str, prompt: str, e_props: dict, r_props: dict, container: dict) -> list:
    """Select properties for entities and for relationships from two dictionaries"""
    # user prompt
    with user():
        lm += GLOBAL_PROMPT
        lm += f"\n\n<question>{question}</question>"
        lm += f"\n\n<task>{prompt}</task>"

    # select the entities
    with assistant():
        # start with entities
        lm += "Determining the relevant properties for the entities\n"

        for key, value in e_props.items():
            container[key] = []

            # determine the number of properties of the considered entity
            number_props = list(range(len(value)))
            number_props = [str(i) for i in number_props]

            # select the number of properties to consider
            lm += f"For entity {key}, I have identified "
            lm += select(number_props, name="current_key")
            lm += " properties:\n"

            # select the properties to consider
            for i in range(int(lm["current_key"])):
                lm += f'''{i + 1}. "{select(e_props[key], name="current_property")}"''' + "\n"
                container[key].append(lm["current_property"])

        # now for relationships
        lm += "\nDetermining the relevant properties for the relationships\n"

        for key, value in r_props.items():
            container[key] = []

            number_props = list(range(len(value)))
            number_props = [str(i) for i in number_props]

            # select the number of properties to consider
            lm += f"For relationship {key}, I have identified "
            lm += select(number_props, name="current_key")
            lm += " properties:\n"

            # select the properties to consider
            for i in range(int(lm["current_key"])):
                lm += f'''{i + 1}. "{select(r_props[key], name="current_property")}"''' + "\n"
                container[key].append(lm["current_property"])

        #remove from the container empty keys
        container = {k: v for k, v in container.items() if v}

    return lm
