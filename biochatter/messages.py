#!/usr/bin/env Python3

from typing import Optional

primary_message_entity = (
    "You have access to a knowledge graph that contains "
    "the entities after the delimiter {delimiter} in the end of this message."
    "Your task is to select the ones that are relevant to the user's question "
    "for subsequent use in a query. Only return the entities, "
    "comma-separated, without any additional text. "
    "{delimiter}"
    "{entities}"
)


primary_message_relationship = (
    "You have access to a knowledge graph that contains entities and relationships "
    "after the delimiter {delimiter} in the end of this message."
    "Your task is to select the relationships that are relevant "
    "to the user's question for subsequent use in a query. Only "
    "return the relationships without their sources or targets, "
    "comma-separated, and without any additional text. Here are the "
    "possible relationships and their source and target entities: "
    "{delimiter}"
    "entities:{selected_entities_joined}"
    "relationships: {relationships}"
)

primary_message_property = (
    "You have access to a knowledge graph that contains entities and "
    "relationships. Their properties are after the delimiter {delimiter} in the end of this message."
    "Your task is to select the properties that are relevant to the "
    "user's question for subsequent use in a query. Only return the "
    "entities and relationships with their relevant properties in JSON "
    "format, without any additional text. Return the "
    "entities/relationships as top-level dictionary keys, and their "
    "properties as dictionary values. "
    "Do not return properties that are not relevant to the question."
    "{delimiter}"
    "entities and their properties: {entity_properties}"
    "relationships and their properties: {relationship_properties}"
)


primary_message_query = (
    "Generate a database query in {query_language} that answers "
    "the user's question. "
    "You can use the following entities: {entities}, "
    "relationships: {relationships_key_list}, and "
    "properties: {properties}. "
)


class message(str):
    def __init__(
        self,
        delimiter: str = "####",
        content: str = None,  ## content of the message
        label: Optional[str] = "primary",
        message_type: Optional[str] = None,
    ):
        super().__init__()
        self.label = label
        self.message_type = message_type
        self.delimiter = delimiter
        self.content = content

        def _add_kg_basic_info(self):
            kg_info_msg = (
                "You have access to a knowledge graph. A knowledge graph is a knowledge representation that models information as a graph, "
                "where nodes represent entities, "
                "and edges represent relationships between those entities. "
                "Entities and relationships always have IDs which you can use to identify them. "
                "Entities and relationships can have properties, which you can also use to identify them, but they are not neceassary. "
            )

            # get a message that tells the model basic things about a KG

            self.content = f"{kg_info_msg}\n{self.content}"


class system_message(message):
    def __init__(
        self,
        message_type: str = "system",
        message_for: str = None,
        content: str = None,  ## content of the message
        delimiter: str = None,
        label: Optional[str] = None,
    ):
        super().__init__(
            message_type=message_type,
            label=label,
            delimiter=delimiter,
            content=content,
        )

        message_for_options = ["entity", "relationship", "property", "query"]

        if message_for is not None:
            if message_for not in message_for_options:
                raise ValueError(
                    "message_for must be one of ['entity', 'relationship', 'property', 'query']"
                )

            self.message_for = message_for

    def _add_system_message_entity(self, entities):
        self.message_for = "entity"

        # TODO: an option to add kg info at the beginning of this message
        # TODO: other propts then primary
        if self.label == "primary":
            system_message_entity = primary_message_entity.format(
                delimiter=self.delimiter, entities=entities
            )
            self.content = f"{self.content}\n{system_message_entity}"

    def _add_system_message_relationship(
        self, selected_entities, relationships
    ):
        self.message_for = "relationship"
        selected_entities_joined = ", ".join(selected_entities)

        if self.label == "primary":
            system_message_relationship = primary_message_relationship.format(
                delimiter=self.delimiter,
                selected_entities=selected_entities_joined,
                relationships=relationships,
            )
            self.content = f"{self.content}\n{system_message_relationship}"

    def _add_system_message_property(
        self, entity_properties, relationship_properties
    ):
        self.message_for = "property"
        if self.label == "primary":
            system_message_property = primary_message_property.format(
                delimiter=self.delimiter,
                entity_properties=entity_properties,
                relationship_properties=relationship_properties,
            )
            self.content = f"{self.content}\n{system_message_property}"

    def _add_system_message_query(
        self, entities, relationships, properties, query_language
    ):
        self.message_for = "query"
        relationships_key_list = list(relationships.keys())

        if self.label == "primary":
            system_message_query = primary_message_query.format(
                query_language=query_language,
                entities=entities,
                relationships_key_list=relationships_key_list,
                properties=properties,
            )
            self.content = f"{self.content}\n{system_message_query}"
