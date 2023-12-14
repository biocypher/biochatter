#!/usr/bin/env Python3

from typing import Optional
from 


class message(str):

    def __init__(
            self,
            delimiter: str = "####",
            content: str = None,   ## content of the message
            label: Optional[str] = None,
            message_type: Optional[str] = None,
           

    ):
       
        super().__init__()
        self.label = label
        self.message_type = message_type
        self.delimiter = delimiter
        self.content = content

        def _add_kg_basic_info(self):

            kg_info_msg=("You have access to a knowledge graph. A knowledge graph is a knowledge representation that models information as a graph, where nodes represent entities, "
                         "and edges represent relationships between those entities. "
                         "Entities and relationships always have IDs which you can use to identify them. "
                         "Entities and relationships can have properties but they are not neceassary. "
                         )

        # get a message that tells the model basic things about a KG
        
            self.content = f"{kg_info_msg}\n{self.content}"




class system_message(message):

    def __init__(
            self,
            message_type: str = 'system',
            message_for: str = None,
            content: str = None,   ## content of the message
            delimiter: str = None,
            label: Optional[str] = None,

    ):
        super().__init__(
            message_type=message_type,
            label=label,
            delimiter=delimiter,
            content=content
        )

        message_for_options = ['entity', 'relationship', 'property', 'query']

        if message_for is not None:
            if message_for not in message_for_options:
                raise ValueError("message_for must be one of ['entity', 'relationship', 'property', 'query']")
            
            self.message_for = message_for



    def _add_system_message_entity(self, entities):

        self.message_for = 'entity'
        self.label = 'initial' # mark that this is the first set of prompts used right now, hard code for now, maybe as a param later for benchmarking

        # TODO: an option to add kg info at the beginning of this message

        system_message_entity=("You have access to a knowledge graph that contains "
                f"the entities after the delimiter {self.delimiter} in the end of this message."
                "Your task is to select the ones that are relevant to the user's question "
                "for subsequent use in a query. Only return the entities, "
                "comma-separated, without any additional text. "
                f"{self.delimiter}"
                f"{entities}")
        
        self.content = f"{self.content}\n{system_message_entity}"

    def _add_system_message_relationship(self, selected_entities, relationships):

        self.message_for = 'relationship'
        self.label = 'initial' # mark that this is the first set of prompts used right now, hard code for now, maybe as a param later for benchmarking

        system_message_relationship = (
            "You have access to a knowledge graph that contains entities and relationships "
            f"after the delimiter {self.delimiter} in the end of this message."
            "Your task is to select the relationships that are relevant "
            "to the user's question for subsequent use in a query. Only "
            "return the relationships without their sources or targets, "
            "comma-separated, and without any additional text. Here are the "
            "possible relationships and their source and target entities: "
            f"{self.delimiter}"
            f"entities:{', '.join(selected_entities)}"
            f"relationships: {relationships}")
        
        self.content = f"{self.content}\n{system_message_relationship}"


    def _add_system_message_property(self, entity_properties, relationship_properties):

        self.message_for = 'property'
        self.label = 'initial' # mark that this is the first set of prompts used right now, hard code for now, maybe as a param later for benchmarking

        system_message_property =("You have access to a knowledge graph that contains entities and "
            f"relationships. Their properties are after the delimiter {self.delimiter} in the end of this message."
            "Your task is to select the properties that are relevant to the "
            "user's question for subsequent use in a query. Only return the "
            "entities and relationships with their relevant properties in JSON "
            "format, without any additional text. Return the "
            "entities/relationships as top-level dictionary keys, and their "
            "properties as dictionary values. "
            "Do not return properties that are not relevant to the question."
            f"{self.delimiter}"
            f"entities and their properties: {entity_properties}"
            f"relationships and their properties: {relationship_properties}")
        
        self.content = f"{self.content}\n{system_message_property}"



    def _add_system_message_query(self, entities, relationships, properties, query_language):

        self.message_for = 'property'
        self.label = 'initial' # mark that this is the first set of prompts used right now, hard code for now, maybe as a param later for benchmarking

        system_message_query = (
            f"Generate a database query in {query_language} that answers "
            f"the user's question. "
            f"You can use the following entities: {entities}, "
            f"relationships: {list(relationships.keys())}, and "
            f"properties: {properties}. "
        )
        
        self.content = f"{self.content}\n{system_message_query}"

