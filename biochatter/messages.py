#!/usr/bin/env Python3

from typing import Optional
import re

## example text templates
message_entity=(
        "You have access to a knowledge graph that contains "
        "the entities which will be provided at the end of this message delimited with {delimiter} characters. "
        "Your task is to select the ones that are relevant to the user's question "
        "for subsequent use in a query. Only return the entities, "
        "comma-separated, without any additional text.\n"
        "{delimiter}\n"
        "{entities}"
    )

message_relationship=(
        "You have access to a knowledge graph that contains entities and relationships "
        "which will be provided at the end of this message delimited with {delimiter} characters. "
        "Your task is to select the relationships that are relevant "
        "to the user's question for subsequent use in a query. Only "
        "return the relationships without their sources or targets, "
        "comma-separated, and without any additional text. Here are the "
        "possible relationships and their source and target entities:\n"
        "{delimiter}\n"
        "entities:{selected_entities_joined}\n"
        "relationships: {relationships}"
    )

message_property=(
        "You have access to a knowledge graph that contains entities and "
        "relationships. Their properties are after the delimiter {delimiter} in the end of this message. "
        "Your task is to select the properties that are relevant to the "
        "user's question for subsequent use in a query. Only return the "
        "entities and relationships with their relevant properties in JSON "
        "format, without any additional text. Return the "
        "entities/relationships as top-level dictionary keys, and their "
        "properties as dictionary values. "
        "Do not return properties that are not relevant to the question.\n"
        "{delimiter}\n"
        "entities and their properties: {entity_properties}\n"
        "relationships and their properties: {relationship_properties}"
    )

message_query=(
        "Generate a database query in {query_language} that answers "
        "the user's question. "
        "You can use the following entities: {entities}, "
        "relationships: {relationships_key_list}, and "
        "properties: {properties}. "
    )


def extract_placeholders(text):
   
    pattern = r'{(.*?)}'
    matches = re.findall(pattern, text)
    return matches


class Prompt:
    def __init__(self, 
                 elements: Optional[dict] = None,
                 text_template: Optional[str] = None,
                 prompt_type: Optional[str] = None):
        self.elements = elements
        self.text_template = text_template
        self.prompt_type = prompt_type

    def __repr__(self):

        if not self.text_template:
            raise ValueError("please provide text template")
        if not self.elements:
            raise ValueError("please provide elements to fill in the text template")

        placeholders = extract_placeholders(self.text_template)
        if not all([ i in self.elements.keys() for i in placeholders ]):
            raise ValueError("Not all placeholders in text template provided in elements")

        return self.text_template.format(**self.elements)

class SystemPrompt(Prompt):
     def __init__(
        self,
        elements: Optional[dict] = None,
        text_template: Optional[str] = None,
        prompt_type: Optional[str] = 'system'
    ):
        super().__init__(
            elements=elements,
            text_template=text_template,
            prompt_type=prompt_type
        )