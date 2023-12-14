#!/usr/bin/env Python3

from typing import Optional


class message(str):

    def __init__(
            self,
            label: Optional[str] = None,
            message_type: Optional[str] = None,

    ):
       
        super().__init__()
        self.label = label
        self.message_type = message_type
        
        


class system_message(message):

    def __init__(
            self,

    ):
        super().__init__()
    

msg = (
            "You have access to a knowledge graph that contains entities and "
            "relationships. They have the following properties. Entities:"
            f"{e_props}, Relationships: {r_props}. "
            "Your task is to select the properties that are relevant to the "
            "user's question for subsequent use in a query. Only return the "
            "entities and relationships with their relevant properties in JSON "
            "format, without any additional text. Return the "
            "entities/relationships as top-level dictionary keys, and their "
            "properties as dictionary values. "
            "Do not return properties that are not relevant to the question."
        )


msg = (
            "You have access to a knowledge graph that contains entities and "
            "relationships, their associated properties are given after the deliminator ####"
            "in the end of this message."
            "Your task is to select the properties that are relevant to the "
            "user's question for subsequent use in a query. Only return the "
            "entities and relationships with their relevant properties in JSON "
            "format, without any additional text. Return the "
            "entities/relationships as top-level dictionary keys, and their "
            "properties as dictionary values of type list. "
            "Do not return properties that are not relevant to the question."
            "Do not invent a property that is not one of the given properties"
            "if you cannot find a relevant property, consider whether the user question is about the id"
            "it is possible to refer to the id in similar ways as a property"
        )
