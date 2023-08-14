import yaml
import os
from biocypher._misc import sentencecase_to_pascalcase
from .llm_connect import GptConversation


class BioCypherPrompt:
    def __init__(self, schema_config_path: str):
        """

        Given a biocypher schema configuration, extract the entities and
        relationships, and for each extract their mode of representation (node
        or edge), properties, and identifier namespace. Using these data, allow
        the generation of prompts for a large language model, informing it of
        the schema constituents and their properties, to enable the
        parameterisation of function calls to a knowledge graph.

        Args:
            schema_config_path: Path to a biocypher schema configuration file.

        """
        # read the schema configuration
        with open(schema_config_path, "r") as f:
            schema_config = yaml.safe_load(f)

        # extract the entities and relationships: each top level key that has
        # a 'represented_as' key
        self.entities = {}
        self.relationships = {}
        for key, value in schema_config.items():
            # hacky, better with biocypher output
            name_indicates_relationship = (
                "interaction" in key.lower() or "association" in key.lower()
            )
            if "represented_as" in value:
                if (
                    value["represented_as"] == "node"
                    and not name_indicates_relationship
                ):
                    self.entities[sentencecase_to_pascalcase(key)] = value
                elif (
                    value["represented_as"] == "node"
                    and name_indicates_relationship
                ):
                    self.relationships[sentencecase_to_pascalcase(key)] = value
                elif value["represented_as"] == "edge":
                    self.relationships[sentencecase_to_pascalcase(key)] = value

        self.selected_entities = []
        self.selected_relationships = []

    def select_entities(self, question: str) -> bool:
        """

        Given a question, select the entities that are relevant to the question
        and store them in `selected_entities` and `selected_relationships`. Use
        LLM conversation to do this.

        Args:
            question: A user's question.

        Returns:
            True if at least one entity was selected, False otherwise.

        """

        conversation = GptConversation(
            model_name="gpt-3.5-turbo",
            prompts={},
            correct=False,
        )

        conversation.set_api_key(
            api_key=os.getenv("OPENAI_API_KEY"), user="entity_selector"
        )

        conversation.append_system_message(
            (
                "You have access to a knowledge graph that contains "
                f"these entities: {', '.join(self.entities)} and these "
                f"relationships: {', '.join(self.relationships)}. Your task is "
                "to select the ones that are relevant to the user's question "
                "for subsequent use in a query. Only return the entities and "
                "relationships, comma-separated, without any additional text. "
                "If you select relationships, make sure to also return "
                "entities that are connected by those relationships."
            )
        )

        msg, token_usage, correction = conversation.query(question)

        result = msg.split(",") if msg else []
        # TODO: do we go back and retry if no entities were selected? or ask for
        # a reason? offer visual selection of entities and relationships by the
        # user?

        if result:
            for entity_or_relationship in result:
                if entity_or_relationship in self.entities:
                    self.selected_entities.append(entity_or_relationship)
                elif entity_or_relationship in self.relationships:
                    self.selected_relationships.append(entity_or_relationship)

        # check for optional edge labels of relationships
        for relationship in self.selected_relationships:
            if (
                "label_as_edge"
                in self.relationships.get(relationship, {}).keys()
            ):
                # replace selected_relationship with the edge label
                self.selected_relationships.remove(relationship)
                self.selected_relationships.append(
                    self.relationships[relationship]["label_as_edge"]
                )

        return bool(result)
