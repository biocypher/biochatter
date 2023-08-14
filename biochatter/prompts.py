import yaml
from biocypher._misc import sentencecase_to_pascalcase


class BioCypherPrompts:
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
