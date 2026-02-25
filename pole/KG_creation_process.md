# Explore dataset

- which classes are in there?

- how large is it?

- how heterogeneous is it?

# Find suitable ontology that

- maps classes most effectively

- has the correct scope

# Extraction and mapping

- from data, create node and edge classes (Enums) from unique entity classes

	- think about queries: what questions should be answered

	- select nodes and edges accordingly

- get_nodes and get_edges: functions that do ETL to provide biocypher with tuple
collections

	- take care to make as general as possible to use data structure as much as
	possible

	- extract type labels from data and pass through to configuration

- schema_config: specify the entities and relationships, and optionally
properties of each

	- define inheritance for entities not covered by the ontology directly

- possible automation: write only the yaml as a user, have the python adapter
generated from a template
