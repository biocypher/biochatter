# Connecting Knowledge Graphs

To increase accessibility of databases, we can leverage the
[BioCypher](https://biocypher.org) integration of BioChatter.  In BioCypher, we
use a YAML configuration (`schema_config.yaml`) to specify the contents of the
knowledge graph and their ontological associations.  We also generate a more
extensive, but essentially similar YAML file during the BioCypher creation of a
knowledge graph (`schema_info.yaml`), which contains more information pertinent
to LLM interaction with the database.  The current prototypical implementation
of query generation through an LLM is implemented in the `prompts.py` module on
the example of a Neo4j knowledge graph connection.

## Connecting

Currently, BioChatter does not handle database connectivity, but simply returns
a query for a given language.  The application using BioChatter should establish
connectivity and send the query to the database, as is implemented in ChatGSE,
for instance.  For a demonstration using a simple Docker compose setup, see the
[Pole Crime Dataset demo repository](https://github.com/biocypher/pole).

## Querying

The generation of a query based on BioCypher configuration files is a multi-step
process.  This is partly to account for the limited token input space of some
models, and partly to better be able to test and compare the individual steps.
We will implement a wrapper function that goes through the steps automatically
soon, but for now the steps need to be run individually.

### Setup

We use the `BioCypherPromptEngine` class to handle the LLM conversation.

```python
from biochatter.prompts import BioCypherPromptEngine
prompt_engine = BioCypherPromptEngine(
    schema_config_or_info_path="test/schema_info.yaml"
)
```

This will load the `schema_config.yaml` or `schema_info.yaml` (preferred) file
and set up the conversation.

### Query generation

Using the `generate_query` wrapper, we can generate a query from a question and
a database language.

```python
query = prompt_engine.generate_query(
    question="Which genes are associated with mucoviscidosis?",
    database_language="Cypher",
)
```

This will return a query that can be used in the database query language (e.g.,
Cypher).  This end to end process executes the steps detailed below, namely,
entity selection, relationship selection, and property selection, as well as the
generation of the final query using the selected components. You can run each of
these steps individually, if you want.

#### Entity selection

Starting from the `schema_config.yaml` or `schema_info.yaml` (preferred) file,
we first have the model decide which entities in the database are relevant to
the user's question.

```python
success = prompt_engine._select_entities(
    question="Which genes are associated with mucoviscidosis?"
)
```

This will select a number of entities from the database schema to be used
subsequently, and return True or False to indicate success.

#### Relationship selection

Next, we will use the entities determined in the first step to select
relationships between them.  The entities selected in the first step will be
stored in the `selected_entities` attribute of the `BioCypherPromptEngine`
instance, and the question is stored in the `question` attribute.  Both are
automatically used to select relationships.

```python
success = prompt_engine._select_relationships()
```

#### Property selection

To not unnecessarily waste token input space, we are only interested in
selecting properties of entities that are of interest given the question asked.
We do so in the third step, which uses the entities and relationships determined
in the first steps.  Again, `question`, `selected_entities`, and
`selected_relationships` are automatically used to select properties.

```python
success = prompt_engine._select_properties()
```

This will select a number of properties to be used in the query, and also return
True or False to indicate success.

#### Query generation

Finally, we can use the entities and relationships, as well as the selected
properties, to ask the LLM to generate a query in the desired language.

```python
query = prompt_engine._generate_query(
    question="Which genes are associated with mucoviscidosis?",
    entities=["Gene", "Disease"],
    relationships=["GeneToDiseaseAssociation"],
    properties={"Disease": ["name", "ICD10", "DSM5"]},
    database_language="Cypher",
)
```

This will (hopefully) return a query that can be used in the database query
language (e.g., Cypher).

### Query interaction
As an optional follow-up, you can interact with the returned query using the 
`BioCypherQueryHandler` class (`query_interaction.py`). It takes the query, 
the original question and the KG information so that the interaction is still aware of the KG. 

```python
from biochatter.query_interaction import BioCypherQueryHandler
query_handler = BioCypherQueryHandler(
    query=query,
    query_lang="Cypher",
    kg_selected={
        entities: ["Gene", "Disease"],
        relationships: ["GeneToDiseaseAssociation"],
        properties: {"Disease": ["name", "ICD10", "DSM5"]}
    },
    question="Which genes are associated with mucoviscidosis?"
)
```

#### Explanation
You can retrieve an explanation of the returned query with:

```python
explanation = query_handler.explain_query()
```

#### Updating
Alternatively, you can ask the LLM for an update of the query with:

```python
request = "Only return 10 results and sort them alphabetically"
explanation = query_handler.update_query(request)
```

NB: for updates, it might sometimes be relevant that all the KG 
enitites/relationships/properties are known to the LLM instead 
of only those that were selected to be relevant for the original question.
For this, you can optionally pass them as input to the query handler 
with `kg` (similar to `kg_selected`).

(Tip: the prompt_engine object contains both the selected and non-selected as attributes)
