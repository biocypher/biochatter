# Benchmark - All Results

## BioCypher query generation

In this set of tasks, we test LLM abilities to generate queries for a BioCypher Knowledge Graph using BioChatter.
The `schema_config.yaml` of the BioCypher Knowledge Graph and a natural language query are passed to BioChatter.

Individual steps of the query generation process are tested separately, as well as the end-to-end performance of the process.

=== "Entity selection"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/entity_selection.csv', colalign=("center","center","center","center","center")) }}

=== "Relationship selection"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/relationship_selection.csv', colalign=("center","center","center","center","center")) }}

=== "Property selection"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/property_selection.csv', colalign=("center","center","center","center","center")) }}

=== "Property hallucination"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/property_exists.csv', colalign=("center","center","center","center","center")) }}

=== "Query generation"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/query_generation.csv', colalign=("center","center","center","center","center")) }}

=== "End to end query generation"

    {{ read_csv('benchmark/results/preprocessed_for_frontend/end_to_end_query_generation.csv', colalign=("center","center","center","center","center")) }}

## Retrieval Augmented Generation (RAG)

In this set of tasks, we test LLM abilities to generate answers to a given question using a RAG agent, or to judge the relevance of a RAG fragment to a given question.
Instructions can be explicit ("is this fragment relevant to the question?") or implicit (just asking the question without instructions and evaluating whether the model responds with 'not enough information given').

=== "Explicit relevance of single fragments"

    TODO: description of rag_interpretation_test_explicit_relevance_of_single_fragments

    {{ read_csv('benchmark/results/preprocessed_for_frontend/explicit_relevance_of_single_fragments.csv', colalign=("center","center","center","center","center")) }}

=== "Implicit relevance of single fragments"

    TODO: description of rag_interpretation_test_implicit_relevance_of_multiple_fragments

    {{ read_csv('benchmark/results/preprocessed_for_frontend/implicit_relevance_of_multiple_fragments.csv', colalign=("center","center","center","center","center")) }}

## Semantic search

Coming soon.
