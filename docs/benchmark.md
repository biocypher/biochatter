# Benchmarking

For trustworthy application of LLMs to real-world and biomedical problems, it is imperative to understand their performance and limitations.
We need to constantly evaluate the multitude of combinations of individual models and versions, their parameters (e.g., temperature), prompt sets, databases and vector databases, and diverse application scenarios.
To this end, we are maintaining a living benchmarking framework that allows us to continuously compare the performance of different models and configurations on a variety of tasks.

The benchmark uses the pytest framework to orchestrate the evaluation of a number of models on a number of tasks.
The benchmark is run on a regular basis, and the results are published on the [BioChatter website](https://biocypher.github.io/biochatter/benchmark/#results).
(Currently in development.)
The benchmarking suite can be found in the `benchmark` directory of the BioChatter repository.
It can be executed using standard pytest syntax, e.g., `poetry run pytest benchmark`.
As default behavior it checks, which test cases have already been executed and only executes the tests that have not been executed yet.
To run all benchmarks again, use `poetry run pytest benchmark --run-all`.

To allow flexible extension of the benchmark, we have implemeted a modular test framework that uses pytest fixtures to allow easy addition of new models and tasks.
All setup is done in the `conftest.py` file in the `benchmark` directory.
The result files are simple CSVs whose file names are generated from the name of the test function; they can be found in `benchmark/results` and contain scores for all executed combination of parameters.

To achieve modularity, we use pytest fixtures and parametrization.
For instance, to add a new model, we can modify the `MODEL_NAMES` list in the query generation test module, or the `EMBEDDING_MODELS` and `CHUNK_SIZES` lists in the vector database test module.
The environment that runs the benchmark needs to make available all prerequisites for the different modules.
For instance, the tasks requiring connection to an LLM need to provide the necessary credentials and API keys, or a connection to a self-hosted model.
Likewise, the benchmarks of retrieval augmented generation (RAG) processes require a connection to the RAG agent, e.g., a vector database.

## Calibration

To ensure valid assessment of LLM performance, we need to ensure calibration and technical validity of the benchmarking framework.
More recent LLMs in particular may be problematic when using publicly available benchmark datasets, since they could have been used for training the model.
This is particularly relevant in closed-source (e.g., OpenAI) models.
Thus, we need to consider strategies for avoiding contamination, such as hand-crafting datasets, carefully testing for contamination, or using perturbation strategies to generate new datasets from existing ones.
Advanced scenarios could utilise LLMs as "examiners," allowing more flexible test design and free-form answers.
There is much research into these phenomena, all of which should be considered in the maintenance of this testing framework.

## Aspects of benchmarking

In the following, we will detail the different aspects of benchmarking that we are currently testing.
This is a living document that will be updated as we add new tests and test modules.

### Models

Naturally the biggest impact on BioChatter performance comes with the model used.
However, model versions can have a significant impact, which can be obfuscated by the fact that model names are often not unique.
For instance, OpenAI's GPT models often have versions with significantly diverging capabilities and performance.
[etc]

### Prompts

As has been recently studied extensively, prompt engineering can make or break the performance of a model on a given task.
As such, it is important to test the default prompts we commonly use, as well as a range of variations to determine factors of prompt performance and robustness.
As an added complexity, LLMs are often used to generate prompts, which theoretically allows for procedural generation of an infinite number of prompts, as long as time and resources allow.

### Model parameters

The parameters of the model can have a significant impact on the performance of the model.
We often set model temperature to 0 to provide consistent results, but some applications may benefit from a higher temperature.
In testing, we mostly rely on a temperature of 0 due to the complexity of testing highly variable results in most cases.

### Databases

An important facet of BioChatter and BioCypher is their combination in querying databases.
This helps to ameliorate the limitations of LLMs by providing structured and validated knowledge to counteract hallucinations.
To ensure the seamless interaction of BioChatter and BioCypher, we need to test the performance of BioChatter on a variety of databases.

### Vector databases

Similarly to regular databases, vector databases are an important tool to provide validated knowledge to LLMs.
Vector databases bring their own set of parameters and application scenarios, which likewise need to be tested.
For instance, the length and overlap of fragments, the embedding algorithms, as well as the semantic search algorithms applied can have an impact on LLM conversation performance.

### Tasks

There is a wide range of tasks that are potentially useful to BioChatter users.
To cover most scenarios of research and development use, as well as clinical applications, we test a variety of tasks and LLM personas.

## Results

### BioChatter for BioCypher query generation

In this task BioChatter is used to generate queries for a BioCypher Knowledge Graph.
The `schema_config.yaml` of the BioCypher Knowledge Graph and a natural language query are passed to BioChatter.

Benchmarking results for BioChatter as a query generator for BioCypher:

{{ read_csv('benchmark/results/preprocessed_for_frontend/biocypher_query_generation_test_end_to_end_query_generation.csv', colalign=("center","center","center","center")) }}

### Retrieval Augmented Generation (RAG)

TODO: description of rag_interpretation_test_explicit_relevance_of_single_fragments

{{ read_csv('benchmark/results/preprocessed_for_frontend/rag_interpretation_test_explicit_relevance_of_single_fragments.csv', colalign=("center","center","center","center")) }}

TODO: description of rag_interpretation_test_implicit_relevance_of_multiple_fragments

{{ read_csv('benchmark/results/preprocessed_for_frontend/rag_interpretation_test_implicit_relevance_of_multiple_fragments.csv', colalign=("center","center","center","center")) }}

### Semantic search

tbd
