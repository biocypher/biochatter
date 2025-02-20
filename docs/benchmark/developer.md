# Benchmark - Developer Notes

To understand the benchmarking procedure, you should be familiar with
[Pytest](https://docs.pytest.org/en/). The benchmark test matrix is executed via
Pytest fixtures that iterate through the combinations of test parameters such as
model name and size.  This basic setup happens in the `conftest.py` file in the
`benchmark` directory.  The benchmark Pytest setup is distinct from the Pytest
setup we use for our continuous integration (in the `test` folder).

You can imagine the benchmark as a set of unit tests, with the only difference
being that the test subject is not our codebase, but the behaviour and
performance of the various LLMs, prompts, etc. These tests are defined in two
parts: the data and the method. Data are collected in a set of YAML files in the
`benchmark/data` directory, while the benchmark methods are implemented in the
Pytest functions in the individual Python modules (file names starting with
`test_`). We also have some Python modules for auxiliary functions, such as
`load_dataset.py`.

In the following, we will describe a walkthrough of how to implement your own
tests according to our benchmark philosophy.

## Test setup

Tests are collected in the typical Pytest manner at the start of the procedure.
In `conftest.py`, we define the model combinations we want to use in the
benchmark.  We distinguish between closed-source and open-source models, since
open-source models offer more flexibility, e.g., by setting their size and
quantisation. In contrast, for OpenAI models, all we need is the name.

### Quickstart

For getting started with developing your own benchmark, OpenAI models offer the
simplest way, only requiring an API key and an internet connection. If you don't
want to run open-source models right away, which is tied to setting up an
external service for deploying the models, we can remove the
`XINFERENCE_MODEL_NAMES` from the list of models to be benchmarked in
`conftest.py` (by deleting or commenting them out):

```python
BENCHMARKED_MODELS = OPENAI_MODEL_NAMES
```

In addition, we can reduce the number of OpenAI models to call to one for
development; `gpt-3.5-turbo-0125` is a well-performing and economical initial
choice (in `conftest.py`).

```python
OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo-0125",
]
```

The last thing to look out for when running the benchmark is to reduce the
number of iterations for each test to one. We run iterations to account for
stochasticity in LLM responses when we run the benchmark for real, but in
development, this iteration brings no benefit and just increases computational
cost. Set `N_ITERATIONS` to 1 in `conftest.py`.

```python
# how often should each benchmark be run?
N_ITERATIONS = 1
```

This setup should allow you to run and debug a newly developed benchmark dataset
or method effectively. For more explanation on how to do this, please read on.

## Debugging the benchmark

You can get some insight into how the benchmark works by debugging the existing
test cases and stepping through the code line-by-line. For this, it is necessary
that you are familiar with the debugging procedure in your programming
environment of choice, for instance,
[VSCode](https://code.visualstudio.com/docs/editor/debugging). You can set
breakpoints in the initial setup (e.g., in `conftest.py` and `load_dataset.py`)
as well as the test functions (e.g., `test_rag_interpretation.py`). Stepping
through the code will give you insights into how the benchmark is designed and
also how the LLMs respond in detail to each specific task. This is particularly
helpful for ensuring that your newly developed benchmark test cases behave as
expected and test accurately the functionality you aim to test.

## Creating new test cases for existing tests

Our test cases are collected in YAML files that follow a simple formalism for
defining each test. These files are found in `benchmark/data` and need to end in
`_data.yaml` in order to be loaded in the test procedure. They include test
cases and auxiliary materials, such as knowledge graph schemata. A test case
consists of

- a descriptive name

- a set of input data to simulate a real-world task (e.g., the question of a
user and some auxiliary information)

- a set of expected results to assess the performance of the model's response,
akin to assert statements in regular unit tests

Here is a simple example test case:

```yaml
rag_interpretation:
# test simple relevance judgement
  - case: explicit_relevance_yes
    input:
      prompt: Which molecular pathways are associated with cancer?
      system_messages:
        [
          "You will receive a text fragment to help answer the user's question. Your task is to judge this text fragment for  relevance to the user's question, and return either 'yes' or 'no'; only respond with one word, do not offer explanation  or justification! Here is the fragment: ",
          "The EGFR pathway is deregulated in a number of cancers.",
        ]
    expected:
      answer: "yes"
```

In this test, we benchmark the model's ability to judge the relevance of text
fragments to be used in a Retrieval-Augmented Generation (RAG) scenario in an
explicit fashion (i.e., we directly ask the model to judge the relevance of the
given fragments). Thus, we choose the descriptive name (`case`)
`explicit_relevance_yes`. The `input`s are a simulated user question (`prompt`)
and the `system_messages` that simulate the prompt engineering and RAG fragments
(that in the real application would be retrieved by some suitable mechanism).
Finally, we define the `expected` output, which in this case is only the
`answer` we expect from the LLM.

The way that these test cases are evaluated is defined in the Pytest functions,
which are tuned to the specific fields we define in the YAML. We can freely
define test definitions and testing implementation by adjusting the YAML
alongside the test Python code. The coordination between cases and Pytest
functions is done by name of the test category, in this example
`rag_interpretation` in the YAML and `test_rag_interpretation.py` in the
`benchmark` directory.

New tests can be arbitrarily complex as long as the test function is
synchronised with the content of the YAML test dataset's fields (see below).

### Combinatorial expansion of tests

Sometimes, it can be efficient to run the same test with slightly different
variations of input data. For instance, LLMs are very sensitive to the prompt
setup, and thus, we can run the same test with varying approaches to the prompt
engineering. To simplify this, we can define input data as dictionaries (with
keys being a descriptive name of the sub-test, and values being the content),
which will lead to expansion of these test cases into full cases according to
the definition. For instance, we can define a RAG interpretation test with
prompts at three different levels of detail (`simple`, `more_explicit`, and
`repeat_instruction`):

```yaml
rag_interpretation:
  # test simple irrelevance judgement
  - case: explicit_relevance_no
    input:
      prompt: Which molecular pathways are associated with cancer?
      system_messages:
        simple:
          [
            "You will receive a text fragment to help answer the user's question. Your task is to judge this text fragment for relevance to the user's question, and return either 'yes' or 'no'! Here is the fragment: ",
            "The earth is a globe.",
          ]
        more_explicit:
          [
            "You will receive a text fragment to help answer the user's question. Your task is to judge this text fragment for relevance to the user's question, and return either 'yes' or 'no'; only respond with one word, do not offer explanation or justification! Here is the fragment: ",
            "The earth is a globe.",
          ]
        repeat_instruction:
          [
            "You will receive a text fragment to help answer the user's question. You should only respond with 'yes' or 'no' without additional words. Your task is to judge this text fragment for relevance to the user's question, and return either 'yes' or 'no'; only respond with one word, do not offer explanation or justification! Here is the fragment: ",
            "The earth is a globe.",
          ]
    expected:
      answer: "no"
```

Upon instantiation of the test matrix, this definition will be expanded into
three full tests, each with their respective prompt setup. You can define as
many combinations as you like (for instance, you could also define a list of
prompts in this example), but be aware that the number of tests will grow
exponentially with the number of combinations.

## Setting up the test data pipeline

Test data are provided to the test functions via fixtures. The fixtures are
defined in the `conftest.py` file and are used to load the test data from the
YAML files. If you add a new test module or a function with a new kind of test
data, you need to add the corresponding fixture to the `pytest_generate_tests`
function in `conftest.py`. This function is responsible for loading the test
data and providing appropriately named fixtures to the test functions. For the
tests defined above, this equates to:

```python
def pytest_generate_tests(metafunc):
    data = BENCHMARK_DATASET
    if "test_data_rag_interpretation" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_rag_interpretation",
            data["rag_interpretation"],
        )
```

We prepend the fixtures with `test_data_` for consistency and higher code
readability. For more information, see the [Pytest
Documentation](https://docs.pytest.org/en/latest/example/parametrize.html).

## Creating new test procedures

If a new kind of test requires a bespoke procedure, such as evaluating a newly
introduced functionality or calculating a score in a distinct way, we can
introduce new methods to the test modules or even entire new modules. Following
the layout of the existing tests, the newly created test functions should refer
to fixtures for their data inputs.  Such a test function typically has as
parameters:

- the `model_name` fixture, to be able to record a model-specific benchmark
metric;

- a `test_data` object that is generated from the benchmark dataset according to
the name of the test module (e.g., `test_data_rag_interpretation`). This is the
fixture you defined above in `conftest.py`;

- a `conversation` instance (the connection to the LLM to be tested);

- the `multiple_testing` fixture that implements running the test multiple times
and averaging the results;

- any number of additional inputs that may be required for the tests.

For instance, the knowledge graph query generation tests acquire additional
tests inputs from the YAML definition (the schema of the BioCypher knowledge
graph underlying the test) and additional functionality from BioChatter (an
instance of the prompt engine class that generates the knowledge graph query
using the aforementioned schema).

## Running the benchmark

If everything is set up correctly, you can run the benchmark by executing the
following command in the root directory of the repository:

```bash
poetry run pytest benchmark
```

We need to specify the `benchmark` directory to run the benchmark tests, because
we also have regular tests in the `test` directory. If you want to run only a
specific test module, you can specify the file name (or use any other Pytest
workflow).

!!! warning "Skipping tests"

    For efficiency reasons, we by default do not rerun tests that have already
    been executed for a given model and test case. For this purpose, we store
    the results in the `benchmark/results` directory, including, for every test
    case, an md5 hash of the input data. If you want to rerun a test, you can
    delete the corresponding line (or entire file) in the `results` directory.

We re-run the benchmark automatically if a test case has changed (reflected in
a different md5 hash) or if there is a new `biochatter` version (potentially
introducing changes in the behaviour of the framework). If a test case has
changed, the old result is automatically removed from the result files. You can
also force a rerun of all tests by using the `--run-all` flag:

```bash
poetry run pytest benchmark --run-all
```

## Processing benchmark results

The benchmark results are processed to generate visualizations and statistics
that are displayed in the documentation. This processing used to happen during
the documentation build but has been moved to a separate workflow for better
efficiency:

1. The processing script is located at `docs/scripts/hooks.py` and can be run
directly:

    ```bash
    python docs/scripts/hooks.py
    ```

2. The processing is automated through a GitHub Action that runs when:

    - Changes are pushed to the `main` branch that affect benchmark results

    - Pull requests targeting `main` include benchmark result changes
   
    - Manual triggers via workflow dispatch in GitHub Actions

This separation means that the heavy processing of benchmark results only
happens when the results actually change, rather than on every documentation
build.

## Running open-source models

To execute the benchmark on any of the open-source models in the test matrix,
you need to deploy an [Xorbits
Inference](https://inference.readthedocs.io/en/latest/) server at an arbitrary
IP, either via [Docker](https://www.docker.com) (available on Linux machines
with dedicated Nvidia GPU) or natively (e.g., on Apple machines). Please refer
to the Xinference
[documentation](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)
for details.

When you have deployed the Xinference server, you can point the benchmark
to the server by setting the `BENCHMARK_URL` parameter in `conftest.py`:

```python
# Xinference IP and port
BENCHMARK_URL = "http://localhost:9997"
```
