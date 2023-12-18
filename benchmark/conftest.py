import os

import pytest

from benchmark.load_dataset import get_benchmark_dataset

RESULT_FILES = [
    "benchmark/results/biocypher_query_generation.csv",
    "benchmark/results/vectorstore.csv",
]

N_ITERATIONS = 1


@pytest.fixture(autouse=True, scope="session")
def delete_csv_files():
    """
    Reset benchmark output each time pytest is run.

    Todo:

        Probably not the most economic way to delete everything every time,
        should be extended to only overwrite the tests that have changed or add
        models that were not present before.

    """
    for f in RESULT_FILES:
        if os.path.exists(f):
            os.remove(f)

    # create blank CSV files
    for f in RESULT_FILES:
        with open(f, "w") as f:
            f.write("")


def pytest_generate_tests(metafunc):
    """pytest hook function to generate test cases.
    Called once for each test case in the benchmark test collection.
    If fixture is part of test declaration, the test is parametrized
    """
    benchmark_dataset = get_benchmark_dataset()

    if "test_data_biocypher_query_generation" in metafunc.fixturenames:
        data_file = benchmark_dataset["./data/biocypher_query_generation/biocypher_query_generation.csv"]
        metafunc.parametrize("test_data_biocypher_query_generation",
                             data_file.values)


def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
