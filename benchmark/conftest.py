import os

import pandas as pd
import pytest

from benchmark.load_dataset import get_benchmark_dataset

RESULT_FILES = [
    "benchmark/results/biocypher_query_generation.csv",
    "benchmark/results/vectorstore.csv",
]

N_ITERATIONS = 1

BENCHMARK_DATASET = get_benchmark_dataset()


# TODO: adapt the docs
def pytest_addoption(parser):
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all benchmark tests from scratch",
    )


@pytest.fixture(autouse=True, scope="session")
def delete_results_csv_file_content(request):
    """
    If --run-all is set, the former benchmark data are deleted and all
    benchmarks are executed again.
    """
    if request.config.getoption("--run-all"):
        for f in RESULT_FILES:
            if os.path.exists(f):
                old_df = pd.read_csv(f, header=0)
                empty_df = pd.DataFrame(columns=old_df.columns)
                empty_df.to_csv(f, index=False)


@pytest.fixture(scope="session")
def result_files():
    result_files = {}
    for file in RESULT_FILES:
        try:
            result_files[file] = pd.read_csv(file, header=0)
        except pd.errors.EmptyDataError:
            result_files[file] = pd.DataFrame()
    return result_files


def pytest_generate_tests(metafunc):
    """pytest hook function to generate test cases.
    Called once for each test case in the benchmark test collection.
    If fixture is part of test declaration, the test is parametrized
    """
    if "test_data_biocypher_query_generation" in metafunc.fixturenames:
        data_file = BENCHMARK_DATASET[
            "./data/biocypher_query_generation/biocypher_query_generation.csv"
        ]
        metafunc.parametrize(
            "test_data_biocypher_query_generation", data_file.values
        )


def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
