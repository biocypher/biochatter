import os

import pytest

import numpy as np
import pandas as pd

from benchmark.load_dataset import get_benchmark_dataset

RESULT_FILES = [
    "benchmark/results/biocypher_query_generation.csv",
    "benchmark/results/vectorstore.csv",
]

N_ITERATIONS = 1

BENCHMARK_DATASET = get_benchmark_dataset()


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
            result_file = pd.read_csv(file, header=0)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            result_file = pd.DataFrame(columns=["model", "subtask", "score"])
            result_file.to_csv(file, index=False)

        if not np.array_equal(
            result_file.columns, ["model", "subtask", "score"]
        ):
            result_file.columns = ["model", "subtask", "score"]

        result_files[file] = result_file

    return result_files


def pytest_generate_tests(metafunc):
    """
    Pytest hook function to generate test cases.
    Called once for each test case in the benchmark test collection.
    If fixture is part of test declaration, the test is parametrized.
    """
    # Load the data file
    data_file = pd.read_csv("./data/test_data.csv")
    data_file["index"] = data_file.index
    # should be BENCHMARK_DATASET["./data/test_data.csv"] ?

    # Iterate over each row in the DataFrame
    for index, row in data_file.iterrows():
        # Check the type of the test case
        if row["test_type"] == "biocypher_query_generation":
            # If the test function requires the "test_data_biocypher_query_generation" fixture
            if "test_data_biocypher_query_generation" in metafunc.fixturenames:
                # Parametrize the fixture with the relevant columns
                metafunc.parametrize(
                    "test_data_biocypher_query_generation",
                    [row[relevant_columns]],
                )
        elif row["test_type"] == "rag_functionality":
            # If the test function requires the "test_data_rag_functionality" fixture
            if "test_data_rag_functionality" in metafunc.fixturenames:
                # Parametrize the fixture with the relevant columns
                metafunc.parametrize(
                    "test_data_rag_functionality", [row[relevant_columns]]
                )
        elif row["test_type"] == "text_extraction":
            # If the test function requires the "test_data_text_extraction" fixture
            if "test_data_text_extraction" in metafunc.fixturenames:
                # Parametrize the fixture with the relevant columns
                metafunc.parametrize(
                    "test_data_text_extraction", [row[relevant_columns]]
                )


def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
