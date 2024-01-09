import os

import pytest

import numpy as np
import pandas as pd

from biochatter.prompts import BioCypherPromptEngine
from benchmark.load_dataset import get_benchmark_dataset
from biochatter.llm_connect import GptConversation, XinferenceConversation

RESULT_FILES = [
    "benchmark/results/biocypher_query_generation.csv",
    "benchmark/results/vectorstore.csv",
]

N_ITERATIONS = 1

BENCHMARK_DATASET = get_benchmark_dataset()


# set model matrix
# TODO should probably go to conftest.py
OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    # "gpt-4",
]

XINFERENCE_MODEL_NAMES = [
    # "llama2-hf",
    # "llama2-chat-hf",
]

BENCHMARKED_MODELS = OPENAI_MODEL_NAMES + XINFERENCE_MODEL_NAMES

BENCHMARK_URL = "http://llm.biocypher.org"


@pytest.fixture(scope="module", params=BENCHMARKED_MODELS)
def prompt_engine(request):
    def setup_prompt_engine(kg_schema_path):
        model_name = request.param
        return BioCypherPromptEngine(
            schema_config_or_info_path=kg_schema_path,
            model_name=model_name,
        )

    return setup_prompt_engine


@pytest.fixture(scope="function", params=BENCHMARKED_MODELS)
def conversation(request):
    model_name = request.param
    if model_name in OPENAI_MODEL_NAMES:
        conversation = GptConversation(
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            os.getenv("OPENAI_API_KEY"), user="benchmark_user"
        )
    elif model_name in XINFERENCE_MODEL_NAMES:
        # TODO here we probably need to start the right model on the server
        conversation = XinferenceConversation(
            base_url=BENCHMARK_URL,
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key()

    return conversation


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
    if "test_data_biocypher_query_generation" in metafunc.fixturenames:
        data_file = BENCHMARK_DATASET[
            "./data/biocypher_query_generation/biocypher_query_generation.csv"
        ]
        data_file["index"] = data_file.index
        metafunc.parametrize(
            "test_data_biocypher_query_generation", data_file.values
        )


def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
