import os

import pytest

import numpy as np
import pandas as pd

from biochatter.prompts import BioCypherPromptEngine
from benchmark.load_dataset import get_benchmark_dataset
from biochatter.llm_connect import GptConversation, XinferenceConversation

from xinference.client import Client

# list all CSV files in the benchmark/results directory
RESULT_FILES = [
    f"benchmark/results/{file}"
    for file in os.listdir("benchmark/results")
    if file.endswith(".csv")
]

N_ITERATIONS = 2

BENCHMARK_DATASET = get_benchmark_dataset()


# set model matrix
# TODO should probably go to conftest.py
OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    # "gpt-4",
]

XINFERENCE_MODELS = {
    "llama-2-chat": {
        "model_size_in_billions": [
            7,
            13,
            # 70,
        ],
        "quantization": [
            "q2_K",
            # "q3_K_L",
            # "q3_K_M",
            # "q3_K_S",
            "q4_0",
            # "q4_1",
            # "q4_K_M",
            # "q4_K_S",
            "q5_0",
            # "q5_1",
            # "q5_K_M",
            # "q5_K_S",
            # "q6_K",
            "q8_0",
        ],
    },
    "mixtral-instruct-v0.1": {
        "model_size_in_billions": [
            46_7,
        ],
        "quantization": [
            "Q2_K",
            # "Q3_K_M",
            "Q4_0",
            # "Q4_K_M",
            "Q5_0",
            # "Q5_K_M",
            # "Q6_K",
            "Q8_0",
        ],
    },
}

# create concrete benchmark list by concatenating all combinations of model
# names, model sizes and quantizations
XINFERENCE_MODEL_NAMES = [
    f"{model_name}:{model_size}:{quantization}"
    for model_name in XINFERENCE_MODELS.keys()
    for model_size in XINFERENCE_MODELS[model_name]["model_size_in_billions"]
    for quantization in XINFERENCE_MODELS[model_name]["quantization"]
]

BENCHMARKED_MODELS = OPENAI_MODEL_NAMES + XINFERENCE_MODEL_NAMES
BENCHMARKED_MODELS.sort()

BENCHMARK_URL = "http://localhost:9997"


# parameterise tests to run for each model
@pytest.fixture(params=BENCHMARKED_MODELS)
def model_name(request):
    return request.param


@pytest.fixture
def multiple_testing(request):
    def run_multiple_times(test_func, *args, **kwargs):
        scores = []
        for _ in range(N_ITERATIONS):
            score, max = test_func(*args, **kwargs)
            scores.append(score)
        mean_score = sum(scores) / N_ITERATIONS
        return (mean_score, max, N_ITERATIONS)

    return run_multiple_times


def calculate_test_score(vector: list[bool]) -> tuple[int, int]:
    score = sum(vector)
    max = len(vector)
    return (score, max)


@pytest.fixture
def prompt_engine(request, model_name):
    """
    Generates a constructor for the prompt engine for the current model name.
    """

    def setup_prompt_engine(kg_schema_path):
        return BioCypherPromptEngine(
            schema_config_or_info_path=kg_schema_path,
            model_name=model_name,
        )

    return setup_prompt_engine


@pytest.fixture
def conversation(request, model_name):
    """
    Decides whether to run the test or skip due to the test having been run
    before. If not skipped, will create a conversation object for interfacing
    with the model.
    """
    test_name = request.node.originalname.replace("test_", "")
    result_file = f"benchmark/results/{test_name}.csv"

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
        _model_name, _model_size, _model_quantization = model_name.split(":")
        _model_size = int(_model_size)

        # get running models
        client = Client(base_url=BENCHMARK_URL)

        # if exact model already running, return conversation
        running_models = client.list_models()
        if running_models:
            for running_model in running_models:
                if (
                    running_models[running_model]["model_name"] == _model_name
                    and running_models[running_model]["model_size_in_billions"]
                    == _model_size
                    and running_models[running_model]["quantization"]
                    == _model_quantization
                ):
                    conversation = XinferenceConversation(
                        base_url=BENCHMARK_URL,
                        model_name=_model_name,
                        prompts={},
                        correct=False,
                    )
                    return conversation

        # else, terminate all running models
        for running_model in running_models:
            client.terminate_model(running_model)

        # and launch model to be tested
        client.launch_model(
            model_name=_model_name,
            model_size_in_billions=_model_size,
            quantization=_model_quantization,
        )

        # return conversation
        conversation = XinferenceConversation(
            base_url=BENCHMARK_URL,
            model_name=_model_name,
            prompts={},
            correct=False,
        )

    return conversation


@pytest.fixture
def evaluation_conversation():
    conversation = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        correct=False,
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="benchmark_user")
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
            result_file = pd.DataFrame(
                columns=["model_name", "subtask", "score"]
            )
            result_file.to_csv(file, index=False)

        if not np.array_equal(
            result_file.columns, ["model_name", "subtask", "score"]
        ):
            result_file.columns = ["model_name", "subtask", "score"]

        result_files[file] = result_file

    return result_files


def pytest_generate_tests(metafunc):
    """
    Pytest hook function to generate test cases.
    Called once for each test case in the benchmark test collection.
    If fixture is part of test declaration, the test is parametrized.
    """
    # Load the data file
    data_file = BENCHMARK_DATASET["./data/benchmark_data.csv"]
    data_file["index"] = data_file.index

    # Initialize a dictionary to collect rows for each test type
    test_rows = {
        "biocypher_query_generation": [],
        "rag_interpretation": [],
        "text_extraction": [],
    }

    # Iterate over each row in the DataFrame
    for index, row in data_file.iterrows():
        test_type = row["test_type"]
        if test_type in test_rows:
            # Add the row to the list for this test type
            test_rows[test_type].append(row)

    # Parametrize the fixtures with the collected rows
    if "test_data_biocypher_query_generation" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_biocypher_query_generation",
            test_rows["biocypher_query_generation"],
        )
    if "test_data_rag_interpretation" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_rag_interpretation",
            test_rows["rag_interpretation"],
        )
    if "test_data_text_extraction" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_text_extraction",
            test_rows["text_extraction"],
        )
