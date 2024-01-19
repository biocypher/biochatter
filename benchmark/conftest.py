import os

import pytest

import numpy as np
import pandas as pd

from biochatter.prompts import BioCypherPromptEngine
from benchmark.load_dataset import get_benchmark_dataset
from biochatter.llm_connect import GptConversation, XinferenceConversation

from xinference.client import Client

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

XINFERENCE_MODELS = {
    # "llama-2-chat": {
    #     "model_size_in_billions": [
    #         7,
    #         # 13,
    #         # 70,
    #     ],
    #     "quantization": [
    #         "q2_K",
    #         # "q3_K_L",
    #         # "q3_K_M",
    #         # "q3_K_S",
    #         # "q4_0",
    #         # "q4_1",
    #         # "q4_K_M",
    #         # "q4_K_S",
    #         # "q5_0",
    #         # "q5_1",
    #         # "q5_K_M",
    #         # "q5_K_S",
    #         # "q6_K",
    #         "q8_0",
    #     ],
    # },
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

BENCHMARK_URL = "http://129.206.191.235:9997"


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
        _model_name, _model_size, _model_quantization = model_name.split(":")
        _model_size = int(_model_size)

        # get running models
        client = Client(base_url=BENCHMARK_URL)

        # TODO if all tests have already been executed, we should find out here,
        # not after the model has been launched

        # TODO why are we concatenating two instances of full model names for
        # all tests? (llama-2-chat:7:q2_K-llama-2-chat:7:q8_0 ...?)

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
    relevant_columns = []

    # Iterate over each row in the DataFrame
    for index, row in data_file.iterrows():
        if row["test_type"] == "biocypher_query_generation":
            if "test_data_biocypher_query_generation" in metafunc.fixturenames:
                # Parametrize the fixture with the relevant columns
                relevant_columns = [
                    "kg_path",
                    "prompt",
                    "entities",
                    "relationships",
                    "relationship_labels",
                    "properties",
                    "parts_of_query",
                    "test_case_purpose",
                    "index",
                ]
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
