import os

import numpy as np
import pandas as pd
import pytest
import requests
from xinference.client import Client

from biochatter.llm_connect import (
    AnthropicConversation,
    GptConversation,
    XinferenceConversation,
)
from biochatter.prompts import BioCypherPromptEngine

from .benchmark_utils import benchmark_already_executed
from .load_dataset import get_benchmark_dataset

# how often should each benchmark be run?
N_ITERATIONS = 1

# which dataset should be used for benchmarking?
BENCHMARK_DATASET = get_benchmark_dataset()

# which models should be benchmarked?
OPENAI_MODEL_NAMES = [
    # "gpt-3.5-turbo-0125",
    # "gpt-4-0613",
    # "gpt-4-0125-preview",
    # "gpt-4-turbo-2024-04-09",
    # "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    # "gpt-4o-mini-2024-07-18",
]

ANTHROPIC_MODEL_NAMES = [
    # "claude-3-5-sonnet-20240620",
    # "claude-3-opus-20240229",  # does not make a lot of sense to use, as more expensive and "less intelligent" (according to anthropic)
]

XINFERENCE_MODELS = {
    # "code-llama-instruct": {
    #     "model_size_in_billions": [
    #         # 7,
    #         # 13,
    #         34,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "Q2_K",
    #         # "Q3_K_L",
    #         "Q3_K_M",
    #         # "Q3_K_S",
    #         # "Q4_0",
    #         "Q4_K_M",
    #         # "Q4_K_S",
    #         # "Q5_0",
    #         "Q5_K_M",
    #         # "Q5_K_S",
    #         "Q6_K",
    #         "Q8_0",
    #     ],
    # },
    # "c4ai-command-r-v01": {
    #     "model_size_in_billions": [
    #         35,
    #         # 104,  # this model has no quantisations
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         # "Q2_K",  # Xinference reports out of memory (makes no sense); apparently the current implementation requires too much memory somehow (the model is only 20GB)
    #         # "Q3_K_L",
    #         # "Q3_K_M",
    #         # "Q3_K_S",
    #         # "Q4_0",
    #         "Q4_K_M",
    #         # "Q4_K_S",
    #         # "Q5_0",
    #         # "Q5_K_M",
    #         # "Q5_K_S",
    #         # "Q6_K",
    #         # "Q8_0",
    #     ],
    # },
    # "custom-llama-3-instruct": {
    #     "model_size_in_billions": [
    #         70,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "IQ1_M",
    #     ],
    # },
    # "gemma-it": {
    #     "model_size_in_billions": [
    #         2,
    #         7,
    #     ],
    #     "model_format": "pytorch",
    #     "quantization": [
    #         "none",
    #         "4-bit",
    #         "8-bit",
    #     ],
    # },
    # "glm4-chat": {
    #     "model_size_in_billions": [
    #         9,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         # "Q2_K",
    #         # "IQ3_XS",
    #         # "IQ3_S",
    #         # "IQ3_M",
    #         # "Q3_K_S",
    #         # "Q3_K_L",
    #         # "Q3_K",
    #         "IQ4_XS",
    #         # "IQ4_NL",
    #         # "Q4_K_S",
    #         # "Q4_K",
    #         # "Q5_K_S",
    #         # "Q5_K",
    #         # "Q6_K",
    #         # "Q8_0",
    #         # "BF16",
    #         # "FP16",
    #     ],
    # },
    # "llama-2-chat": {
    #     "model_size_in_billions": [
    #         7,
    #         # 13,
    #         # 70,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "Q2_K",
    #         # "Q3_K_S",
    #         "Q3_K_M",
    #         # "Q3_K_L",
    #         # "Q4_0",
    #         # "Q4_K_S",
    #         "Q4_K_M",
    #         # "Q5_0",
    #         # "Q5_K_S",
    #         "Q5_K_M",
    #         "Q6_K",
    #         "Q8_0",
    #     ],
    # },
    # "llama-3-instruct": {
    #     "model_size_in_billions": [
    #         8,
    #         # 70,  # currently buggy
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         # 8B model quantisations
    #         # "IQ3_M",  # currently buggy
    #         "Q4_K_M",
    #         "Q5_K_M",
    #         "Q6_K",
    #         "Q8_0",
    #         # 70B model quantisations
    #         # "IQ1_M",
    #         # "IQ2_XS",
    #         # "Q4_K_M",
    #     ],
    # },
    # "llama-3.1-instruct": {
    #     "model_size_in_billions": [
    #         8,
    #         # 70,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         # 8B model quantisations
    #         "Q3_K_L",
    #         "IQ4_XS",
    #         "Q4_K_M",
    #         # "Q5_K_M",
    #         # "Q6_K",
    #         "Q8_0",
    #         # 70B model quantisations
    #         # "IQ2_M",
    #         # "Q2_K",
    #         # "Q3_K_S",
    #         # "IQ4_XS",
    #         # "Q4_K_M",  # crazy slow on mbp m3 max
    #         # "Q5_K_M",
    #         # "Q6_K",
    #         # "Q8_0",
    #     ],
    # },
    # "mistral-instruct-v0.2": {
    #     "model_size_in_billions": [
    #         7,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "Q2_K",
    #         # "Q3_K_S",
    #         "Q3_K_M",
    #         # "Q3_K_L",
    #         # "Q4_0",
    #         # "Q4_K_S",
    #         "Q4_K_M",
    #         # "Q5_0",
    #         # "Q5_K_S",
    #         "Q5_K_M",
    #         "Q6_K",
    #         "Q8_0",
    #     ],
    # },
    # "mixtral-instruct-v0.1": {
    #     "model_size_in_billions": [
    #         "46_7",
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "Q2_K",
    #         # "Q3_K_M",
    #         # "Q4_0",
    #         "Q4_K_M",
    #         # "Q5_0",
    #         "Q5_K_M",
    #         "Q6_K",
    #         "Q8_0",
    #     ],
    # },
    # "openbiollm-llama3-8b": {
    #     "model_size_in_billions": [
    #         8,
    #     ],
    #     "model_format": "pytorch",
    #     "quantization": [
    #         "none",
    #     ],
    # },
    # "openhermes-2.5": {
    #     "model_size_in_billions": [
    #         7,
    #     ],
    #     "model_format": "ggufv2",
    #     "quantization": [
    #         "Q2_K",
    #         # "Q3_K_S",
    #         "Q3_K_M",
    #         # "Q3_K_L",
    #         # "Q4_0",
    #         # "Q4_K_S",
    #         "Q4_K_M",
    #         # "Q5_0",
    #         # "Q5_K_S",
    #         "Q5_K_M",
    #         "Q6_K",
    #         "Q8_0",
    #     ],
    # },
}

# create concrete benchmark list by concatenating all combinations of model
# names, model sizes and quantizations
XINFERENCE_MODEL_NAMES = [
    f"{model_name}:{model_size}:{model_format}:{quantization}"
    for model_name in XINFERENCE_MODELS
    for model_size in XINFERENCE_MODELS[model_name]["model_size_in_billions"]
    for model_format in [XINFERENCE_MODELS[model_name]["model_format"]]
    for quantization in XINFERENCE_MODELS[model_name]["quantization"]
]

BENCHMARKED_MODELS = OPENAI_MODEL_NAMES + ANTHROPIC_MODEL_NAMES + XINFERENCE_MODEL_NAMES
BENCHMARKED_MODELS.sort()

# Xinference IP and port
BENCHMARK_URL = "http://localhost:9997"


@pytest.fixture(scope="session")
def client():
    try:
        return Client(base_url=BENCHMARK_URL)
    except requests.exceptions.ConnectionError:
        return None  # ignore if server is not running


@pytest.fixture(scope="session", autouse=True)
def register_model(client):
    """Register custom (non-builtin) models with the Xinference server. Should only
    happen once per session.
    """
    if client is None:
        return  # ignore if server is not running

    registrations = client.list_model_registrations(model_type="LLM")
    registered_models = [registration["model_name"] for registration in registrations]

    if "openbiollm-llama3-8b" not in registered_models:
        with open("benchmark/models/openbiollm-llama3-8b.json") as fd:
            model = fd.read()
        client.register_model(model_type="LLM", model=model, persist=False)

    # if "custom-llama-3-instruct-70b" not in registered_models:
    #     with open("benchmark/models/custom-llama-3-instruct-70b.json") as fd:
    #         model = fd.read()
    #     client.register_model(model_type="LLM", model=model, persist=False)


def pytest_collection_modifyitems(items):
    """Pytest hook function to modify the collected test items.
    Called once after collection has been performed.

    Used here to order items by their `callspec.id` (which starts with the
    model name and configuration) to ensure running all tests for one model
    before moving to the next model.
    """
    items.sort(
        key=lambda item: (item.callspec.id if hasattr(item, "callspec") else ""),
    )

    # can we skip here the tests (model x hash) that have already been executed?


# parameterise tests to run for each model
@pytest.fixture(params=BENCHMARKED_MODELS)
def model_name(request):
    return request.param


@pytest.fixture()
def multiple_testing(request):
    def run_multiple_times(test_func, *args, **kwargs):
        scores = []
        for _ in range(N_ITERATIONS):
            score, max = test_func(*args, **kwargs)
            scores.append(score)
        score_string = ";".join([str(score) for score in scores])
        return (score_string, max, N_ITERATIONS)

    return run_multiple_times


def calculate_bool_vector_score(vector: list[bool]) -> tuple[int, int]:
    score = sum(vector)
    max = len(vector)
    return (score, max)


@pytest.fixture()
def conversation(request, model_name, client):
    """Decides whether to run the test or skip due to the test having been run
    before. If not skipped, will create a conversation object for interfacing
    with the model.
    """
    test_name = request.node.originalname.replace("test_", "")
    subtask = "?"  # TODO can we get the subtask here?
    if benchmark_already_executed(model_name, test_name, subtask):
        pass
        # pytest.skip(
        #     f"benchmark {test_name}: {subtask} with {model_name} already executed"
        # )

    if model_name in OPENAI_MODEL_NAMES:
        conversation = GptConversation(
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            os.getenv("OPENAI_API_KEY"),
            user="benchmark_user",
        )

    elif model_name in ANTHROPIC_MODEL_NAMES:
        conversation = AnthropicConversation(
            model_name=model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            os.getenv("ANTHROPIC_API_KEY"),
            user="benchmark_user",
        )

    elif model_name in XINFERENCE_MODEL_NAMES:
        (
            _model_name,
            _model_size,
            _model_format,
            _model_quantization,
        ) = model_name.split(":")
        if "_" not in _model_size:
            _model_size = int(_model_size)

        # if exact model already running, return conversation
        running_models = client.list_models()
        if running_models:
            for running_model in running_models:
                if (
                    running_models[running_model]["model_name"] == _model_name
                    and running_models[running_model]["model_size_in_billions"] == _model_size
                    and running_models[running_model]["quantization"] == _model_quantization
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
        if _model_format == "pytorch":
            _model_engine = "transformers"
        elif _model_format in ["ggufv2", "ggmlv3"]:
            _model_engine = "llama.cpp"
        client.launch_model(
            model_engine=_model_engine,
            model_name=_model_name,
            model_size_in_billions=_model_size,
            model_format=_model_format,
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


@pytest.fixture()
def prompt_engine(request, model_name, conversation):
    """Generates a constructor for the prompt engine for the current model name."""

    def conversation_factory():
        return conversation

    def setup_prompt_engine(kg_schema_dict):
        return BioCypherPromptEngine(
            schema_config_or_info_dict=kg_schema_dict,
            model_name=model_name,
            conversation_factory=conversation_factory,
        )

    return setup_prompt_engine


@pytest.fixture()
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
    """If --run-all is set, the former benchmark data are deleted and all
    benchmarks are executed again.
    """
    if request.config.getoption("--run-all"):
        RESULT_FILES = [
            f"benchmark/results/{file}" for file in os.listdir("benchmark/results") if file.endswith(".csv")
        ]
        for f in RESULT_FILES:
            if os.path.exists(f):
                old_df = pd.read_csv(f, header=0)
                empty_df = pd.DataFrame(columns=old_df.columns)
                empty_df.to_csv(f, index=False)


@pytest.fixture(scope="session")
def result_files():
    RESULT_FILES = [f"benchmark/results/{file}" for file in os.listdir("benchmark/results") if file.endswith(".csv")]
    result_files = {}
    result_columns = [
        "model_name",
        "subtask",
        "score",
        "iterations",
        "md5_hash",
    ]
    for file in RESULT_FILES:
        try:
            result_file = pd.read_csv(file, header=0)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            result_file = pd.DataFrame(
                columns=result_columns,
            )
            result_file.to_csv(file, index=False)

        if not np.array_equal(
            result_file.columns,
            result_columns,
        ):
            result_file.columns = result_columns

        result_files[file] = result_file

    return result_files


def pytest_generate_tests(metafunc):
    """Pytest hook function to generate test cases.
    Called once for each test case in the benchmark test collection.
    If fixture is part of test declaration, the test is parametrized.
    """
    # Load the data
    data = BENCHMARK_DATASET

    # Parametrize the fixtures with the collected rows
    if "test_data_biocypher_query_generation" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_biocypher_query_generation",
            data["biocypher_query_generation"],
        )
    if "test_data_rag_interpretation" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_rag_interpretation",
            data["rag_interpretation"],
        )
    if "test_data_text_extraction" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_text_extraction",
            data["text_extraction"],
        )
    if "test_data_api_calling" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_api_calling",
            data["api_calling"],
        )
    if "test_data_medical_exam" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_medical_exam",
            data["medical_exam"],
        )


@pytest.fixture()
def kg_schemas():
    data = BENCHMARK_DATASET
    return data["kg_schemas"]
