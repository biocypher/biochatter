import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
import requests
from xinference.client import Client

from biochatter.llm_connect import (
    AnthropicConversation,
    Conversation,
    GptConversation,
    LangChainConversation,
    XinferenceConversation,
)
from biochatter.prompts import BioCypherPromptEngine

from .benchmark_utils import benchmark_already_executed, get_judgement_dataset
from .load_dataset import get_benchmark_dataset

# Patch event loop if running in pytest
if "ipykernel" not in sys.modules:
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

# how often should each benchmark be run?
N_ITERATIONS = 1

# which dataset should be used for benchmarking?
BENCHMARK_DATASET = get_benchmark_dataset()

# which models should be benchmarked?
# List generated from https://api.openai.com/v1/models
OPENAI_MODEL_NAMES = [
    # GPT-3.5 models
    # "gpt-3.5-turbo-0125",
    # GPT-4 models
    # "gpt-4-0613",
    # "gpt-4-0125-preview",
    # "gpt-4-1106-preview",
    # "gpt-4-turbo",
    # "gpt-4-turbo-2024-04-09",
    # "gpt-4-turbo-preview",
    # GPT-4o models
    # "gpt-4o-2024-05-13",
    # "gpt-4o-2024-08-06",
    # "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    # GPT-4.1 models
    # "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    # "gpt-4.1-nano-2025-04-14",
    # GPT-5 models
    "gpt-5-2025-08-07",
    # "gpt-5-chat-latest",
    # "gpt-5-codex",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    # "gpt-5-pro-2025-10-06",
    # O1 models (reasoning models)
    # "o1",
    # "o1-2024-12-17",
    # "o1-mini",
    # "o1-mini-2024-09-12",
    # "o1-pro",
    # "o1-pro-2025-03-19",
    # O3 models (reasoning models)
    # "o3",
    # "o3-2025-04-16",
    # "o3-mini",
    # "o3-mini-2025-01-31",
]

GROQ_MODEL_NAMES = [
    # "llama-3.3-70b-versatile",
]

LM_STUDIO_MODEL_NAMES = [
    # "llama-3.2-1b-instruct",
    # "llama-3.2-3b-instruct",
    # "qwen2.5-14b-instruct",
]

ANTHROPIC_MODEL_NAMES = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
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

BENCHMARKED_MODELS = (
    OPENAI_MODEL_NAMES + ANTHROPIC_MODEL_NAMES + XINFERENCE_MODEL_NAMES + GROQ_MODEL_NAMES + LM_STUDIO_MODEL_NAMES
)
BENCHMARKED_MODELS.sort()

# Xinference IP and port
BENCHMARK_URL = "http://localhost:9997"

OPENAI_JUDGE = [
    "gpt-4o-2024-11-20",
    # "gpt-4o-mini-2024-07-18",
    # "o3-mini",
]

JUDGES = OPENAI_JUDGE

METRICS = [
    "correctness",
    # "comprehensiveness",
    # "usefulness",
    # "interpretability_explainability",
    # "toxicity",
]


@pytest.fixture(scope="session")
def client():
    try:
        return Client(base_url=BENCHMARK_URL)
    except requests.exceptions.ConnectionError:
        return None  # ignore if server is not running


@pytest.fixture(scope="session", autouse=True)
def register_model(client):
    """Register custom (non-builtin) models with the Xinference server.

    Should only happen once per session.
    """
    if client is None:
        return  # ignore if server is not running

    registrations = client.list_model_registrations(model_type="LLM")
    registered_models = [registration["model_name"] for registration in registrations]

    if "openbiollm-llama3-8b" not in registered_models:
        with open("benchmark/models/openbiollm-llama3-8b.json") as fd:
            model = fd.read()
        client.register_model(model_type="LLM", model=model, persist=False)


def pytest_collection_modifyitems(items):
    """Modify the collected test items (Pytest hook).

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


@pytest.fixture(params=JUDGES)
def judge_name(request):
    """Provide parameterized model names for testing.

    This fixture iterates over the list of models defined in the global `JUDGES`
    variable, allowing each test that uses this fixture to be executed with a
    different model name.

    Args:
    ----
        request (FixtureRequest): A built-in Pytest object that provides access
        to the current parameter (`request.param`).

    Returns:
    -------
        str: A model name from the `JUDGES` list.

    """
    return request.param


@pytest.fixture(params=METRICS)
def judge_metric(request):
    return request.param


@pytest.fixture
def multiple_testing(request):
    def run_multiple_times(test_func, *args, **kwargs) -> tuple[str, int, int]:
        scores = []
        for _ in range(N_ITERATIONS):
            score, max = test_func(*args, **kwargs)
            scores.append(score)
        score_string = ";".join([str(score) for score in scores])
        return (score_string, max, N_ITERATIONS)

    return run_multiple_times


def calculate_bool_vector_score(vector: list[bool]) -> tuple[int, int]:
    score = sum(vector)
    max_score = len(vector)
    return (score, max_score)


@pytest.fixture
def multiple_responses(request):
    def run_multiple_times(test_func, *args, **kwargs):
        responses = []
        for _ in range(N_ITERATIONS):
            response = test_func(*args, **kwargs)
            cleaned_response = re.sub(r"\n\d*", "", response[0])
            responses.append(
                cleaned_response.replace(".. ", ".").replace(":. ", ": ").replace(".", ". ").replace(".  ", ". ")
            )
        resps = [resp for resp in responses]
        return (N_ITERATIONS, resps)

    return run_multiple_times


def return_response(response: list):
    return response


@pytest.fixture
def conversation(request, model_name, client) -> Conversation:
    """Return a conversation object.

    Could skip due to the test having been run before (but not sure how to
    implement yet). If not skipped, will create a conversation object for
    interfacing with the model.
    """
    test_name = request.node.originalname.replace("test_", "")
    subtask = "?"  # TODO: can we get the subtask here?
    if benchmark_already_executed(model_name, test_name, subtask):
        pass

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

    elif model_name in GROQ_MODEL_NAMES:
        print(model_name)
        conversation = GptConversation(
            model_name=model_name,
            prompts={},
            correct=False,
            base_url="https://api.groq.com/openai/v1",
        )
        conversation.set_api_key(
            os.getenv("GROQ_API_KEY"),
            user="benchmark_user",
        )

    elif model_name in LM_STUDIO_MODEL_NAMES:
        print(model_name)
        conversation = GptConversation(
            model_name=model_name,
            prompts={},
            correct=False,
            base_url="http://localhost:1234/v1",
        )
        conversation.set_api_key(
            os.getenv("LM_STUDIO_API_KEY"),
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

        conversation = XinferenceConversation(
            base_url=BENCHMARK_URL,
            model_name=_model_name,
            prompts={},
            correct=False,
        )

    return conversation


@pytest.fixture
def prompt_engine(request, model_name, conversation):
    """Generate a constructor for the prompt engine for current model name."""

    def conversation_factory() -> Conversation:
        return conversation

    def setup_prompt_engine(kg_schema_dict) -> BioCypherPromptEngine:
        return BioCypherPromptEngine(
            schema_config_or_info_dict=kg_schema_dict,
            model_name=model_name,
            conversation_factory=conversation_factory,
        )

    return setup_prompt_engine


@pytest.fixture
def judge_conversation(judge_name):
    if judge_name in OPENAI_JUDGE:
        conversation = GptConversation(
            model_name=judge_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="benchmark_user")
    return conversation


@pytest.fixture
def evaluation_conversation() -> Conversation:
    conversation = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        correct=False,
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="benchmark_user")
    return conversation


@pytest.fixture(scope="session")
def mcp_server_config():
    """Get MCP server configuration.

    Returns the directory and module path for running the MCP server.
    The server must be run as a module from its parent directory to handle
    relative imports correctly.
    """
    from pathlib import Path

    # Path to the edammcp repository root
    # You can override this with MCP_SERVER_DIR environment variable
    server_dir = os.getenv("MCP_SERVER_DIR", None)
    if server_dir is None:
        # Default: try to find it in common locations
        server_dir = Path.home() / "GitHub" / "edammcp"
        if not server_dir.exists():
            pytest.skip("MCP server directory not found. Set MCP_SERVER_DIR environment variable.")
    else:
        server_dir = Path(server_dir)

    if not server_dir.exists():
        pytest.skip(f"MCP server directory not found at {server_dir}")

    # Verify the module exists
    main_module = server_dir / "edam_mcp" / "main.py"
    if not main_module.exists():
        pytest.skip(f"MCP server main.py not found at {main_module}")

    # Try to find virtual environment Python, fallback to system python
    venv_python = server_dir / ".venv" / "bin" / "python"
    python_command = str(venv_python) if venv_python.exists() else "python"

    return {
        "cwd": server_dir,
        "module": "edam_mcp.main",
        "python": python_command,
    }


@pytest.fixture
def mcp_server(mcp_server_config):
    """Set up and tear down the MCP server for each test.

    This fixture provides an MCP server session. The server is
    started fresh for each test to ensure clean state.

    Note: You need to set the MCP_SERVER_PATH environment variable or
    modify the server_path in mcp_server_config fixture to point to your EDAM MCP server.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import asyncio

        server_config = mcp_server_config
        server_dir = server_config["cwd"]
        server_module = server_config["module"]
        python_command = server_config["python"]

        # Run as a module from the parent directory to handle relative imports
        # The server needs to run from its parent directory to resolve relative imports
        # and access any data files it might need
        # Use -u flag for unbuffered output (important for stdio communication)
        # Match the Cursor MCP configuration approach
        server_params = StdioServerParameters(
            command=python_command,
            args=["-u", "-m", server_module],
            env=dict(
                os.environ,
                PYTHONPATH=str(server_dir),
                PYTHONUNBUFFERED="1",
            ),
        )

        # Note: StdioServerParameters doesn't directly support cwd parameter.
        # However, by setting PYTHONPATH to the server directory, the module should
        # be importable. The stdio_client will spawn the process, and we rely on
        # the server's ability to handle relative paths from its package location.
        # If the server needs to access files relative to its directory, those should
        # be resolved using __file__ or similar within the server code itself.

        # Create a new event loop for this fixture
        # Using a new loop avoids conflicts with pytest's event loop handling
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Store context managers to keep them alive
        context_managers = []
        session = None

        try:
            # Run async context manager with timeout
            async def _get_session():
                stdio_ctx = stdio_client(server_params)
                read, write = await stdio_ctx.__aenter__()
                context_managers.append(stdio_ctx)

                session_ctx = ClientSession(read, write)
                sess = await session_ctx.__aenter__()
                context_managers.append(session_ctx)

                await sess.initialize()
                return sess

            # Start the server and get session with timeout
            try:
                session = loop.run_until_complete(asyncio.wait_for(_get_session(), timeout=30.0))
            except asyncio.TimeoutError:
                pytest.skip("MCP server connection timed out after 30 seconds")

            yield session

        finally:
            # Cleanup: exit context managers in reverse order
            for ctx in reversed(context_managers):
                try:
                    loop.run_until_complete(asyncio.wait_for(ctx.__aexit__(None, None, None), timeout=5.0))
                except Exception:
                    pass
            loop.close()

    except ImportError:
        pytest.skip("MCP dependencies not installed. Install with: poetry install --with mcp")
    except Exception as e:
        pytest.skip(f"Could not start MCP server: {e}")


@pytest.fixture
def mcp_conversation(request, model_name, mcp_server):
    """Create a conversation with MCP tools loaded.

    This fixture creates a LangChainConversation with MCP tools
    loaded from the MCP server session.
    """
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
        import asyncio

        # Get event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Load MCP tools from the session
        tools = loop.run_until_complete(load_mcp_tools(mcp_server))

        # Determine model provider and API key
        if model_name in OPENAI_MODEL_NAMES:
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
        elif model_name in ANTHROPIC_MODEL_NAMES:
            provider = "anthropic"
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            pytest.skip(f"Model {model_name} not configured for MCP benchmark")

        if not api_key:
            pytest.skip(f"API key not set for {provider}")

        # Create conversation with MCP support
        conversation = LangChainConversation(
            model_name=model_name,
            model_provider=provider,
            prompts={},
            mcp=True,
            tools=tools,
        )

        conversation.set_api_key(api_key, user="benchmark_user")

        return conversation
    except ImportError:
        pytest.skip("MCP dependencies not installed. Install with: poetry install --with mcp")
    except Exception as e:
        pytest.skip(f"Could not create MCP conversation: {e}")


@pytest.fixture
def baseline_conversation(request, model_name):
    """Create a conversation without MCP tools for baseline comparison.

    This fixture creates a LangChainConversation with the same model
    but without any tools, to serve as a baseline for comparison.
    """
    # Determine model provider and API key
    if model_name in OPENAI_MODEL_NAMES:
        provider = "openai"
        api_key = os.getenv("OPENAI_API_KEY")
    elif model_name in ANTHROPIC_MODEL_NAMES:
        provider = "anthropic"
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:
        pytest.skip(f"Model {model_name} not configured for baseline benchmark")

    if not api_key:
        pytest.skip(f"API key not set for {provider}")

    # Create conversation without tools (baseline)
    conversation = LangChainConversation(
        model_name=model_name,
        model_provider=provider,
        prompts={},
        mcp=False,
        tools=None,
    )

    conversation.set_api_key(api_key, user="benchmark_user")

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
    """Delete the content of the result files.

    If --run-all is set, the former benchmark data are deleted and all
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


@pytest.fixture
def test_judge_longevity_responses():
    """Dynamically load judgment data."""
    path = "./benchmark/results/"
    if not os.path.exists(path) or not os.listdir(path):
        pytest.skip(f"No files found in directory: {path}")

    try:
        JUDGEMENT_DATA = get_judgement_dataset(path)
        return JUDGEMENT_DATA["judgement"]
    except ValueError as e:
        pytest.fail(f"Failed to load judgment data: {e}")


def pytest_generate_tests(metafunc):
    """Generate test cases.

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
    if "test_create_longevity_responses_simultaneously" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_create_longevity_responses_simultaneously",
            data["longevity_geriatric_case_assessment"],
        )
    if "test_data_mcp_edam_qa" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_data_mcp_edam_qa",
            data.get("mcp_edam_qa", []),
        )


@pytest.fixture
def kg_schemas():
    data = BENCHMARK_DATASET
    return data["kg_schemas"]
