import os
import pytest

import openai
from judge_connect import GPTJudgeConnection
from judgement_utils import get_judgement_dataset

N_ITERATIONS = 2

OPENAI_JUDGE = [
    "gpt-4o-mini-2024-07-18",
    # "gpt-4o-2024-08-06",
]

JUDGEMENT_DATA = get_judgement_dataset(path = "./judgement/responses/")

JUDGES = OPENAI_JUDGE

@pytest.fixture()
def create_conversation():
    """
    A Pytest fixture to create a `conversation` for interacting with a model.

    This fixture defines a `conversation` function that initializes a connection 
    to a specific model and generates a message based on the provided system 
    and user prompts.

    Args:
        - model (str): The name of the model to interact with.
        - base_url (str): The base URL for the model's API.
        - system_prompt (str): The system prompt to initialize the conversation.
        - user_prompt (str): The user prompt to query the model.

    Returns:
        callable: A function (`conversation`) that can be used to create a 
        conversation with the specified parameters.
    """

    def conversation(
        model: str, 
        base_url: str,
        system_prompt: str,
        user_prompt: str,
    ):
        if model in OPENAI_JUDGE:
            conversation = GPTJudgeConnection(
                model_name = model,
                base_url = base_url,
            )
            conversation.initialize_client(
                api_key = os.getenv("OPENAI_API_KEY")
            )
            message = conversation.create_message(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
            )
        return message
    return conversation

@pytest.fixture(params=JUDGES)
def model_name(request):
    """
    A Pytest fixture that provides parameterized model names for testing.

    This fixture iterates over the list of models defined in the global `JUDGES` 
    variable, allowing each test that uses this fixture to be executed with a 
    different model name.

    Args:
        request (FixtureRequest): A built-in Pytest object that provides access 
        to the current parameter (`request.param`).

    Returns:
        str: A model name from the `JUDGES` list.
    """

    return request.param

@pytest.fixture()
def multiple_testing(request):
    """
    A Pytest fixture to repeatedly execute a test function multiple times and collect results.

    This fixture provides a `run_multiple_times` function that can be used to execute 
    a given test function (`test_func`) a predefined number of times (`N_ITERATIONS`), 
    collect scores from each iteration, and return aggregated results.

    Args `run_multiple_times`:
        - test_func: The test function to be executed.
        - *args: Positional arguments to be passed to `test_func`.
        - **kwargs: Keyword arguments to be passed to `test_func`.

    `run_multiple_times` returns:
        - score_string (str): A semicolon-separated string of scores from each iteration.
        - max: The `max` value returned by the test function.
        - N_ITERATIONS (int): The total number of iterations performed.

    The function/fixture returns:
        callable: A function (`run_multiple_times`) that takes a test function as input 
        and runs it multiple times, collecting and aggregating results.
    """

    def run_multiple_times(test_func, *args, **kwargs):
        scores = []
        for _ in range(N_ITERATIONS):
            score, max = test_func(*args, **kwargs)
            scores.append(score)
        score_string = ";".join([str(score) for score in scores])
        return (score_string, max, N_ITERATIONS)

    return run_multiple_times

def calculate_bool_vector_score(vector: list[bool]) -> tuple[int, int, list]:
    """
    Calculates a score and maximum value from a boolean vector.

    This function computes the score of a boolean vector by dividing the sum 
    of `True` values by the length of the vector. It also calculates the maximum 
    possible score (i.e., the length of the vector).

    Args:
        vector (list[bool]): A list of boolean values (`True` or `False`).

    Returns:
        tuple:
            - score (int): The proportion of `True` values in the vector as an integer percentage.
            - max (int): The length of the vector, representing the maximum possible score.
            - vector (list): The original boolean vector.
    """

    score = sum(vector)/len(vector)
    max = len(vector)
    return (score, max)

def pytest_generate_tests(metafunc):
    """
    Dynamically generates parameterized tests for judgement data.

    This function is a Pytest hook that automatically generates test cases based 
    on the `JUDGEMENT_DATA` variable. If the `test_judgement` fixture is used in 
    a test, the function will parameterize it with data from the `"judgement"` key 
    in `JUDGEMENT_DATA`.

    Args:
        metafunc (Metafunc): A Pytest object representing the test function and 
        its fixtures. Used to dynamically parameterize tests.
    """

    data = JUDGEMENT_DATA

    if "test_judgement" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_judgement",
            data["judgement"],
        )