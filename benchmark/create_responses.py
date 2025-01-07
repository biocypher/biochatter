import re
import inspect

import nltk
import pytest

from biochatter._misc import ensure_iterable
from .conftest import return_response
from .benchmark_utils import (
    skip_if_already_run,
    get_response_mode_file_path,
    write_responses_to_file,
)

def test_longevity_geriatrics_responses(
    model_name,
    test_data_responses,
    conversation,
    multiple_responses,
):
    """
    Tests the response generation capabilities of a model for the longevity and geriatrics 
    benchmark (or any custom benchmark).

    This function benchmarks the model by running multiple tests using predefined 
    prompts, system messages, and expected outputs. The results are then logged to result file
    for further downstream evaluation (e.g. LLM-as-a-Judge).

    Args:
        - model_name (str): name of the model being tested
        - test_data_responses: contains test-specific data, including system messages, 
          user prompts, expected outputs, and a unique hash etc. for the test data
        - conversation: used for managing system and user interactions during the test
        - multiple_testing (callable): used to execute the test multiple 
          times, ensuring reliable response evaluation
    """
    # Downloads the naturale language synonym toolkit, just need to be done once per device
    # nltk.download()

    yaml_data = test_data_responses
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    skip_if_already_run(
        model_name = model_name, 
        task = task, 
        md5_hash = yaml_data["hash"],
    )

    def run_test():
        conversation.reset()
        # Define the system prompt
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        # Define the user prompt
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        resp_ = []

        resp_.append(response)

        return return_response(resp_)

    n_iterations, responses = multiple_responses(run_test)

    write_responses_to_file(
        model_name,
        yaml_data["case"],
        yaml_data["expected"]["individual"],
        yaml_data["input"]["prompt"],
        responses,
        yaml_data["expected"]["answer"][0],
        yaml_data["expected"]["summary"],
        yaml_data["expected"]["key_words"],
        f"{n_iterations}",
        yaml_data["hash"],
        get_response_mode_file_path(task, model_name),
    )