import re
import inspect
import ast
import os

import nltk
import pytest

from biochatter._misc import ensure_iterable
from .conftest import return_response
from .benchmark_utils import (
    skip_if_already_run,
    get_response_mode_file_path,
    write_responses_to_file,
    write_judgement_to_file,
    get_prompt_binary,
)

@pytest.mark.dependency(name="generate_responses")
@pytest.mark.order(1)
def test_longevity_response_generation_plain(
    model_name,
    test_create_longevity_responses,
    conversation,
    multiple_responses,
):
    """
    Response generation of a model for the longevity and geriatrics 
    benchmark (or any custom benchmark).

    The responses are then logged to a response file for further downstream 
    evaluation (e.g. LLM-as-a-Judge).

    Args:
        - model_name (str): name of the model being tested
        - test_data_responses: contains test-specific data, including system messages, 
          user prompts, expected outputs, and a unique hash etc. for the test data
        - conversation: used for managing system and user interactions during the test
        - multiple_testing (callable): used to execute the test multiple 
          times, ensuring reliable response evaluation
    """

    yaml_data = test_create_longevity_responses
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    skip_if_already_run(
        model_name = model_name, 
        task = task, 
        md5_hash = yaml_data["hash"],
    )

    def run_test():
        conversation.reset()
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        resp_ = []

        resp_.append(response)

        return return_response(resp_)

    n_iterations, responses = multiple_responses(run_test)

    write_responses_to_file(
        model_name,
        yaml_data["case_id"],
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

@pytest.mark.dependency(depends=["generate_responses"])
@pytest.mark.order(2)
def test_longevity_judgement(
    judge_name,
    judge_metric,
    test_judge_longevity_responses,
    judge_conversation,
):
    for yaml_data in test_judge_longevity_responses:
        print(f"{judge_name} judges {yaml_data['model_name']} for {judge_metric} in {yaml_data['md5_hash']}")

        task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
        skip_if_already_run(
            model_name = judge_name, 
            task = task, 
            md5_hash = yaml_data["md5_hash"],
        )

        means = []
        for response in ast.literal_eval(yaml_data["response"]):
            ITERATIONS = 3
            prompts = get_prompt_binary(path = "./benchmark/LLM_as_a_Judge/prompts/prompts.yaml")

            success = prompts[judge_metric]["success"]
            failure = prompts[judge_metric]["failure"]

            system_message = "As a research assistant, your task is to assess the processing of a scientific question and the transmitted answer as carried out by another LLM."
            judge_conversation.append_system_message(system_message)

            prompt = prompts[judge_metric]["prompt"].format(
                success = success,
                failure = failure,
                prompt = yaml_data["prompt"],
                summary = yaml_data["summary"],
                keywords = yaml_data["key_words"],
                response = response,
                expected_answer = yaml_data["expected_answer"]
            )

            scores = []
            for iter in range(ITERATIONS):
                judgement, _, _ = judge_conversation.query(prompt)
                
                if judgement.lower().replace(".", "") == success:
                    scores.append(True)
                elif judgement.lower().replace(".", "") == failure:
                    scores.append(False)
                else:
                    scores.append(False)
            score = sum(scores)
            mean = score / len(scores)
            means.append(mean)
        
        score_string = ";".join([str(mean) for mean in means])
        
        if "list" in yaml_data["subtask"]:
            prompt_type = "list"
        elif "freetext" in yaml_data["subtask"]:
            prompt_type = "freetext"

        if ":appendix:" in yaml_data["subtask"]:
            clause = "appendix"
        elif "no_appendix" in yaml_data["subtask"]:
            clause = "no_appendix"

        if "simple" in yaml_data["subtask"]:
            system_prompt = "simple"
        elif "detailed" in yaml_data["subtask"]:
            system_prompt = "detailed"
        elif "explicit" in yaml_data["subtask"]:
            system_prompt = "explicit"

        write_judgement_to_file(
            judge_model = judge_name,
            evaluated_model = yaml_data["model_name"],
            iterations = f"{ITERATIONS}",
            metric = judge_metric,
            case_id = yaml_data["case_id"],
            subtask = yaml_data["subtask"],
            individual = yaml_data["individual"],
            md5_hash = yaml_data["md5_hash"],
            prompt = yaml_data["prompt"],
            system_prompt = system_prompt,
            prompt_type = prompt_type,
            is_appendix = clause,
            responses = yaml_data["response"],
            expected_answer = yaml_data["expected_answer"],
            rating = f"{score_string}/{1}",
            path = f"./benchmark/LLM_as_a_Judge/judgement/{task}_{yaml_data['model_name']}.csv",
        )