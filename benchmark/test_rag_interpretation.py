import inspect

import pytest

from biochatter._misc import ensure_iterable
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def test_explicit_relevance_of_single_fragments(
    model_name,
    test_data_rag_interpretation,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_rag_interpretation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    if "explicit" not in yaml_data["case"]:
        pytest.skip(
            f"test case {yaml_data['case']} not supported for {task} benchmark"
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        # lower case, remove punctuation
        response = (
            response.lower().replace(".", "").replace("?", "").replace("!", "")
        ).strip()

        score = []

        score.append(response == yaml_data["expected"]["answer"])

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def test_implicit_relevance_of_multiple_fragments(
    model_name,
    test_data_rag_interpretation,
    conversation,
    evaluation_conversation,
    multiple_testing,
):
    yaml_data = test_data_rag_interpretation
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    if "implicit" not in yaml_data["case"]:
        pytest.skip(
            f"test case {yaml_data['case']} not supported for {task} benchmark"
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        msg = (
            "You will receive a statement as an answer to this question: "
            f"{yaml_data['input']['prompt']} "
            "If the statement is an answer to the question, please type 'answer'. "
            "If the statement declines to answer to the question or apologises, giving the reason of lack of relevance of the given text fragments, please type 'decline'. "
            "Do not type anything except these two options. Here is the statement: "
        )

        # evaluator LLM
        evaluation_conversation.append_system_message(msg)

        eval, _, _ = evaluation_conversation.query(response)

        # lower case, remove punctuation
        eval = (
            eval.lower().replace(".", "").replace("?", "").replace("!", "")
        ).strip()

        score = (
            [True] if eval == yaml_data["expected"]["behaviour"] else [False]
        )

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )
