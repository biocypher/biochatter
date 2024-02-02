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
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    subtask = f"{str(test_data_rag_interpretation['hash'])}_{test_data_rag_interpretation['test_case_purpose']}"
    if not test_data_rag_interpretation["test_case_purpose"] == "explicit":
        pytest.skip(
            f"test case {test_data_rag_interpretation['test_case_purpose']} not supported for {subtask} benchmark"
        )
    skip_if_already_run(model_name=model_name, task=task, subtask=subtask)

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in test_data_rag_interpretation["system_messages"]
        ]
        response, _, _ = conversation.query(
            test_data_rag_interpretation["prompt"]
        )

        # lower case, remove punctuation
        response = (
            response.lower().replace(".", "").replace("?", "").replace("!", "")
        ).strip()

        score = []

        score.append(response.lower() == test_data_rag_interpretation["answer"])

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        subtask,
        f"{mean_score}/{max}",
        f"{n_iterations}",
        get_result_file_path(task),
    )


def test_implicit_relevance_of_multiple_fragments(
    model_name,
    test_data_rag_interpretation,
    conversation,
    evaluation_conversation,
    multiple_testing,
):
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    subtask = f"{str(test_data_rag_interpretation['hash'])}_{test_data_rag_interpretation['test_case_purpose']}"
    if not test_data_rag_interpretation["test_case_purpose"] == "implicit":
        pytest.skip(
            f"test case {test_data_rag_interpretation['test_case_purpose']} not supported for {subtask} benchmark"
        )
    skip_if_already_run(model_name=model_name, task=task, subtask=subtask)

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in test_data_rag_interpretation["system_messages"]
        ]
        response, _, _ = conversation.query(
            test_data_rag_interpretation["prompt"]
        )

        # evaluator LLM
        evaluation_conversation.append_system_message(
            "Evaluate the following response regarding whether it acknowledges the irrelevance of provided information to the question. "
            "Answer 'yes' if the response acknowledges the irrelevance of provided information to the question, 'no' if the response attempts to answer the question. "
        )

        eval, _, _ = evaluation_conversation.query(response)

        score = [True] if eval.lower() == "yes" else [False]

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        subtask,
        f"{mean_score}/{max}",
        f"{n_iterations}",
        get_result_file_path(task),
    )
