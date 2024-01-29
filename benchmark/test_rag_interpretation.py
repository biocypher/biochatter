import inspect

import pytest

from biochatter._misc import ensure_iterable
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def get_test_data(test_data_rag_interpretation: list) -> tuple:
    """

    Helper function to unpack the test data from the
    test_data_rag_interpretation fixture.

    Args:
        test_data_rag_interpretation (list): The test data from the
        test_data_rag_interpretation fixture

    Returns:
        tuple: The unpacked test data
    """
    return (
        test_data_rag_interpretation["system_messages"],
        test_data_rag_interpretation["prompt"],
        test_data_rag_interpretation["entities"],
        test_data_rag_interpretation["test_case_purpose"],
        test_data_rag_interpretation["index"],
    )


def test_explicit_relevance_of_single_fragments(
    model_name,
    test_data_rag_interpretation,
    conversation,
    multiple_testing,
):
    (
        system_messages,
        prompt,
        expected_answers,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_rag_interpretation)
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    subtask = f"{str(test_case_index)}_{test_case_purpose}"
    if not test_case_purpose == "explicit":
        pytest.skip(
            f"test case {test_case_purpose} not supported for {subtask} benchmark"
        )
    skip_if_already_run(model_name=model_name, task=task, subtask=subtask)

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [conversation.append_system_message(m) for m in system_messages]
        response, _, _ = conversation.query(prompt)
        answers = ensure_iterable(response.split(","))

        score = []

        if len(answers) == len(expected_answers):
            for index, answer in enumerate(answers):
                if answer == expected_answers[index]:
                    score.append(True)
                else:
                    score.append(False)
        else:
            [score.append(False) for _ in expected_answers]

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
    (
        system_messages,
        prompt,
        expected_answers,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_rag_interpretation)
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    subtask = f"{str(test_case_index)}_{test_case_purpose}"
    if not test_case_purpose == "implicit":
        pytest.skip(
            f"test case {test_case_purpose} not supported for {subtask} benchmark"
        )
    skip_if_already_run(model_name=model_name, task=task, subtask=subtask)

    def run_test():
        conversation.reset()  # needs to be reset for each test
        [conversation.append_system_message(m) for m in system_messages]
        response, _, _ = conversation.query(prompt)

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
