import inspect

import pytest

import pandas as pd

from .conftest import calculate_test_score
from .benchmark_utils import (
    get_result_file_path,
    write_results_to_file,
    benchmark_already_executed,
)
from biochatter._misc import ensure_iterable

TASK = "rag_interpretation"
FILE_PATH = get_result_file_path(TASK)


def get_test_data(test_data_rag_interpretation: list) -> tuple:
    """Helper function to unpack the test data from the test_data_rag_interpretation fixture.

    Args:
        test_data_rag_interpretation (list): The test data from the test_data_rag_interpretation fixture

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


def skip_if_already_run(
    model_name: str,
    result_files: dict[str, pd.DataFrame],
    subtask: str,
) -> None:
    """Helper function to check if the test case is already executed.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        result_files (dict[str, pd.DataFrame]): The result files
        subtask (str): The benchmark subtask test case, e.g. "entities_0"
    """
    if benchmark_already_executed(TASK, subtask, model_name, result_files):
        pytest.skip(
            f"benchmark {TASK}: {subtask} with {model_name} already executed"
        )


def test_relevance_of_single_fragments(
    model_name,
    test_data_rag_interpretation,
    result_files,
    conversation,
):
    (
        system_messages,
        prompt,
        expected_answers,
        test_case_purpose,
        test_case_index,
    ) = get_test_data(test_data_rag_interpretation)
    subtask = f"{inspect.currentframe().f_code.co_name}_{str(test_case_index)}_{test_case_purpose}"
    skip_if_already_run(model_name, result_files, subtask)

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

    write_results_to_file(
        model_name,
        subtask,
        calculate_test_score(score),
        FILE_PATH,
    )
