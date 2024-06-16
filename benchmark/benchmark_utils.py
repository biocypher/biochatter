from datetime import datetime
import re
from nltk.corpus import wordnet
import pytest
import importlib_metadata

import pandas as pd


def benchmark_already_executed(
    model_name: str,
    task: str,
    md5_hash: str,
) -> bool:
    """

    Checks if the benchmark task and subtask test case for the model_name have
    already been executed.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        task (str): The benchmark task, e.g. "biocypher_query_generation"

        md5_hash (str): The md5 hash of the test case, e.g.,
            "72434e7a340a3f6dd047b944988491b7". It is created from the
            dictionary representation of the test case.

    Returns:

        bool: True if the benchmark case for the model_name has already been
            run, False otherwise
    """
    task_results = return_or_create_result_file(task)

    # check if failure group csv already exists
    return_or_create_failure_mode_file(task)

    if task_results.empty:
        return False

    run = (
        task_results[
            (task_results["model_name"] == model_name)
            & (task_results["md5_hash"] == md5_hash)
        ].shape[0]
        > 0
    )

    return run


def skip_if_already_run(
    model_name: str,
    task: str,
    md5_hash: str,
) -> None:
    """Helper function to check if the test case is already executed.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        task (str): The benchmark task, e.g. "biocypher_query_generation"

        md5_hash (str): The md5 hash of the test case, e.g.,
            "72434e7a340a3f6dd047b944988491b7". It is created from the
            dictionary representation of the test case.
    """
    if benchmark_already_executed(model_name, task, md5_hash):
        pytest.skip(
            f"Benchmark for {task} with hash {md5_hash} with {model_name} already executed"
        )


def return_or_create_result_file(
    task: str,
):
    """
    Returns the result file for the task or creates it if it does not exist.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
        pd.DataFrame: The result file for the task
    """
    file_path = get_result_file_path(task)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "subtask",
                "score",
                "iterations",
                "md5_hash",
                "datetime",
                "biochatter_version",
            ]
        )
        results.to_csv(file_path, index=False)
    return results


def return_or_create_failure_mode_file(task: str):
    """
    Returns the failure mode file for the task or creates it if it does not
    exist.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
        pd.DataFrame: The failure mode recording file for the task
    """
    file_path = get_failure_mode_file_path(task)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "subtask",
                "actual_answer",
                "expected_answer",
                "failure_modes",
                "md5_hash",
                "datetime",
            ]
        )
        results.to_csv(file_path, index=False)
    return results


def get_failure_mode_file_path(task: str) -> str:
    """

    Returns the path to the failure mode recording file.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
        str: The path to the failure mode file
    """
    return f"benchmark/results/{task}_failure_modes.csv"


def write_results_to_file(
    model_name: str,
    subtask: str,
    score: str,
    iterations: str,
    md5_hash: str,
    file_path: str,
):
    """Writes the benchmark results for the subtask to the result file.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        subtask (str): The benchmark subtask test case, e.g. "entities"
        score (str): The benchmark score, e.g. "5"
        iterations (str): The number of iterations, e.g. "7"
        md5_hash (str): The md5 hash of the test case
        file_path (str): The path to the result file
    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bc_version = importlib_metadata.version("biochatter")
    new_row = pd.DataFrame(
        [[model_name, subtask, score, iterations, md5_hash, now, bc_version]],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"]
    )
    results.to_csv(file_path, index=False)


def write_failure_modes_to_file(
    model_name: str,
    subtask: str,
    actual_answer: str,
    expected_answer: str,
    failure_modes: str,
    md5_hash: str,
    file_path: str,
):
    """

    Writes the failure modes identified for a given response to a subtask to
    the given file path.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        subtask (str): The benchmark subtask test case, e.g. "entities"

        actual_answer (str): The actual response given to the subtask question

        expected_answer (str): The expected response for the subtask

        failure_modes (str): The mode of failure, e.g. "Wrong word count",
        "Formatting", etc.

        md5_hash (str): The md5 hash of the test case

        file_path (str): The path to the result file
    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame(
        [
            [
                model_name,
                subtask,
                actual_answer,
                expected_answer,
                failure_modes,
                md5_hash,
                now,
            ]
        ],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"]
    )
    results.to_csv(file_path, index=False)


def categorize_failure_modes(
    actual_answer, expected_answer, regex=False
) -> str:
    """
    Categorises the mode of failure for a given response to a subtask.

    Args:
        actual_answer (str): The actual response given to the subtask question

        expected_answer (str): The expected response for the subtask

        regex (bool): Whether the expected answer is a regex expression

    Returns:
        str: The mode of failure, e.g. "Case Sensitivity", "Partial Match",
            "Format Error", "Synonym", "Words Missing", "Entire Answer Incorrect",
            "Other"
    """
    if not regex:
        # Check if the answer is right, but the case sensitivity was wrong (e.g. a / A)
        if actual_answer.lower() == expected_answer.lower():
            return "Case Sensitivity"

        # Check if the wrong answer contains the expected answer followed by ")"
        elif actual_answer.strip() == expected_answer + ")":
            return "Format Error"

        # Check if some of the answer is partially right, but only if it's more than one letter
        elif len(expected_answer) > 1 and (actual_answer in expected_answer or expected_answer in actual_answer):
            return "Partial Match"

        # Check if the format of the answer is wrong, but the answer otherwise is right (e.g. "a b" instead of "ab")
        elif re.sub(r"\s+", "", actual_answer.lower()) == re.sub(
            r"\s+", "", expected_answer.lower()
        ):
            return "Format Error"

        # Check if the answer is a synonym with nltk (e.g. Illness / Sickness)
        elif is_synonym(actual_answer, expected_answer):
            return "Synonym"

        # Check if the format of the answer is wrong due to numerical or alphabetic differences (e.g. "123" vs "one two three")
        elif (
            re.search(r"\w+", actual_answer)
            and re.search(r"\w+", expected_answer)
            and any(char.isdigit() for char in actual_answer)
            != any(char.isdigit() for char in expected_answer)
        ):
            return "Format Error"

        # Check if partial match with case sensitivity
        elif (
            actual_answer.lower() in expected_answer.lower()
            or expected_answer.lower() in actual_answer.lower()
        ):
            return "Partial Match / case Sensitivity"

        # Else the answer may be completely wrong
        else:
            return "Other"

    else:
        # Check if all the words in actual_answer are expected but some of the expected are missing
        if all(word in expected_answer for word in actual_answer.split()):
            return "Words Missing"

        # Check if some words in actual_answer are incorrect (present in actual_answer but not in expected_answer)
        # elif any(word not in expected_answer for word in actual_answer.split()):
        #   return "Incorrect Words"

        # Check if the entire actual_answer is completely different from the expected_answer
        else:
            return "Entire Answer Incorrect"


def is_synonym(word1, word2):
    if word2 is "yes" or "no" or "ja" or "nein":
        return False

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.wup_similarity(synset2) is not None:
                return True
    return False


# TODO should we use SQLite? An online database (REDIS)?
def get_result_file_path(file_name: str) -> str:
    """Returns the path to the result file.

    Args:
        file_name (str): The name of the result file

    Returns:
        str: The path to the result file
    """
    return f"benchmark/results/{file_name}.csv"
