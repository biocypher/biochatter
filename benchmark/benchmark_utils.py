import pytest

import pandas as pd
from datetime import datetime


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
            ]
        )
        results.to_csv(file_path, index=False)
    return results


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
    new_row = pd.DataFrame(
        [[model_name, subtask, score, iterations, md5_hash, now]],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"]
    )
    results.to_csv(file_path, index=False)


# TODO should we use SQLite? An online database (REDIS)?
def get_result_file_path(file_name: str) -> str:
    """Returns the path to the result file.

    Args:
        file_name (str): The name of the result file

    Returns:
        str: The path to the result file
    """
    return f"benchmark/results/{file_name}.csv"
