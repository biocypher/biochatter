import pandas as pd
import pytest


def benchmark_already_executed(
    model_name: str,
    task: str,
    subtask: str,
) -> bool:
    """
    Checks if the benchmark task and subtask test case for the model_name have already
    been executed.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"
        subtask (str): The benchmark subtask test case, e.g. "0_entities"
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

    Returns:

        bool: True if the benchmark task and subtask for the model_name has
            already been run, False otherwise
    """
    task_results = return_or_create_result_file(task)
    task_results_subset = (task_results["model_name"] == model_name) & (
        task_results["subtask"] == subtask
    )
    return task_results_subset.any()


def skip_if_already_run(
    model_name: str,
    task: str,
    subtask: str,
) -> None:
    """Helper function to check if the test case is already executed.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        result_files (dict[str, pd.DataFrame]): The result files
        task (str): The benchmark task, e.g. "biocypher_query_generation"
        subtask (str): The benchmark subtask test case, e.g. "0_single_word"
    """
    if benchmark_already_executed(model_name, task, subtask):
        pytest.skip(
            f"benchmark {task}: {subtask} with {model_name} already executed"
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
        results = pd.DataFrame(columns=["model_name", "subtask", "score"])
        results.to_csv(file_path, index=False)
    return results


def write_results_to_file(
    model_name: str, subtask: str, score: str, file_path: str
):
    """Writes the benchmark results for the subtask to the result file.

    Args:
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        subtask (str): The benchmark subtask test case, e.g. "entities_0"
        score (str): The benchmark score, e.g. "1/1"
    """
    results = pd.read_csv(file_path, header=0)
    new_row = pd.DataFrame(
        [[model_name, subtask, score]], columns=results.columns
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
