import pandas as pd

from benchmark.conftest import RESULT_FILES


def benchmark_already_executed(
    task: str,
    subtask: str,
    model_name: str,
    result_files: dict[str, pd.DataFrame],
) -> bool:
    """
    Checks if the benchmark task and subtask test case for the model_name have already
    been executed.

    Args:
        task (str): The benchmark task, e.g. "biocypher_query_generation"
        subtask (str): The benchmark subtask test case, e.g. "entities_0"
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        result_files (dict[str, pd.DataFrame]): The result files

    Returns:

        bool: True if the benchmark task and subtask for the model_name has
            already been run, False otherwise
    """
    task_results = result_files[f"benchmark/results/{task}.csv"]
    task_results_subset = (task_results["model"] == model_name) & (
        task_results["subtask"] == subtask
    )
    return task_results_subset.any()


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
        by="subtask"
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
    file_path = [file for file in RESULT_FILES if file_name in file]
    if not file_path:
        raise ValueError(f"Could not find result file {file_name}")
    return file_path[0]
