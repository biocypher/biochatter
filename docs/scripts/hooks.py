import re

import pandas as pd


def on_pre_build(config, **kwargs) -> None:
    """This function is called when building the documentation."""

    result_files_path = "benchmark/results/"
    result_file_names = ["biocypher_query_generation", "rag_interpretation"]

    for file_name in result_file_names:
        results = pd.read_csv(f"{result_files_path}{file_name}.csv")
        preprocess_results_for_frontend(results, result_files_path, file_name)


def preprocess_results_for_frontend(
    raw_results: pd.DataFrame, path: str, file_name: str
) -> None:
    """Preprocesses the results for the frontend and store them on disk.

    Args:
        raw_results (pd.DataFrame): The raw results.
        path (str): The path to the result files.
        file_name (str): The file name of the result file.
    """
    raw_results["subtask"] = raw_results["subtask"].apply(
        extract_string_before_number
    )
    raw_results["number_test_cases"] = raw_results["score"].apply(
        lambda x: float(x.split("/")[1])
    )
    raw_results["passed_test_cases"] = raw_results["score"].apply(
        lambda x: float(x.split("/")[0])
    )
    aggregated_scores = raw_results.groupby(["model_name", "subtask"]).sum()
    aggregated_scores["Score"] = (
        aggregated_scores["passed_test_cases"]
        / aggregated_scores["number_test_cases"]
    )
    dfs_for_each_subtask = {
        subtask: group
        for subtask, group in aggregated_scores.groupby("subtask")
    }
    for subtask, results in dfs_for_each_subtask.items():
        results["Model name"] = results.index.get_level_values("model_name")
        results["Passed test cases"] = results["passed_test_cases"]
        results["Total test cases"] = results["number_test_cases"]
        new_order = [
            "Model name",
            "Passed test cases",
            "Total test cases",
            "Score",
        ]
        results = results[new_order]
        results.to_csv(
            f"{path}preprocessed_for_frontend/{file_name}_{subtask}.csv",
            index=False,
        )


def extract_string_before_number(input_string: str) -> str:
    """Extracts the string before the first number in a string.
    e.g. "subtask_name_1_some_text" should return "subtask_name"

    Args:
        input_string (str): The input string.

    Returns:
        str: The extracted string.
    """
    match = re.search(r"(.+?)_\d+", input_string)
    if match:
        return match.group(1)
    else:
        return input_string
