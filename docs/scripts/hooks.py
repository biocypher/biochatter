import os
import re

import pandas as pd


def on_pre_build(config, **kwargs) -> None:
    """This function is called when building the documentation."""

    result_files_path = "benchmark/results/"

    result_file_names = [
        f
        for f in os.listdir(result_files_path)
        if os.path.isfile(os.path.join(result_files_path, f))
    ]

    for file_name in result_file_names:
        results = pd.read_csv(f"{result_files_path}{file_name}")
        preprocess_results_for_frontend(results, result_files_path, file_name)

    create_overview_table(result_files_path, result_file_names)


def extract_string_after_number(input_string: str) -> str:
    """Extracts the string before the first number in a string.
    e.g. "subtask_name_1_some_text" should return "subtask_name"

    Args:
        input_string (str): The input string.

    Returns:
        str: The extracted string.
    """
    match = re.search(r"\d+_(.+)", input_string)
    if match:
        return match.group(1)
    else:
        return input_string


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
        extract_string_after_number
    )
    raw_results["number_test_cases"] = raw_results["score"].apply(
        lambda x: float(x.split("/")[1])
    )
    raw_results["passed_test_cases"] = raw_results["score"].apply(
        lambda x: float(x.split("/")[0])
    )
    aggregated_scores = raw_results.groupby(["model_name"]).agg(
        {
            "number_test_cases": "sum",
            "passed_test_cases": "sum",
            "iterations": "first",
        }
    )

    aggregated_scores["Score"] = (
        aggregated_scores["passed_test_cases"]
        / aggregated_scores["number_test_cases"]
    )

    aggregated_scores["Model name"] = aggregated_scores.index.get_level_values(
        "model_name"
    )
    aggregated_scores["Passed test cases"] = aggregated_scores[
        "passed_test_cases"
    ]
    aggregated_scores["Total test cases"] = aggregated_scores[
        "number_test_cases"
    ]
    aggregated_scores["Iterations"] = aggregated_scores["iterations"]
    new_order = [
        "Model name",
        "Passed test cases",
        "Total test cases",
        "Score",
        "Iterations",
    ]
    results = aggregated_scores[new_order]
    results = results.sort_values(by="Score", ascending=False)
    results.to_csv(
        f"{path}preprocessed_for_frontend/{file_name}",
        index=False,
    )


def create_overview_table(result_files_path: str, result_file_names: list[str]):
    """

    Creates an overview table for the frontend with y-axis = models and x-axis =
    tasks.

    Args:
        result_files_path (str): The path to the result files.
        result_file_names (list[str]): The names of the result files.
    """
    subtask_results = []
    for file in result_file_names:
        subtask_result = pd.read_csv(
            f"{result_files_path}preprocessed_for_frontend/{file}"
        )
        file_name_without_extension = os.path.splitext(file)[0]
        subtask_result[file_name_without_extension] = subtask_result["Score"]
        subtask_result.set_index("Model name", inplace=True)
        subtask_result = subtask_result[file_name_without_extension]
        subtask_results.append(subtask_result)
    overview = pd.concat(subtask_results, axis=1)
    overview["Mean"] = overview.mean(axis=1)
    overview = overview.sort_values(by="Mean", ascending=False)
    overview.to_csv(
        f"{result_files_path}preprocessed_for_frontend/overview.csv", index=True
    )
    overview_aggregated = overview[["Mean"]]
    overview_aggregated.to_csv(
        f"{result_files_path}preprocessed_for_frontend/overview-aggregated.csv",
        index=True,
    )
