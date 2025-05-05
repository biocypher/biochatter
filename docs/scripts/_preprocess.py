import os

import numpy as np
import pandas as pd


def preprocess_results_for_frontend(
    raw_results: pd.DataFrame,
    path: str,
    file_name: str,
) -> None:
    """Preprocesses the results for the frontend and store them on disk.

    Args:
    ----
        raw_results (pd.DataFrame): The raw results.
        path (str): The path to the result files.
        file_name (str): The file name of the result file.

    """
    raw_results["score_possible"] = raw_results.apply(
        lambda x: float(x["score"].split("/")[1]) * x["iterations"],
        axis=1,
    )
    raw_results["scores"] = raw_results["score"].apply(
        lambda x: x.split("/")[0],
    )
    raw_results["score_achieved"] = raw_results["scores"].apply(
        lambda x: (np.sum([float(score) for score in x.split(";")]) if ";" in x else float(x)),
    )
    # multiply score_achieved by iterations if no semicolon in scores
    # TODO remove once all benchmarks are in new format
    raw_results["score_achieved"] = raw_results.apply(
        lambda x: (x["score_achieved"] * x["iterations"] if ";" not in x["scores"] else x["score_achieved"]),
        axis=1,
    )
    raw_results["score_sd"] = raw_results["scores"].apply(
        lambda x: (np.std([float(score) for score in x.split(";")], ddof=1) if ";" in x else 0),
    )
    aggregated_scores = raw_results.groupby(["model_name"]).agg(
        {
            "score_possible": "sum",
            "score_achieved": "sum",
            "score_sd": "sum",
            "iterations": "first",
        },
    )

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (row["score_achieved"] / row["score_possible"] if row["score_possible"] != 0 else 0),
        axis=1,
    )

    aggregated_scores["Full model name"] = aggregated_scores.index.get_level_values("model_name")
    aggregated_scores["Score achieved"] = aggregated_scores["score_achieved"]
    aggregated_scores["Score possible"] = aggregated_scores["score_possible"]
    aggregated_scores["Score SD"] = aggregated_scores["score_sd"]
    aggregated_scores["Iterations"] = aggregated_scores["iterations"]
    new_order = [
        "Full model name",
        "Score achieved",
        "Score possible",
        "Score SD",
        "Accuracy",
        "Iterations",
    ]
    results = aggregated_scores[new_order]
    results = results.sort_values(by="Accuracy", ascending=False)
    results.to_csv(
        f"{path}processed/{file_name}",
        index=False,
    )

    if file_name == "sourcedata_info_extraction.csv":
        write_individual_extraction_task_results(raw_results)


def write_individual_extraction_task_results(raw_results: pd.DataFrame) -> None:
    """Write one csv file to disk per subtask.

    Writes individual files for sourcedata_info_extraction results in the
    same format as the other results files.
    """
    raw_results["subtask"] = raw_results["subtask"].apply(
        lambda x: x.split(":")[1],
    )
    raw_results["score_possible"] = raw_results.apply(
        lambda x: float(x["score"].split("/")[1]) * x["iterations"],
        axis=1,
    )
    raw_results["scores"] = raw_results["score"].apply(
        lambda x: x.split("/")[0],
    )
    raw_results["score_achieved"] = raw_results["scores"].apply(
        lambda x: (np.sum([float(score) for score in x.split(";")]) if ";" in x else float(x)),
    )
    raw_results["score_sd"] = raw_results["scores"].apply(
        lambda x: (np.std([float(score) for score in x.split(";")], ddof=1) if ";" in x else 0),
    )
    aggregated_scores = raw_results.groupby(["model_name", "subtask"]).agg(
        {
            "score_possible": "sum",
            "score_achieved": "sum",
            "score_sd": "mean",
            "iterations": "first",
        },
    )

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (row["score_achieved"] / row["score_possible"] if row["score_possible"] != 0 else 0),
        axis=1,
    )

    aggregated_scores["Full model name"] = aggregated_scores.index.get_level_values("model_name")
    aggregated_scores["Subtask"] = aggregated_scores.index.get_level_values(
        "subtask",
    )
    aggregated_scores["Score achieved"] = aggregated_scores["score_achieved"]
    aggregated_scores["Score possible"] = aggregated_scores["score_possible"]
    aggregated_scores["Score SD"] = aggregated_scores["score_sd"]
    aggregated_scores["Iterations"] = aggregated_scores["iterations"]
    new_order = [
        "Full model name",
        "Subtask",
        "Score achieved",
        "Score possible",
        "Score SD",
        "Accuracy",
        "Iterations",
    ]
    results = aggregated_scores[new_order]
    results = results.sort_values(by="Accuracy", ascending=False)
    for subtask in results["Subtask"].unique():
        results_subtask = results[results["Subtask"] == subtask]
        results_subtask.to_csv(
            f"benchmark/results/processed/extraction_{subtask}.csv",
            index=False,
        )


def create_overview_table(result_files_path: str, result_file_names: list[str]) -> pd.DataFrame:
    """Create an overview table of benchmark results.

    Creates a table for visualisation on the website with y-axis = models and
    x-axis = tasks.

    Args:
    ----
        result_files_path (str): The path to the result files.
        result_file_names (List[str]): The names of the result files.

    """
    subtask_results = []
    for file in result_file_names:
        subtask_result = pd.read_csv(f"{result_files_path}processed/{file}")
        file_name_without_extension = os.path.splitext(file)[0]
        subtask_result[file_name_without_extension] = subtask_result["Accuracy"]
        subtask_result.set_index("Full model name", inplace=True)
        subtask_result = subtask_result[file_name_without_extension]
        subtask_results.append(subtask_result)
    overview = pd.concat(subtask_results, axis=1)
    overview["Mean Accuracy"] = overview.mean(axis=1)
    overview["Median Accuracy"] = overview.median(axis=1)
    overview["SD"] = overview.std(axis=1)
    overview = overview.sort_values(by="Median Accuracy", ascending=False)
    # split "Full model name" at : to get Model name, size, version, and quantisation
    overview.to_csv(
        f"{result_files_path}processed/overview.csv",
        index=True,
    )

    overview_per_quantisation = overview
    overview_per_quantisation["Full model name"] = overview_per_quantisation.index
    overview_per_quantisation[["Model name", "Size", "Version", "Quantisation"]] = overview_per_quantisation[
        "Full model name"
    ].str.split(":", expand=True)
    # convert underscores in Size to commas
    overview_per_quantisation["Size"] = overview_per_quantisation["Size"].str.replace("_", ",")
    # add size 175 for gpt-3.5-turbo and Unknown for gpt-4
    overview_per_quantisation["Size"] = overview_per_quantisation.apply(
        lambda row: ("175" if "gpt-3.5-turbo" in row["Model name"] else row["Size"]),
        axis=1,
    )
    overview_per_quantisation["Size"] = overview_per_quantisation.apply(
        lambda row: ("Unknown" if "gpt-4" in row["Model name"] or "claude" in row["Model name"] else row["Size"]),
        axis=1,
    )
    overview_per_quantisation = overview_per_quantisation[
        [
            "Model name",
            "Size",
            "Version",
            "Quantisation",
            "Median Accuracy",
            "SD",
        ]
    ]
    # round mean and sd to 2 decimal places
    overview_per_quantisation.loc[:, "Median Accuracy"] = overview_per_quantisation["Median Accuracy"].round(2)
    overview_per_quantisation.loc[:, "SD"] = overview_per_quantisation["SD"].round(2)
    overview_per_quantisation.to_csv(
        f"{result_files_path}processed/overview-quantisation.csv",
        index=False,
    )

    # group by model name and size, aggregate different quantisations
    # keep models that do not have sizes
    overview_per_size = overview_per_quantisation.groupby(
        ["Model name", "Size"],
    ).agg(
        {
            "Median Accuracy": "median",
            "SD": "mean",
        },
    )
    # round mean and SD to 2 decimal places
    overview_per_size["Median Accuracy"] = overview_per_size["Median Accuracy"].round(2)
    overview_per_size["SD"] = overview_per_size["SD"].round(2)
    # sort by mean, descending
    overview_per_size = overview_per_size.sort_values(
        by="Median Accuracy",
        ascending=False,
    )
    overview_per_size.to_csv(
        f"{result_files_path}processed/overview-model.csv",
        index=True,
    )

    return overview
