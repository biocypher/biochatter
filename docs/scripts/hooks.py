import os
import re
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    overview = create_overview_table(result_files_path, result_file_names)

    plot_accuracy_per_model(overview)
    plot_accuracy_per_quantisation(overview)
    plot_accuracy_per_task(overview)
    plot_scatter_per_quantisation(overview)


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

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (
            row["passed_test_cases"] / row["number_test_cases"]
            if row["number_test_cases"] != 0
            else 0
        ),
        axis=1,
    )

    aggregated_scores["Full model name"] = (
        aggregated_scores.index.get_level_values("model_name")
    )
    aggregated_scores["Passed test cases"] = aggregated_scores[
        "passed_test_cases"
    ]
    aggregated_scores["Total test cases"] = aggregated_scores[
        "number_test_cases"
    ]
    aggregated_scores["Iterations"] = aggregated_scores["iterations"]
    new_order = [
        "Full model name",
        "Passed test cases",
        "Total test cases",
        "Accuracy",
        "Iterations",
    ]
    results = aggregated_scores[new_order]
    results = results.sort_values(by="Accuracy", ascending=False)
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
        f"{result_files_path}preprocessed_for_frontend/overview.csv",
        index=True,
    )

    overview_per_quantisation = overview
    overview_per_quantisation["Full model name"] = (
        overview_per_quantisation.index
    )
    overview_per_quantisation[
        ["Model name", "Size", "Version", "Quantisation"]
    ] = overview_per_quantisation["Full model name"].str.split(":", expand=True)
    # convert underscores in Size to commas
    overview_per_quantisation["Size"] = overview_per_quantisation[
        "Size"
    ].str.replace("_", ",")
    # add size 175 for gpt-3.5-turbo and Unknown for gpt-4
    overview_per_quantisation["Size"] = overview_per_quantisation.apply(
        lambda row: (
            "175" if row["Model name"] == "gpt-3.5-turbo" else row["Size"]
        ),
        axis=1,
    )
    overview_per_quantisation["Size"] = overview_per_quantisation.apply(
        lambda row: "Unknown" if row["Model name"] == "gpt-4" else row["Size"],
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
    overview_per_quantisation.loc[:, "Median Accuracy"] = (
        overview_per_quantisation["Median Accuracy"].round(2)
    )
    overview_per_quantisation.loc[:, "SD"] = overview_per_quantisation[
        "SD"
    ].round(2)
    overview_per_quantisation.to_csv(
        f"{result_files_path}preprocessed_for_frontend/overview-quantisation.csv",
        index=False,
    )

    # group by model name and size, aggregate different quantisations
    # keep models that do not have sizes
    overview_per_size = overview_per_quantisation.groupby(
        ["Model name", "Size"]
    ).agg(
        {
            "Median Accuracy": "median",
            "SD": "mean",
        }
    )
    # round mean and SD to 2 decimal places
    overview_per_size["Median Accuracy"] = overview_per_size[
        "Median Accuracy"
    ].round(2)
    overview_per_size["SD"] = overview_per_size["SD"].round(2)
    # sort by mean, descending
    overview_per_size = overview_per_size.sort_values(
        by="Median Accuracy", ascending=False
    )
    overview_per_size.to_csv(
        f"{result_files_path}preprocessed_for_frontend/overview-model.csv",
        index=True,
    )

    return overview


def plot_accuracy_per_model(overview) -> None:
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model name", y="Accuracy", hue="Size", data=overview_melted)
    plt.title(
        "Boxplot across tasks, per Model, coloured by size (billions of parameters)"
    )
    plt.ylim(-0.1, 1.1)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(0, 0), loc="lower left")
    plt.savefig(
        f"docs/images/boxplot-per-model.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_accuracy_per_quantisation(overview) -> None:
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Model name", y="Accuracy", hue="Quantisation", data=overview_melted
    )
    plt.title("Boxplot across tasks, per Quantisation")
    plt.xticks(rotation=45)
    plt.savefig(
        f"docs/images/boxplot-per-quantisation.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_accuracy_per_task(overview):
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Task", y="Accuracy", hue="Model name", data=overview_melted)
    plt.title("Boxplot across models, per Task")
    plt.xticks(rotation=45)
    plt.savefig(
        f"docs/images/boxplot-per-task.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_scatter_per_quantisation(overview):
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    # order x axis quantisation values numerically
    overview_melted["Quantisation"] = pd.Categorical(
        overview_melted["Quantisation"],
        categories=[
            "None",
            "2-bit",
            "3-bit",
            "4-bit",
            "5-bit",
            "6-bit",
            "8-bit",
            ">= 16-bit*",
        ],
        ordered=True,
    )
    overview_melted["Size"] = pd.Categorical(
        overview_melted["Size"],
        categories=[
            "Unknown",
            "175",
            "70",
            "46,7",
            "34",
            "13",
            "7",
            "6",
        ],
        ordered=True,
    )
    sns.scatterplot(
        x="Quantisation",
        y="Mean Accuracy",
        hue="Model name",
        size="Size",
        sizes=(10, 300),
        data=overview_melted,
        alpha=0.5,
    )
    plt.ylim(0, 1.1)
    plt.title(
        "Scatter plot across models, per quantisation, coloured by model name, size by model size (billions of parameters)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.savefig(
        f"docs/images/scatter-per-quantisation-name.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        f"docs/images/scatter-per-quantisation-name.svg",
        bbox_inches="tight",
    )
    plt.close()


def melt_and_process(overview):
    overview_melted = overview.melt(
        id_vars=[
            "Full model name",
            "Model name",
            "Size",
            "Version",
            "Quantisation",
            "Mean Accuracy",
            "Median Accuracy",
            "SD",
        ],
        var_name="Task",
        value_name="Accuracy",
    )
    # unify quantisation names: 2-bit, 3-bit, etc
    digit_pattern = r"\d+"
    overview_melted["Quantisation"] = overview_melted["Quantisation"].apply(
        lambda x: f"{re.findall(digit_pattern, x)[0]}-bit" if x else "None"
    )
    # set quantisation of gpt models to None
    overview_melted["Quantisation"] = overview_melted.apply(
        lambda row: (
            ">= 16-bit*"
            if row["Model name"] in ["gpt-3.5-turbo", "gpt-4"]
            else row["Quantisation"]
        ),
        axis=1,
    )

    return overview_melted


if __name__ == "__main__":
    on_pre_build(None)
