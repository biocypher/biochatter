import os
import re
import math

import seaborn as sns
import colorcet as cc
import matplotlib

import numpy as np

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
        and f.endswith(".csv")
        and not "failure_mode" in f
        and not "confidence" in f
    ]

    for file_name in result_file_names:
        results = pd.read_csv(f"{result_files_path}{file_name}")
        preprocess_results_for_frontend(results, result_files_path, file_name)

    overview = create_overview_table(result_files_path, result_file_names)

    plot_text2cypher()
    plot_image_caption_confidence()
    plot_medical_exam()
    plot_extraction_tasks()
    plot_scatter_per_quantisation(overview)
    plot_accuracy_per_model(overview)
    plot_accuracy_per_quantisation(overview)
    plot_accuracy_per_task(overview)
    plot_task_comparison(overview)
    plot_rag_tasks(overview)
    plot_comparison_naive_biochatter(overview)
    calculate_stats(overview)


def plot_text2cypher():
    """

    Get entity_selection, relationship_selection, property_selection,
    property_exists, query_generation, and end_to_end_query_generation results
    files, combine and preprocess them and plot the accuracy for each model as a
    boxplot.

    """
    entity_selection = pd.read_csv("benchmark/results/entity_selection.csv")
    entity_selection["task"] = "entity_selection"
    relationship_selection = pd.read_csv(
        "benchmark/results/relationship_selection.csv"
    )
    relationship_selection["task"] = "relationship_selection"
    property_selection = pd.read_csv("benchmark/results/property_selection.csv")
    property_selection["task"] = "property_selection"
    property_exists = pd.read_csv("benchmark/results/property_exists.csv")
    property_exists["task"] = "property_exists"
    query_generation = pd.read_csv("benchmark/results/query_generation.csv")
    query_generation["task"] = "query_generation"
    end_to_end_query_generation = pd.read_csv(
        "benchmark/results/end_to_end_query_generation.csv"
    )
    end_to_end_query_generation["task"] = "end_to_end_query_generation"

    # combine all results
    results = pd.concat(
        [
            entity_selection,
            relationship_selection,
            property_selection,
            property_exists,
            query_generation,
            end_to_end_query_generation,
        ]
    )

    # calculate accuracy
    results["score_possible"] = results["score"].apply(
        lambda x: float(x.split("/")[1])
    )
    results["scores"] = results["score"].apply(lambda x: x.split("/")[0])
    results["score_achieved"] = results["scores"].apply(
        lambda x: (
            np.mean([float(score) for score in x.split(";")])
            if ";" in x
            else float(x)
        )
    )
    results["accuracy"] = results["score_achieved"] / results["score_possible"]
    results["score_sd"] = results["scores"].apply(
        lambda x: (
            np.std([float(score) for score in x.split(";")], ddof=1)
            if ";" in x
            else 0
        )
    )

    results["model"] = results["model_name"].apply(lambda x: x.split(":")[0])
    # create labels: openhermes, llama-3, gpt, based on model name, for all
    # other models, use "other open source"
    results["model_family"] = results["model"].apply(
        lambda x: (
            "openhermes"
            if "openhermes" in x
            else (
                "llama-3"
                if "llama-3" in x
                else "gpt" if "gpt" in x else "other open source"
            )
        )
    )

    # order task by median accuracy ascending
    task_order = (
        results.groupby("task")["accuracy"].median().sort_values().index
    )

    # order model_family by median accuracy ascending within each task
    results["model_family"] = results["model_family"].astype(
        pd.CategoricalDtype(
            categories=["other open source", "llama-3", "openhermes", "gpt"],
            ordered=True,
        )
    )

    # plot results per task
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.xticks(rotation=45, ha="right")
    sns.boxplot(
        x="task",
        y="accuracy",
        hue="model_family",
        data=results,
        order=task_order,
    )
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(
        "docs/images/boxplot-text2cypher.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_image_caption_confidence():
    """
    Get multimodal_answer_confidence.csv file, preprocess it and plot the
    confidence scores for correct and incorrect answers as histograms. Correct
    answer confidence values are in the correct_confidence column and incorrect
    answer confidence values are in the incorrect_confidence column; both
    columns contain individual confidence values (integers between 1 and 10)
    separated by semicolons.
    """
    results = pd.read_csv("benchmark/results/multimodal_answer_confidence.csv")
    correct_values = results["correct_confidence"].to_list()
    incorrect_values = results["incorrect_confidence"].to_list()
    # flatten lists of confidence values
    correct_values = [
        int(value) for sublist in correct_values for value in sublist.split(";")
    ]
    for value in list(incorrect_values):
        if math.isnan(value):
            incorrect_values.remove(value)
        if isinstance(value, float):
            continue
        if ";" in value:
            incorrect_values.remove(value)
            incorrect_values.extend([int(val) for val in value.split(";")])

    incorrect_values = [int(value) for value in incorrect_values]

    # plot histograms of both correct and incorrect confidence values with
    # transparency, correct green, incorrect red
    plt.figure(figsize=(6, 4))
    plt.hist(
        [correct_values, incorrect_values],
        bins=range(1, 12),
        color=["green", "red"],
        label=["Correct", "Incorrect"],
    )
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig(
        "docs/images/histogram-image-caption-confidence.png",
        bbox_inches="tight",
        dpi=300,
    )


def preprocess_results_for_frontend(
    raw_results: pd.DataFrame, path: str, file_name: str
) -> None:
    """Preprocesses the results for the frontend and store them on disk.

    Args:
        raw_results (pd.DataFrame): The raw results.
        path (str): The path to the result files.
        file_name (str): The file name of the result file.
    """
    raw_results["score_possible"] = raw_results.apply(
        lambda x: float(x["score"].split("/")[1]) * x["iterations"], axis=1
    )
    raw_results["scores"] = raw_results["score"].apply(
        lambda x: x.split("/")[0]
    )
    raw_results["score_achieved"] = raw_results["scores"].apply(
        lambda x: (
            np.sum([float(score) for score in x.split(";")])
            if ";" in x
            else float(x)
        )
    )
    # multiply score_achieved by iterations if no semicolon in scores
    # TODO remove once all benchmarks are in new format
    raw_results["score_achieved"] = raw_results.apply(
        lambda x: (
            x["score_achieved"] * x["iterations"]
            if ";" not in x["scores"]
            else x["score_achieved"]
        ),
        axis=1,
    )
    raw_results["score_sd"] = raw_results["scores"].apply(
        lambda x: (
            np.std([float(score) for score in x.split(";")], ddof=1)
            if ";" in x
            else 0
        )
    )
    aggregated_scores = raw_results.groupby(["model_name"]).agg(
        {
            "score_possible": "sum",
            "score_achieved": "sum",
            "score_sd": "sum",
            "iterations": "first",
        }
    )

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (
            row["score_achieved"] / row["score_possible"]
            if row["score_possible"] != 0
            else 0
        ),
        axis=1,
    )

    aggregated_scores["Full model name"] = (
        aggregated_scores.index.get_level_values("model_name")
    )
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
    """
    Write one csv file per subtask for sourcedata_info_extraction results in the
    same format as the other results files.
    """
    raw_results["subtask"] = raw_results["subtask"].apply(
        lambda x: x.split(":")[1]
    )
    raw_results["score_possible"] = raw_results.apply(
        lambda x: float(x["score"].split("/")[1]) * x["iterations"], axis=1
    )
    raw_results["scores"] = raw_results["score"].apply(
        lambda x: x.split("/")[0]
    )
    raw_results["score_achieved"] = raw_results["scores"].apply(
        lambda x: (
            np.sum([float(score) for score in x.split(";")])
            if ";" in x
            else float(x)
        )
    )
    raw_results["score_sd"] = raw_results["scores"].apply(
        lambda x: (
            np.std([float(score) for score in x.split(";")], ddof=1)
            if ";" in x
            else 0
        )
    )
    aggregated_scores = raw_results.groupby(["model_name", "subtask"]).agg(
        {
            "score_possible": "sum",
            "score_achieved": "sum",
            "score_sd": "mean",
            "iterations": "first",
        }
    )

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (
            row["score_achieved"] / row["score_possible"]
            if row["score_possible"] != 0
            else 0
        ),
        axis=1,
    )

    aggregated_scores["Full model name"] = (
        aggregated_scores.index.get_level_values("model_name")
    )
    aggregated_scores["Subtask"] = aggregated_scores.index.get_level_values(
        "subtask"
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


def create_overview_table(result_files_path: str, result_file_names: list[str]):
    """

    Creates an overview table for the frontend with y-axis = models and x-axis =
    tasks.

    Args:
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
            "175" if "gpt-3.5-turbo" in row["Model name"] else row["Size"]
        ),
        axis=1,
    )
    overview_per_quantisation["Size"] = overview_per_quantisation.apply(
        lambda row: (
            "Unknown"
            if "gpt-4" in row["Model name"] or "claude" in row["Model name"]
            else row["Size"]
        ),
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
        f"{result_files_path}processed/overview-quantisation.csv",
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
        f"{result_files_path}processed/overview-model.csv",
        index=True,
    )

    return overview


def plot_accuracy_per_model(overview) -> None:
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        x="Model name", y="Accuracy", hue="Size", data=overview_melted
    )
    plt.title(
        "Strip plot across tasks, per Model, coloured by size (billions of parameters)"
    )
    plt.ylim(-0.1, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(0, 0), loc="lower left")
    plt.savefig(
        "docs/images/stripplot-per-model.png",
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
        "docs/images/boxplot-per-quantisation.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_accuracy_per_task(overview):
    overview_melted = melt_and_process(overview)

    # concatenate model name and quantisation
    overview_melted["Coarse model name"] = overview_melted[
        "Model name"
    ].replace(
        {
            "gpt-3.5-turbo-0613": "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
            "gpt-4-0613": "gpt-4",
            "gpt-4-0125-preview": "gpt-4",
            "gpt-4o-2024-05-13": "gpt-4",
        },
        regex=True,
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))

    # Define the color palette
    palette = sns.color_palette(
        "Set1", len(overview_melted["Coarse model name"].unique())
    )

    # Calculate mean accuracy for each task
    task_order = (
        overview_melted.groupby("Task")["Accuracy"]
        .mean()
        .sort_values()
        .index[::-1]
    )

    # Sort the dataframe
    overview_melted["Task"] = pd.Categorical(
        overview_melted["Task"], categories=task_order, ordered=True
    )
    overview_melted = overview_melted.sort_values("Task")

    sns.stripplot(
        x="Task",
        y="Accuracy",
        hue="Coarse model name",
        data=overview_melted,
        dodge=True,
        palette=palette,
        jitter=0.2,
        alpha=0.8,
    )

    sns.lineplot(
        x="Task",
        y="Accuracy",
        hue="Coarse model name",
        data=overview_melted,
        sort=False,
        legend=False,
        palette=palette,
        alpha=0.3,
    )

    # Get current axis
    ax = plt.gca()

    # Add vertical lines at each x tick
    for x in ax.get_xticks():
        ax.axvline(x=x, color="gray", linestyle="--", lw=0.5)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper right")
    plt.title("Dot plot across models / quantisations, per Task")
    plt.xticks(rotation=45)
    plt.savefig(
        "docs/images/dotplot-per-task.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/dotplot-per-task.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_scatter_per_quantisation(overview):
    overview_melted = melt_and_process(overview)

    # remove individual task accuracy columns
    overview_melted = overview_melted[
        [
            "Model name",
            "Size",
            "Quantisation",
            "Mean Accuracy",
            "Median Accuracy",
            "SD",
        ]
    ]

    # deduplicate remaining rows
    overview_melted = overview_melted.drop_duplicates()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    # order x axis quantisation values numerically
    overview_melted["Quantisation"] = pd.Categorical(
        overview_melted["Quantisation"],
        categories=[
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
            "8",
            "7",
            "6",
        ],
        ordered=True,
    )

    # Add jitter to x-coordinates
    x = pd.Categorical(overview_melted["Quantisation"]).codes.astype(float)

    # Create a mask for 'openhermes' and closed models
    mask_openhermes = overview_melted["Model name"] == "openhermes-2.5"
    mask_closed = overview_melted["Model name"].str.contains(
        "gpt|claude", case=False, regex=True
    )

    # Do not add jitter for 'openhermes' model
    x[mask_openhermes] += 0

    # Manually enter jitter values for closed models
    jitter_values = {
        "gpt-3": -0.2,
        "gpt-4": 0.2,
        "claude-3-opus-20240229": -0.05,
        "claude-3-5-sonnet-20240620": 0.05,
    }

    for model, jitter in jitter_values.items():
        mask_model = overview_melted["Model name"].str.contains(model)
        x[mask_model] += jitter

    # For other models, add the original jitter
    x[~mask_openhermes & ~mask_closed] += np.random.normal(
        0, 0.1, size=len(x[~mask_openhermes & ~mask_closed])
    )

    # Define the order of model names
    model_names_order = [
        "chatglm3",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "code-llama-instruct",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-0125",
        "gpt-4-0613",
        "gpt-4-0125-preview",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "llama-2-chat",
        "llama-3-instruct",
        "llama-3.1-instruct",
        "mixtral-instruct-v0.1",
        "mistral-instruct-v0.2",
        "openhermes-2.5",
    ]

    # Define the order of sizes
    size_order = [
        "Unknown",
        "175",
        "70",
        "46,7",
        "34",
        "13",
        "8",
        "7",
        "6",
    ]

    # Create a ColorBrewer palette
    palette = sns.color_palette(cc.glasbey, n_colors=len(model_names_order))

    # Define a dictionary mapping model names to colors using the order list
    color_dict = {
        model: palette[i] for i, model in enumerate(model_names_order)
    }

    # Use the dictionary as the palette argument in sns.scatterplot
    ax = sns.scatterplot(
        x=x,
        y="Median Accuracy",
        hue="Model name",
        size="Size",
        sizes=(10, 300),
        data=overview_melted,
        palette=color_dict,  # Use the color dictionary here
        alpha=0.5,
    )

    # Reorder the legend using the same order list
    handles, labels = ax.get_legend_handles_labels()
    order = (
        ["Model name"]
        + [name for name in model_names_order if name in labels]
        + ["Size"]
        + [size for size in size_order if size in labels]
    )
    order_indices = [labels.index(name) for name in order if name in labels]
    plt.legend(
        [handles[idx] for idx in order_indices],
        [labels[idx] for idx in order_indices],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.ylim(0, 1)
    plt.xticks(
        ticks=range(len(overview_melted["Quantisation"].unique())),
        labels=overview_melted["Quantisation"].cat.categories,
    )
    plt.title(
        "Scatter plot across models, per quantisation, coloured by model name, size by model size (billions of parameters)"
    )
    plt.xticks(rotation=45)
    plt.savefig(
        "docs/images/scatter-per-quantisation-name.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/scatter-per-quantisation-name.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_task_comparison(overview):
    overview_melted = melt_and_process(overview)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Task",
        y="Accuracy",
        hue="Task",
        data=overview_melted,
    )
    plt.xticks(rotation=45, ha="right")
    plt.savefig(
        "docs/images/boxplot-tasks.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_rag_tasks(overview):
    overview_melted = melt_and_process(overview)

    # select tasks naive_query_generation_using_schema and query_generation
    overview_melted = overview_melted[
        overview_melted["Task"].isin(
            [
                "explicit_relevance_of_single_fragments",
                "implicit_relevance_of_multiple_fragments",
            ]
        )
    ]

    # order models by median accuracy, inverse
    overview_melted["Model name"] = pd.Categorical(
        overview_melted["Model name"],
        categories=overview_melted.groupby("Model name")["Median Accuracy"]
        .median()
        .sort_values()
        .index[::-1],
        ordered=True,
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.stripplot(
        x="Model name",
        y="Accuracy",
        hue="Quantisation",
        data=overview_melted,
        jitter=0.2,
        alpha=0.8,
    )
    plt.xlabel(None)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1, 0), loc="lower left")
    # Get current axis
    ax = plt.gca()

    # Add vertical lines at each x tick
    for x in ax.get_xticks():
        ax.axvline(x=x, color="gray", linestyle="--", lw=0.5)

    plt.savefig(
        "docs/images/stripplot-rag-tasks.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/stripplot-rag-tasks.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_extraction_tasks():
    """
    Load raw result file for sourcedata_info_extraction; aggregate based on the
    subtask name and calculate mean accuracy for each model. Plot a stripplot
    of the mean accuracy across models, coloured by subtask.
    """
    sourcedata_info_extraction = pd.read_csv(
        "benchmark/results/sourcedata_info_extraction.csv"
    )
    # split subtask at colon and use second element
    sourcedata_info_extraction["subtask"] = sourcedata_info_extraction[
        "subtask"
    ].apply(lambda x: x.split(":")[1])
    sourcedata_info_extraction["score_possible"] = sourcedata_info_extraction[
        "score"
    ].apply(lambda x: float(x.split("/")[1]))
    sourcedata_info_extraction["scores"] = sourcedata_info_extraction[
        "score"
    ].apply(lambda x: x.split("/")[0])
    sourcedata_info_extraction["score_achieved"] = sourcedata_info_extraction[
        "scores"
    ].apply(lambda x: np.mean(float(x.split(";")[0])) if ";" in x else float(x))
    sourcedata_info_extraction["score_sd"] = sourcedata_info_extraction[
        "scores"
    ].apply(lambda x: np.std(float(x.split(";")[0])) if ";" in x else 0)
    aggregated_scores = sourcedata_info_extraction.groupby(
        ["model_name", "subtask"]
    ).agg(
        {
            "score_possible": "sum",
            "score_achieved": "sum",
            "score_sd": "first",
            "iterations": "first",
        }
    )

    aggregated_scores["Accuracy"] = aggregated_scores.apply(
        lambda row: (
            row["score_achieved"] / row["score_possible"]
            if row["score_possible"] != 0
            else 0
        ),
        axis=1,
    )

    aggregated_scores["Full model name"] = (
        aggregated_scores.index.get_level_values("model_name")
    )
    aggregated_scores["Subtask"] = aggregated_scores.index.get_level_values(
        "subtask"
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

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.stripplot(
        x="Subtask",
        y="Accuracy",
        hue="Full model name",
        data=results,
    )

    plt.title("Strip plot across models, per subtask, coloured by model name")
    plt.ylim(-0.1, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(
        "docs/images/stripplot-extraction-tasks.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_medical_exam():
    """
    Load raw result for medical_exam; aggregate based on the language and
    calculate mean accuracy for each model. Plot a stripplot of the mean
    accuracy across models, coloured by language.
    """
    medical_exam = pd.read_csv("benchmark/results/medical_exam.csv")

    medical_exam["score_possible"] = medical_exam["score"].apply(
        lambda x: float(x.split("/")[1])
    )
    medical_exam["scores"] = medical_exam["score"].apply(
        lambda x: x.split("/")[0]
    )
    medical_exam["score_achieved"] = medical_exam["scores"].apply(
        lambda x: (
            np.mean([float(score) for score in x.split(";")])
            if ";" in x
            else float(x)
        )
    )
    medical_exam["accuracy"] = (
        medical_exam["score_achieved"] / medical_exam["score_possible"]
    )
    medical_exam["score_sd"] = medical_exam["scores"].apply(
        lambda x: (
            np.std([float(score) for score in x.split(";")], ddof=1)
            if ";" in x
            else 0
        )
    )
    medical_exam["task"] = medical_exam["subtask"].apply(
        lambda x: x.split(":")[0]
    )
    medical_exam["domain"] = medical_exam["subtask"].apply(
        lambda x: x.split(":")[1]
    )
    medical_exam["language"] = medical_exam["subtask"].apply(
        lambda x: x.split(":")[2]
    )

    # processing: remove "short_words" task, not informative
    medical_exam = medical_exam[medical_exam["task"] != "short_words"]

    # plot language comparison
    aggregated_scores_language = medical_exam.groupby(
        ["model_name", "language"]
    ).agg(
        {
            "accuracy": "mean",
            "score_sd": "mean",
        }
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="language",
        y="accuracy",
        data=aggregated_scores_language,
    )

    plt.savefig(
        "docs/images/boxplot-medical-exam-language.png",
        bbox_inches="tight",
        dpi=300,
    )

    # plot language comparison per domain
    aggregated_scores_language_domain = medical_exam.groupby(
        ["model_name", "language", "domain"]
    ).agg(
        {
            "accuracy": "mean",
            "score_sd": "mean",
        }
    )
    # calculate mean accuracy per language and domain
    mean_accuracy = aggregated_scores_language_domain.groupby(
        ["language", "domain"]
    )["accuracy"].mean()
    # sort domains by mean accuracy
    sorted_domains = mean_accuracy.sort_values(
        ascending=False
    ).index.get_level_values("domain")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.xticks(rotation=45, ha="right")
    sns.boxplot(
        x="domain",
        y="accuracy",
        hue="language",
        data=aggregated_scores_language_domain,
        order=sorted_domains,
    )

    plt.savefig(
        "docs/images/boxplot-medical-exam-language-domain.png",
        bbox_inches="tight",
        dpi=300,
    )

    # plot task comparison
    aggregated_scores_task = medical_exam.groupby(["model_name", "task"]).agg(
        {
            "accuracy": "mean",
            "score_sd": "mean",
        }
    )
    # calculate mean accuracy per task
    mean_accuracy = aggregated_scores_task.groupby("task")["accuracy"].mean()
    # sort tasks by mean accuracy
    sorted_tasks = mean_accuracy.sort_values(ascending=False).index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.xticks(rotation=45, ha="right")
    sns.boxplot(
        x="task",
        y="accuracy",
        data=aggregated_scores_task,
        order=sorted_tasks,
    )

    plt.savefig(
        "docs/images/boxplot-medical-exam-task.png",
        bbox_inches="tight",
        dpi=300,
    )

    # plot domain comparison
    aggregated_scores_domain = medical_exam.groupby(
        ["model_name", "domain"]
    ).agg(
        {
            "accuracy": "mean",
            "score_sd": "mean",
        }
    )
    # calculate mean accuracy per domain
    mean_accuracy = aggregated_scores_domain.groupby("domain")[
        "accuracy"
    ].mean()
    # sort domains by mean accuracy
    sorted_domains = mean_accuracy.sort_values(ascending=False).index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.xticks(rotation=45, ha="right")
    sns.boxplot(
        x="domain",
        y="accuracy",
        data=aggregated_scores_domain,
        order=sorted_domains,
    )

    plt.savefig(
        "docs/images/boxplot-medical-exam-domain.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_comparison_naive_biochatter(overview):
    overview_melted = melt_and_process(overview)

    # select tasks naive_query_generation_using_schema and query_generation
    overview_melted = overview_melted[
        overview_melted["Task"].isin(
            ["naive_query_generation_using_schema", "query_generation"]
        )
    ]

    # print number of rows of each task
    print(overview_melted["Task"].value_counts())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="Task",
        y="Accuracy",
        hue="Task",
        data=overview_melted,
    )
    plt.ylim(0, 1)
    plt.xlabel(None)
    plt.xticks(
        ticks=range(len(overview_melted["Task"].unique())),
        labels=["BioChatter", "Naive LLM (using full YAML schema)"],
    )
    plt.savefig(
        "docs/images/boxplot-naive-vs-biochatter.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/boxplot-naive-vs-biochatter.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # plot scatter plot
    plt.figure(figsize=(6, 4))
    sns.stripplot(
        x="Task",
        y="Accuracy",
        data=overview_melted,
        jitter=0.2,
        alpha=0.8,
    )
    plt.ylim(0, 1)
    plt.xlabel(None)
    plt.xticks(
        ticks=range(len(overview_melted["Task"].unique())),
        labels=["BioChatter", "Naive LLM (using full YAML schema)"],
    )
    plt.savefig(
        "docs/images/scatter-naive-vs-biochatter.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/scatter-naive-vs-biochatter.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # plit violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(
        x="Task",
        y="Accuracy",
        data=overview_melted,
    )
    plt.ylim(0, 1)
    plt.xlabel(None)
    plt.xticks(
        ticks=range(len(overview_melted["Task"].unique())),
        labels=["BioChatter", "Naive LLM (using full YAML schema)"],
    )
    plt.savefig(
        "docs/images/violin-naive-vs-biochatter.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/violin-naive-vs-biochatter.pdf",
        bbox_inches="tight",
    )
    plt.close()


def calculate_stats(overview):
    overview_melted = melt_and_process(overview)
    # calculate p-value between naive and biochatter
    from scipy.stats import ttest_ind

    biochatter = overview_melted[overview_melted["Task"] == "query_generation"][
        "Accuracy"
    ].dropna()
    naive = overview_melted[
        overview_melted["Task"] == "naive_query_generation_using_schema"
    ]["Accuracy"].dropna()

    t_stat, p_value = ttest_ind(biochatter, naive)

    # write to file
    with open("benchmark/results/processed/naive_vs_biochatter.txt", "w") as f:
        f.write(f"mean: {biochatter.mean()} vs {naive.mean()}\n")
        f.write(f"std: {biochatter.std()} vs {naive.std()}\n")
        f.write(f"p-value: {p_value}\n")
        f.write(f"t-statistic: {t_stat}\n")

    # TODO publish this test and other related ones on website as well?

    # calculate correlation between LLM size and accuracy for all tasks
    # convert size to float, make Unknown = 300, replace commas with dots
    size = overview_melted["Size"].apply(
        lambda x: 300 if x == "Unknown" else float(x.replace(",", "."))
    )
    from scipy.stats import pearsonr

    # Create a mask of rows where 'Accuracy' is not NaN
    mask = overview_melted["Accuracy"].notna()

    # Use the mask to select only the rows from 'size' and 'Accuracy' where 'Accuracy' is not NaN
    size = size[mask]
    accuracy = overview_melted["Accuracy"][mask]

    size_accuracy_corr, size_accuracy_p_value = pearsonr(size, accuracy)

    # plot scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=size, y=overview_melted["Accuracy"])
    plt.xlabel("Model size (billions of parameters)")
    plt.ylabel("Accuracy")
    plt.title("Scatter plot of model size vs accuracy")
    plt.savefig(
        "docs/images/scatter-size-accuracy.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/scatter-size-accuracy.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # calculate correlation between quantisation and accuracy for all tasks
    # convert quantisation to float, make >= 16-bit* = 16, replace -bit with nothing
    quantisation = overview_melted["Quantisation"].apply(
        lambda x: 16 if x == ">= 16-bit*" else float(x.replace("-bit", ""))
    )
    # Create a mask of rows where 'Accuracy' is not NaN
    mask = overview_melted["Accuracy"].notna()

    # Use the mask to select only the rows from 'quantisation' and 'Accuracy' where 'Accuracy' is not NaN
    quantisation = quantisation[mask]
    accuracy = overview_melted["Accuracy"][mask]

    quant_accuracy_corr, quant_accuracy_p_value = pearsonr(
        quantisation, accuracy
    )
    # plot scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=quantisation, y=overview_melted["Accuracy"])
    plt.xlabel("Quantisation (bits)")
    plt.ylabel("Accuracy")
    plt.title("Scatter plot of quantisation vs accuracy")
    plt.savefig(
        "docs/images/scatter-quantisation-accuracy.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "docs/images/scatter-quantisation-accuracy.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # write to file
    with open("benchmark/results/processed/correlations.txt", "w") as f:
        f.write(f"Size vs accuracy Pearson correlation: {size_accuracy_corr}\n")
        f.write(
            f"Size vs accuracy Pearson correlation p-value: {size_accuracy_p_value}\n"
        )
        f.write(
            f"Quantisation vs accuracy Pearson correlation: {quant_accuracy_corr}\n"
        )
        f.write(
            f"Quantisation vs accuracy Pearson correlation p-value: {quant_accuracy_p_value}\n"
        )


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
            if "gpt-3.5-turbo" in row["Model name"]
            or "gpt-4" in row["Model name"]
            or "claude" in row["Model name"]
            else row["Quantisation"]
        ),
        axis=1,
    )

    return overview_melted


if __name__ == "__main__":
    on_pre_build(None)
