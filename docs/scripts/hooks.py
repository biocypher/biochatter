import os
import re
import numpy as np
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
        and f.endswith(".csv")
    ]

    for file_name in result_file_names:
        results = pd.read_csv(f"{result_files_path}{file_name}")
        preprocess_results_for_frontend(results, result_files_path, file_name)

    overview = create_overview_table(result_files_path, result_file_names)

    plot_accuracy_per_model(overview)
    plot_accuracy_per_quantisation(overview)
    plot_accuracy_per_task(overview)
    plot_scatter_per_quantisation(overview)
    plot_task_comparison(overview)
    plot_rag_tasks(overview)
    plot_comparison_naive_biochatter(overview)
    calculate_stats(overview)


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
        f"{path}processed/{file_name}",
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
            "Unknown" if "gpt-4" in row["Model name"] else row["Size"]
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
    plt.xticks(rotation=45)
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
    plt.figure(figsize=(6, 4))
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
            "7",
            "6",
        ],
        ordered=True,
    )

    # Add jitter to x-coordinates
    x = pd.Categorical(overview_melted["Quantisation"]).codes.astype(float)

    # Create a mask for 'openhermes' and 'gpt' models
    mask_openhermes = overview_melted["Model name"] == "openhermes-2.5"
    mask_gpt = overview_melted["Model name"].str.contains("gpt")

    # Do not add jitter for 'openhermes' model
    x[mask_openhermes] += 0

    # Manually enter jitter values for 'gpt' models
    jitter_values = {
        "gpt-3": -0.2,
        "gpt-4": 0.2,
    }

    for model, jitter in jitter_values.items():
        mask_model = overview_melted["Model name"].str.contains(model)
        x[mask_model] += jitter

    # For other models, add the original jitter
    x[~mask_openhermes & ~mask_gpt] += np.random.normal(
        0, 0.1, size=len(x[~mask_openhermes & ~mask_gpt])
    )

    # Create a ColorBrewer palette
    palette = sns.color_palette("Paired", n_colors=12)

    # Define a dictionary mapping model names to colors
    color_dict = {
        "gpt-3.5-turbo-0613": palette[0],
        "gpt-3.5-turbo-0125": palette[1],
        "gpt-4-0613": palette[2],
        "gpt-4-0125-preview": palette[3],
        "openhermes-2.5": palette[5],
        "code-llama-instruct": palette[6],
        "llama-2-chat": palette[7],
        "mixtral-instruct-v0.1": palette[8],
        "mistral-instruct-v0.2": palette[9],
        "chatglm3": palette[11],
    }

    # Use the dictionary as the palette argument in sns.scatterplot
    ax = sns.scatterplot(
        x=x,
        y="Mean Accuracy",
        hue="Model name",
        size="Size",
        sizes=(10, 300),
        data=overview_melted,
        palette=color_dict,  # Use the color dictionary here
        alpha=0.5,
    )

    model_names_order = [
        "Model name",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0613",
        "openhermes-2.5",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "llama-2-chat",
        "code-llama-instruct",
        "mixtral-instruct-v0.1",
        "mistral-instruct-v0.2",
        "chatglm3",
        "Size",
        "Unknown",
        "175",
        "70",
        "46,7",
        "34",
        "13",
        "7",
        "6",
    ]

    # Reorder the legend
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in model_names_order]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
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
    plt.xticks(rotation=45)
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


def plot_comparison_naive_biochatter(overview):
    overview_melted = melt_and_process(overview)

    # select tasks naive_query_generation_using_schema and query_generation
    overview_melted = overview_melted[
        overview_melted["Task"].isin(
            ["naive_query_generation_using_schema", "query_generation"]
        )
    ]

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
    size_accuracy_corr = size.corr(overview_melted["Accuracy"])
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
    quant_accuracy_corr = quantisation.corr(overview_melted["Accuracy"])
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
            f"Quantisation vs accuracy Pearson correlation: {quant_accuracy_corr}\n"
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
            else row["Quantisation"]
        ),
        axis=1,
    )

    return overview_melted


if __name__ == "__main__":
    on_pre_build(None)
