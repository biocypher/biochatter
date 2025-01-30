import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_api_calling():
    """Analyze and plot results from the API calling benchmarks."""
    # Read and process Python API calling results
    python_results = preprocess_results("benchmark/results/python_api_calling.csv", "Python API")
    python_reduced_results = preprocess_results(
        "benchmark/results/python_api_calling_reduced.csv", "Python API (Reduced)"
    )

    # Combine Python results
    all_results = pd.concat([python_results, python_reduced_results])

    # Plot results
    plot_model_family_comparison(all_results)
    plot_benchmark_comparison(all_results)
    plot_task_type_comparison(all_results)
    save_summary_statistics(all_results)


def preprocess_results(filepath: str, benchmark_type: str) -> pd.DataFrame:
    """Preprocess results from a single benchmark file."""
    results = pd.read_csv(filepath)

    # Calculate accuracy
    results["score_possible"] = results["score"].apply(lambda x: float(x.split("/")[1]))
    results["scores"] = results["score"].apply(lambda x: x.split("/")[0])
    results["score_achieved"] = results["scores"].apply(
        lambda x: (np.mean([float(score) for score in x.split(";")]) if ";" in x else float(x))
    )
    results["accuracy"] = results["score_achieved"] / results["score_possible"]

    # Add benchmark type
    results["benchmark_type"] = benchmark_type

    # Add task type analysis and merge similar categories
    results["task_type"] = results["subtask"].apply(lambda x: x.split(" - ")[0].split(":")[-1])

    # Merge similar categories
    results["task_type"] = results["task_type"].replace(
        {
            "abbreviations": "abbreviation",
            "general_question": "general",
            "explicit_variable_names": "specific",
        }
    )

    return results


def plot_model_family_comparison(results: pd.DataFrame) -> None:
    """Create violin plot comparing individual models across all benchmarks."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # # Define model order (from oldest to newest)
    # model_order = [
    #     "gpt-3.5-turbo-0125",
    #     "gpt-4-0613",
    #     "gpt-4-0125-preview",
    #     "gpt-4-1106-preview",
    #     "gpt-4-turbo-2024-04-09",
    #     "gpt-4o-2024-08-06",
    #     "gpt-4o-2023-11-20",
    #     "gpt-4o-mini-2024-07-18"
    # ]
    # # Ensure model order has no duplicates and only includes models present in the data
    # model_order = [m for m in model_order if m in results['model_name'].unique()]

    sns.violinplot(
        x="model_name",
        y="accuracy",
        hue="benchmark_type",
        data=results,
        cut=0,  # Don't extend beyond observed data
        # order=model_order
    )

    plt.title("API Calling Performance by Model and Benchmark Type")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.savefig("docs/images/api-calling-model-comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_benchmark_comparison(results: pd.DataFrame) -> None:
    """Create violin plot comparing benchmark types."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))

    sns.violinplot(
        x="benchmark_type",
        y="accuracy",
        data=results,
        hue="benchmark_type",
        cut=0,  # Don't extend beyond observed data
    )

    plt.title("Performance Comparison Across API Calling Benchmarks")
    plt.xticks(rotation=45)

    plt.savefig("docs/images/api-calling-benchmark-comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_task_type_comparison(results: pd.DataFrame) -> None:
    """Create violin plot comparing performance across different task types."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    sns.violinplot(
        x="task_type",
        y="accuracy",
        hue="benchmark_type",
        data=results,
        cut=0,  # Don't extend beyond observed data
    )

    plt.title("API Calling Performance by Task Type")
    plt.xlabel("Task Type")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.savefig("docs/images/api-calling-task-type-comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def save_summary_statistics(results: pd.DataFrame) -> None:
    """Calculate and save summary statistics."""
    # Summary by model and benchmark type
    summary = (
        results.groupby(["benchmark_type", "model_name"])["accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )

    summary.to_csv("benchmark/results/processed/api_calling_summary.csv")

    # Overall benchmark comparison
    benchmark_summary = (
        results.groupby("benchmark_type")["accuracy"].agg(["mean", "std", "min", "max", "count"]).round(3)
    )

    benchmark_summary.to_csv("benchmark/results/processed/api_calling_benchmark_summary.csv")

    # Add task type summary
    task_summary = (
        results.groupby(["benchmark_type", "task_type"])["accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )

    task_summary.to_csv("benchmark/results/processed/api_calling_task_summary.csv")


if __name__ == "__main__":
    analyze_api_calling()
