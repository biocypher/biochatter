"""MkDocs hooks for preprocessing data and generating plots.

Preprocessing and plotting scripts are run during documentation build.
"""

import os

import pandas as pd

from ._plotting import (
    plot_accuracy_per_model,
    plot_accuracy_per_quantisation,
    plot_accuracy_per_task,
    plot_comparison_naive_biochatter,
    plot_extraction_tasks,
    plot_image_caption_confidence,
    plot_medical_exam,
    plot_rag_tasks,
    plot_scatter_per_quantisation,
    plot_task_comparison,
    plot_text2cypher,
    plot_text2cypher_safety_only,
)
from ._preprocess import create_overview_table, preprocess_results_for_frontend
from ._stats import calculate_stats


def on_pre_build(config, **kwargs) -> None:
    """Run pre-processing and plotting scripts.

    This function is called when building the documentation.
    """
    result_files_path = "benchmark/results/"

    result_file_names = [
        f
        for f in os.listdir(result_files_path)
        if os.path.isfile(os.path.join(result_files_path, f))
        and f.endswith(".csv")
        and "failure_mode" not in f
        and "confidence" not in f
    ]

    for file_name in result_file_names:
        results = pd.read_csv(f"{result_files_path}{file_name}")
        preprocess_results_for_frontend(results, result_files_path, file_name)

    overview = create_overview_table(result_files_path, result_file_names)

    plot_text2cypher()
    plot_text2cypher_safety_only()
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


if __name__ == "__main__":
    on_pre_build(None)
