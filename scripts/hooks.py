"""MkDocs hooks for preprocessing data and generating plots.

Preprocessing and plotting scripts are run during documentation build.
"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

# Get the current directory
CURRENT_DIR = Path(__file__).parent

# Dynamically import local modules with full paths
def import_local_module(module_name: str) -> None:
    """Import a local module."""
    module_path = CURRENT_DIR / f"_{module_name}.py"
    module = None

    # If module is already loaded, return it
    if f"_{module_name}" in sys.modules:
        return sys.modules[f"_{module_name}"]

    spec = spec_from_file_location(f"_{module_name}", module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module {module_name}"
        raise ImportError(msg)
    module = module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module

# Import the modules
plotting = import_local_module("plotting")
preprocess = import_local_module("preprocess")
stats = import_local_module("stats")

# Extract the needed functions
plot_accuracy_per_model = plotting.plot_accuracy_per_model
plot_accuracy_per_quantisation = plotting.plot_accuracy_per_quantisation
plot_accuracy_per_task = plotting.plot_accuracy_per_task
plot_comparison_naive_biochatter = plotting.plot_comparison_naive_biochatter
plot_extraction_tasks = plotting.plot_extraction_tasks
plot_image_caption_confidence = plotting.plot_image_caption_confidence
plot_medical_exam = plotting.plot_medical_exam
plot_rag_tasks = plotting.plot_rag_tasks
plot_scatter_per_quantisation = plotting.plot_scatter_per_quantisation
plot_task_comparison = plotting.plot_task_comparison
plot_text2cypher = plotting.plot_text2cypher
plot_text2cypher_safety_only = plotting.plot_text2cypher_safety_only

create_overview_table = preprocess.create_overview_table
preprocess_results_for_frontend = preprocess.preprocess_results_for_frontend

calculate_stats = stats.calculate_stats

# Add both the scripts directory and project root to Python path
current_dir = str(Path(__file__).parent)
project_root = str(Path(__file__).parent.parent.parent)

for path in [current_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

def on_pre_build() -> None:
    """Run pre-processing and plotting scripts.

    This function is no longer called when building the documentation. The hooks
    are run by a GitHub action (see .github/workflows/benchmark-processing.yml).
    For local development, the hooks can be run by executing this file directly.
    """
    result_files_path = "benchmark/results/"

    result_file_names = [
        f
        for f in os.listdir(result_files_path)
        if (Path(result_files_path) / f).is_file()
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
    on_pre_build()
