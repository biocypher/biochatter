"""Statistical analysis of the results."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import plotting module using full path
CURRENT_DIR = Path(__file__).parent
plotting_path = CURRENT_DIR / "_plotting.py"

if "_plotting" not in sys.modules:
    spec = spec_from_file_location("_plotting", plotting_path)
    if spec is None or spec.loader is None:
        msg = "Could not load _plotting module"
        raise ImportError(msg)
    plotting = module_from_spec(spec)
    sys.modules["_plotting"] = plotting
    spec.loader.exec_module(plotting)
else:
    plotting = sys.modules["_plotting"]

melt_and_process = plotting.melt_and_process


def calculate_stats(overview: pd.DataFrame) -> None:
    overview_melted = melt_and_process(overview)
    # calculate p-value between naive and biochatter
    from scipy.stats import ttest_ind

    biochatter = overview_melted[overview_melted["Task"] == "query_generation"]["Accuracy"].dropna()
    naive = overview_melted[overview_melted["Task"] == "naive_query_generation_using_schema"]["Accuracy"].dropna()

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
        lambda x: 300 if x == "Unknown" else float(x.replace(",", ".")),
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
        lambda x: 16 if x == ">= 16-bit*" else float(x.replace("-bit", "")),
    )
    # Create a mask of rows where 'Accuracy' is not NaN
    mask = overview_melted["Accuracy"].notna()

    # Use the mask to select only the rows from 'quantisation' and 'Accuracy' where 'Accuracy' is not NaN
    quantisation = quantisation[mask]
    accuracy = overview_melted["Accuracy"][mask]

    quant_accuracy_corr, quant_accuracy_p_value = pearsonr(
        quantisation,
        accuracy,
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
            f"Size vs accuracy Pearson correlation p-value: {size_accuracy_p_value}\n",
        )
        f.write(
            f"Quantisation vs accuracy Pearson correlation: {quant_accuracy_corr}\n",
        )
        f.write(
            f"Quantisation vs accuracy Pearson correlation p-value: {quant_accuracy_p_value}\n",
        )
