#!/usr/bin/env python3
"""Visualize MCP vs Baseline performance comparison for EDAM QA benchmark.

This script creates visualizations comparing the performance of models with
MCP tools enabled versus baseline (no tools) performance.

Usage:
    python benchmark/scripts/visualize_mcp_baseline_comparison.py
    python benchmark/scripts/visualize_mcp_baseline_comparison.py --output-dir benchmark/images
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_score(score_str: str) -> tuple[float, float]:
    """Parse score string in format 'X/Y' to (score, max_score).

    Args:
    ----
        score_str: Score string like "1/1", "2/3", etc.

    Returns:
    -------
        Tuple of (score, max_score) as floats

    """
    if pd.isna(score_str) or score_str == "":
        return (0.0, 1.0)
    parts = str(score_str).split("/")
    if len(parts) != 2:
        return (0.0, 1.0)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (0.0, 1.0)


def calculate_accuracy(score_str: str) -> float:
    """Calculate accuracy from score string.

    Args:
    ----
        score_str: Score string like "1/1", "2/3", etc.

    Returns:
    -------
        Accuracy as float between 0 and 1

    """
    score, max_score = parse_score(score_str)
    if max_score == 0:
        return 0.0
    return score / max_score


def load_and_prepare_data(mcp_file: str, baseline_file: str) -> pd.DataFrame:
    """Load and prepare data for comparison.

    Args:
    ----
        mcp_file: Path to MCP results CSV
        baseline_file: Path to baseline results CSV

    Returns:
    -------
        DataFrame with merged and prepared data

    """
    # Load data
    mcp_df = pd.read_csv(mcp_file)
    baseline_df = pd.read_csv(baseline_file)

    # Add accuracy column
    mcp_df["accuracy"] = mcp_df["score"].apply(calculate_accuracy)
    baseline_df["accuracy"] = baseline_df["score"].apply(calculate_accuracy)

    # Add method column
    mcp_df["method"] = "MCP"
    baseline_df["method"] = "Baseline"

    # Merge on model_name, subtask, and md5_hash
    merged = pd.merge(
        mcp_df[["model_name", "subtask", "md5_hash", "accuracy", "score"]],
        baseline_df[["model_name", "subtask", "md5_hash", "accuracy", "score"]],
        on=["model_name", "subtask", "md5_hash"],
        suffixes=("_mcp", "_baseline"),
        how="outer",
    )

    # Fill missing values (if one method doesn't have a result)
    merged["accuracy_mcp"] = merged["accuracy_mcp"].fillna(0.0)
    merged["accuracy_baseline"] = merged["accuracy_baseline"].fillna(0.0)

    # Calculate improvement
    merged["improvement"] = merged["accuracy_mcp"] - merged["accuracy_baseline"]
    merged["improvement_pct"] = merged["improvement"] * 100

    return merged


def create_overall_comparison_plot(df: pd.DataFrame, output_path: Path):
    """Create overall performance comparison plot.

    Args:
    ----
        df: Prepared DataFrame
        output_path: Path to save figure

    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy distribution
    ax1 = axes[0]
    mcp_acc = df["accuracy_mcp"].values
    baseline_acc = df["accuracy_baseline"].values

    ax1.hist(baseline_acc, bins=20, alpha=0.6, label="Baseline", color="#e74c3c", edgecolor="black")
    ax1.hist(mcp_acc, bins=20, alpha=0.6, label="MCP", color="#2ecc71", edgecolor="black")
    ax1.set_xlabel("Accuracy", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Accuracy Distribution: MCP vs Baseline", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)

    # Plot 2: Improvement scatter
    ax2 = axes[1]
    ax2.scatter(baseline_acc, mcp_acc, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
    ax2.plot([0, 1], [0, 1], "r--", alpha=0.5, label="No improvement")
    ax2.set_xlabel("Baseline Accuracy", fontsize=12)
    ax2.set_ylabel("MCP Accuracy", fontsize=12)
    ax2.set_title("MCP vs Baseline Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    # Add quadrant labels
    ax2.text(
        0.25,
        0.9,
        "MCP Better",
        fontsize=10,
        color="green",
        alpha=0.7,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax2.text(
        0.7,
        0.1,
        "Baseline Better",
        fontsize=10,
        color="red",
        alpha=0.7,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved overall comparison plot to {output_path}")


def create_subtask_comparison_plot(df: pd.DataFrame, output_path: Path):
    """Create subtask-level comparison plot.

    Args:
    ----
        df: Prepared DataFrame
        output_path: Path to save figure

    """
    # Calculate mean accuracy per subtask
    subtask_stats = (
        df.groupby("subtask")
        .agg({"accuracy_mcp": "mean", "accuracy_baseline": "mean", "improvement": "mean"})
        .reset_index()
    )

    # Sort by improvement
    subtask_stats = subtask_stats.sort_values("improvement", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(subtask_stats) * 0.4)))

    y_pos = range(len(subtask_stats))

    # Create grouped bar chart
    width = 0.35
    ax.barh(
        [y - width / 2 for y in y_pos],
        subtask_stats["accuracy_baseline"],
        width,
        label="Baseline",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
    )
    ax.barh(
        [y + width / 2 for y in y_pos],
        subtask_stats["accuracy_mcp"],
        width,
        label="MCP",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
    )

    # Add improvement annotations
    for i, (idx, row) in enumerate(subtask_stats.iterrows()):
        improvement = row["improvement"]
        if improvement > 0:
            ax.text(
                row["accuracy_mcp"] + 0.02,
                i,
                f"+{improvement:.2f}",
                va="center",
                fontsize=9,
                color="green",
                fontweight="bold",
            )
        elif improvement < 0:
            ax.text(
                row["accuracy_baseline"] + 0.02,
                i,
                f"{improvement:.2f}",
                va="center",
                fontsize=9,
                color="red",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subtask_stats["subtask"], fontsize=10)
    ax.set_xlabel("Mean Accuracy", fontsize=12)
    ax.set_title("Performance by Subtask: MCP vs Baseline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved subtask comparison plot to {output_path}")


def create_model_comparison_plot(df: pd.DataFrame, output_path: Path):
    """Create model-level comparison plot.

    Args:
    ----
        df: Prepared DataFrame
        output_path: Path to save figure

    """
    # Calculate mean accuracy per model
    model_stats = (
        df.groupby("model_name")
        .agg({"accuracy_mcp": "mean", "accuracy_baseline": "mean", "improvement": "mean"})
        .reset_index()
    )

    # Sort by improvement
    model_stats = model_stats.sort_values("improvement", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(model_stats) * 0.5)))

    y_pos = range(len(model_stats))

    # Create grouped bar chart
    width = 0.35
    ax.barh(
        [y - width / 2 for y in y_pos],
        model_stats["accuracy_baseline"],
        width,
        label="Baseline",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
    )
    ax.barh(
        [y + width / 2 for y in y_pos],
        model_stats["accuracy_mcp"],
        width,
        label="MCP",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
    )

    # Add improvement annotations
    for i, (idx, row) in enumerate(model_stats.iterrows()):
        improvement = row["improvement"]
        if improvement > 0:
            ax.text(
                row["accuracy_mcp"] + 0.02,
                i,
                f"+{improvement:.2%}",
                va="center",
                fontsize=10,
                color="green",
                fontweight="bold",
            )
        elif improvement < 0:
            ax.text(
                row["accuracy_baseline"] + 0.02,
                i,
                f"{improvement:.2%}",
                va="center",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_stats["model_name"], fontsize=10)
    ax.set_xlabel("Mean Accuracy", fontsize=12)
    ax.set_title("Performance by Model: MCP vs Baseline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to {output_path}")


def create_improvement_summary_plot(df: pd.DataFrame, output_path: Path):
    """Create improvement summary plot.

    Args:
    ----
        df: Prepared DataFrame
        output_path: Path to save figure

    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Improvement distribution
    ax1 = axes[0]
    improvements = df["improvement"].values
    ax1.hist(improvements, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No improvement")
    ax1.axvline(
        x=improvements.mean(), color="green", linestyle="--", linewidth=2, label=f"Mean: {improvements.mean():.3f}"
    )
    ax1.set_xlabel("Improvement (MCP - Baseline)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Improvement Distribution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement by subtask type
    ax2 = axes[1]

    # Categorize subtasks
    df["subtask_type"] = df["subtask"].apply(
        lambda x: "single"
        if "single_term" in x
        else "two"
        if "two_term" in x
        else "three"
        if "three_term" in x
        else "other"
    )

    type_stats = df.groupby("subtask_type")["improvement"].agg(["mean", "std"]).reset_index()
    type_stats = type_stats.sort_values("mean", ascending=True)

    colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in type_stats["mean"]]
    ax2.barh(
        type_stats["subtask_type"],
        type_stats["mean"],
        xerr=type_stats["std"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        capsize=5,
    )
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Mean Improvement", fontsize=12)
    ax2.set_ylabel("Subtask Type", fontsize=12)
    ax2.set_title("Improvement by Subtask Complexity", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved improvement summary plot to {output_path}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics to console.

    Args:
    ----
        df: Prepared DataFrame

    """
    print("\n" + "=" * 80)
    print("MCP vs Baseline Performance Summary")
    print("=" * 80)

    mcp_mean = df["accuracy_mcp"].mean()
    baseline_mean = df["accuracy_baseline"].mean()
    improvement_mean = df["improvement"].mean()

    print("\nOverall Performance:")
    print(f"  Baseline Mean Accuracy: {baseline_mean:.3f} ({baseline_mean*100:.1f}%)")
    print(f"  MCP Mean Accuracy:      {mcp_mean:.3f} ({mcp_mean*100:.1f}%)")
    print(f"  Mean Improvement:        {improvement_mean:.3f} ({improvement_mean*100:.1f} percentage points)")

    # Count improvements
    improved = (df["improvement"] > 0).sum()
    worsened = (df["improvement"] < 0).sum()
    unchanged = (df["improvement"] == 0).sum()
    total = len(df)

    print("\nImprovement Breakdown:")
    print(f"  Improved:   {improved:3d} ({improved/total*100:.1f}%)")
    print(f"  Worsened:   {worsened:3d} ({worsened/total*100:.1f}%)")
    print(f"  Unchanged:  {unchanged:3d} ({unchanged/total*100:.1f}%)")
    print(f"  Total:      {total:3d}")

    # Best and worst improvements
    best = df.loc[df["improvement"].idxmax()]
    worst = df.loc[df["improvement"].idxmin()]

    print("\nBest Improvement:")
    print(f"  Subtask: {best['subtask']}")
    print(
        f"  Baseline: {best['accuracy_baseline']:.3f} → MCP: {best['accuracy_mcp']:.3f} "
        f"(+{best['improvement']:.3f})"
    )

    if worst["improvement"] < 0:
        print("\nWorst Performance:")
        print(f"  Subtask: {worst['subtask']}")
        print(
            f"  Baseline: {worst['accuracy_baseline']:.3f} → MCP: {worst['accuracy_mcp']:.3f} "
            f"({worst['improvement']:.3f})"
        )

    print("\n" + "=" * 80 + "\n")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(description="Visualize MCP vs Baseline performance comparison")
    parser.add_argument("--mcp-file", default="benchmark/results/mcp_edam_qa.csv", help="Path to MCP results CSV file")
    parser.add_argument(
        "--baseline-file",
        default="benchmark/results/mcp_edam_qa_baseline.csv",
        help="Path to baseline results CSV file",
    )
    parser.add_argument("--output-dir", default="benchmark/results/images", help="Directory to save output figures")

    args = parser.parse_args()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data(args.mcp_file, args.baseline_file)

    if df.empty:
        print("Error: No data found. Please ensure both CSV files exist and contain data.")
        sys.exit(1)

    # Print summary statistics
    print_summary_statistics(df)

    # Create visualizations
    print("Generating visualizations...")
    create_overall_comparison_plot(df, output_dir / "mcp_baseline_overall_comparison.png")
    create_subtask_comparison_plot(df, output_dir / "mcp_baseline_subtask_comparison.png")
    create_model_comparison_plot(df, output_dir / "mcp_baseline_model_comparison.png")
    create_improvement_summary_plot(df, output_dir / "mcp_baseline_improvement_summary.png")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
