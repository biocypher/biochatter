from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = (
    ROOT / "results" / "final_analysis" / "final_response_metric_scores.csv"
)
MODEL_ORDER = [
    "Llama 3.1 8B Instruct",
    "Med42-8B",
    "Llama 4 Maverick",
    "GLM 4.5",
    "GPT-OSS-120B",
    "Claude Sonnet 4.6",
    "Gemini 3.5 Flash",
    "GPT-5.4",
]
STRUCTURED_METRICS = [
    "adverse_effects_recall",
    "very_common_adverse_effects_coverage",
    "common_adverse_effects_coverage",
    "uncommon_adverse_effects_coverage",
    "adverse_effects_specificity",
    "contraindications_recall",
    "contraindications_specificity",
]
COMMUNICATION_METRICS = [
    "understandability",
    "usefulness",
    "patient_attitude_responsiveness",
]
WITHIN_MODEL_AXES = {
    "system_prompt": {
        "column": "system_prompt",
        "levels": [
            "none",
            "minimal",
            "role_encouraging",
            "role_attitude_sensitive",
        ],
        "paired_on": ["case_id", "patient_attitude", "response_index"],
    },
    "patient_attitude": {
        "column": "patient_attitude",
        "levels": ["very_anxious", "anxious", "neutral", "confident"],
        "paired_on": ["case_id", "system_prompt", "response_index"],
    },
}


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    order = valid_positions[np.argsort(values[valid_positions], kind="stable")]
    running_max = 0.0
    total = len(order)
    for rank, position in enumerate(order):
        candidate = min(1.0, (total - rank) * values[position])
        running_max = max(running_max, candidate)
        adjusted[position] = running_max
    return adjusted


def wilcoxon_large_sample(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> tuple[float, float]:
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    differences = differences[~np.isclose(differences, 0.0)]
    if not len(differences):
        return 0.0, 1.0

    absolute = np.abs(differences)
    ranks = pd.Series(absolute).rank(method="average").to_numpy(float)
    rank_positive = float(ranks[differences > 0].sum())
    rank_negative = float(ranks[differences < 0].sum())
    statistic = min(rank_positive, rank_negative)
    count = len(differences)
    mean = count * (count + 1) / 4
    _, tie_counts = np.unique(absolute, return_counts=True)
    tie_correction = float(np.sum(tie_counts**3 - tie_counts))
    variance = (
        count * (count + 1) * (2 * count + 1) - tie_correction / 2
    ) / 24
    if variance <= 0:
        return statistic, 1.0
    z_value = (rank_positive - mean) / math.sqrt(variance)
    return statistic, float(min(1.0, math.erfc(abs(z_value) / math.sqrt(2))))


def exact_mcnemar_p(positive_negative: int, negative_positive: int) -> float:
    discordant = positive_negative + negative_positive
    if discordant == 0:
        return 1.0
    lower = min(positive_negative, negative_positive)
    lower_tail = sum(math.comb(discordant, successes) for successes in range(lower + 1))
    return min(1.0, float(Fraction(2 * lower_tail, 2**discordant)))


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Return the chi-square survival probability without a SciPy dependency."""
    if statistic <= 0:
        return 1.0
    if degrees_of_freedom != 3:
        raise ValueError("This analysis uses four-condition omnibus tests only.")

    x = statistic / 2.0
    # Q(3/2, x) = erfc(sqrt(x)) + 2 sqrt(x / pi) exp(-x).
    return float(math.erfc(math.sqrt(x)) + 2 * math.sqrt(x / math.pi) * math.exp(-x))


def average_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    _, tie_counts = np.unique(values, return_counts=True)
    return ranks, tie_counts[tie_counts > 1]


def friedman_test(matrix: np.ndarray) -> tuple[float, int, float]:
    """Compute a tie-corrected Friedman omnibus test for complete paired rows."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Friedman analysis requires a complete four-condition matrix.")
    if not len(values):
        return float("nan"), 3, float("nan")
    if not np.isfinite(values).all():
        raise ValueError("Friedman analysis requires finite complete-pair scores.")

    n_units, n_levels = values.shape
    rank_sums = np.zeros(n_levels, dtype=float)
    tie_correction_numerator = 0.0
    for row in values:
        ranks, tie_counts = average_ranks(row)
        rank_sums += ranks
        tie_correction_numerator += float(np.sum(tie_counts**3 - tie_counts))

    statistic = (
        12.0 * np.sum(rank_sums**2) / (n_units * n_levels * (n_levels + 1))
        - 3.0 * n_units * (n_levels + 1)
    )
    correction = 1.0 - tie_correction_numerator / (
        n_units * n_levels * (n_levels**2 - 1)
    )
    if correction <= 0:
        return float("nan"), n_levels - 1, float("nan")
    statistic /= correction
    return float(statistic), n_levels - 1, chi_square_survival(statistic, n_levels - 1)


def strata() -> list[tuple[str, str, dict[str, str]]]:
    definitions: list[tuple[str, str, dict[str, str]]] = [("overall", "all", {})]
    definitions.extend(
        ("system_prompt", value, {"system_prompt": value})
        for value in ["none", "minimal", "role_encouraging", "role_attitude_sensitive"]
    )
    definitions.extend(
        ("patient_attitude", value, {"patient_attitude": value})
        for value in ["very_anxious", "anxious", "neutral", "confident"]
    )
    return definitions


def pair_frame(
    frame: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    keys = ["case_id", "system_prompt", "patient_attitude", "response_index"]
    subset = frame[
        frame["metric"].eq(metric)
        & frame["model_label"].isin([model_a, model_b])
    ]
    wide = subset.pivot(index=keys, columns="model_label", values="score")
    if model_a not in wide or model_b not in wide:
        return pd.DataFrame(columns=[model_a, model_b])
    return wide[[model_a, model_b]].dropna()


def analyze_structured(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for stratum_type, stratum, filters in strata():
        subset = scores
        for column, value in filters.items():
            subset = subset[subset[column].eq(value)]
        for metric in STRUCTURED_METRICS:
            for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
                paired = pair_frame(subset, metric, model_a, model_b)
                values_a = paired[model_a].to_numpy(float)
                values_b = paired[model_b].to_numpy(float)
                statistic, p_value = wilcoxon_large_sample(values_a, values_b)
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": len(paired),
                        "mean_a": float(values_a.mean()),
                        "mean_b": float(values_b.mean()),
                        "mean_difference_a_minus_b": float(
                            (values_a - values_b).mean()
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"])
    return result


def analyze_communication(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for stratum_type, stratum, filters in strata():
        subset = scores
        for column, value in filters.items():
            subset = subset[subset[column].eq(value)]
        for metric in COMMUNICATION_METRICS:
            for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
                paired = pair_frame(subset, metric, model_a, model_b)
                strict_a = paired[model_a].eq(1.0).to_numpy(int)
                strict_b = paired[model_b].eq(1.0).to_numpy(int)
                a_positive_b_negative = int(np.sum((strict_a == 1) & (strict_b == 0)))
                a_negative_b_positive = int(np.sum((strict_a == 0) & (strict_b == 1)))
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": len(paired),
                        "strict_positive_rate_a": float(strict_a.mean()),
                        "strict_positive_rate_b": float(strict_b.mean()),
                        "a_positive_b_negative": a_positive_b_negative,
                        "a_negative_b_positive": a_negative_b_positive,
                        "discordant_pairs": (
                            a_positive_b_negative + a_negative_b_positive
                        ),
                        "p_raw": exact_mcnemar_p(
                            a_positive_b_negative,
                            a_negative_b_positive,
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"])
    return result


def within_model_matrix(
    scores: pd.DataFrame,
    *,
    metric: str,
    model: str,
    axis: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    definition = WITHIN_MODEL_AXES[axis]
    axis_column = str(definition["column"])
    levels = list(definition["levels"])
    paired_on = list(definition["paired_on"])
    subset = scores[
        scores["model_label"].eq(model) & scores["metric"].eq(metric)
    ]
    wide = subset.pivot(index=paired_on, columns=axis_column, values="score")
    wide = wide.reindex(columns=levels).dropna()
    return wide, {
        "paired_on": ";".join(paired_on),
        "levels": ";".join(levels),
        "n_levels": len(levels),
    }


def analyze_structured_within(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for axis in WITHIN_MODEL_AXES:
        for metric in STRUCTURED_METRICS:
            for model in MODEL_ORDER:
                paired, details = within_model_matrix(
                    scores, metric=metric, model=model, axis=axis
                )
                statistic, degrees_of_freedom, p_value = friedman_test(
                    paired.to_numpy(float)
                )
                rows.append(
                    {
                        "comparison_axis": axis,
                        "paired_on": details["paired_on"],
                        "evaluated_model": model,
                        "metric": metric,
                        "n_levels": details["n_levels"],
                        "levels": details["levels"],
                        "n_units": len(paired),
                        "friedman_statistic": statistic,
                        "degrees_of_freedom": degrees_of_freedom,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust(result["p_value"])
    return result


def cochran_q_test(matrix: np.ndarray) -> tuple[float, int, float]:
    values = np.asarray(matrix, dtype=int)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Cochran's Q analysis requires a complete four-condition matrix.")
    n_units, n_levels = values.shape
    column_sums = values.sum(axis=0)
    row_sums = values.sum(axis=1)
    numerator = (n_levels - 1) * (
        n_levels * np.sum(column_sums**2) - np.sum(column_sums) ** 2
    )
    denominator = n_levels * np.sum(row_sums) - np.sum(row_sums**2)
    if denominator == 0:
        return float("nan"), n_levels - 1, float("nan")
    statistic = float(numerator / denominator)
    return statistic, n_levels - 1, chi_square_survival(statistic, n_levels - 1)


def analyze_communication_within(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for axis in WITHIN_MODEL_AXES:
        for metric in COMMUNICATION_METRICS:
            for model in MODEL_ORDER:
                paired, details = within_model_matrix(
                    scores, metric=metric, model=model, axis=axis
                )
                strict = paired.eq(1.0).to_numpy(int)
                statistic, degrees_of_freedom, p_value = cochran_q_test(strict)
                rows.append(
                    {
                        "comparison_axis": axis,
                        "paired_on": details["paired_on"],
                        "evaluated_model": model,
                        "metric": metric,
                        "n_levels": details["n_levels"],
                        "levels": details["levels"],
                        "n_units": len(paired),
                        "q_statistic": statistic,
                        "degrees_of_freedom": degrees_of_freedom,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust(result["p_value"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce final inferential tests.")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    structured = analyze_structured(scores[scores["metric_group"].eq("structured")])
    communication = analyze_communication(scores[scores["metric_group"].eq("judge")])
    structured_within = analyze_structured_within(
        scores[scores["metric_group"].eq("structured")]
    )
    communication_within = analyze_communication_within(
        scores[scores["metric_group"].eq("judge")]
    )
    structured_path = (
        args.output_root
        / "structured_metrics"
        / "final_structured_pairwise_wilcoxon.csv"
    )
    communication_path = (
        args.output_root
        / "llm_judge_metrics"
        / "final_communication_pairwise_mcnemar.csv"
    )
    structured_within_path = (
        args.output_root
        / "structured_metrics"
        / "final_structured_within_friedman.csv"
    )
    communication_within_path = (
        args.output_root
        / "llm_judge_metrics"
        / "final_communication_within_cochran_q.csv"
    )
    figure_source_path = (
        args.output_root
        / "figure_source_data"
        / "supplementary_figure_s3_within_model_effects_source_data.csv"
    )
    structured_path.parent.mkdir(parents=True, exist_ok=True)
    communication_path.parent.mkdir(parents=True, exist_ok=True)
    structured.to_csv(structured_path, index=False)
    communication.to_csv(communication_path, index=False)
    structured_within.to_csv(structured_within_path, index=False)
    communication_within.to_csv(communication_within_path, index=False)
    figure_source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [structured_within, communication_within], ignore_index=True, sort=False
    ).to_csv(figure_source_path, index=False)

    report = {
        "structured_tests": len(structured),
        "communication_tests": len(communication),
        "structured_within_tests": len(structured_within),
        "communication_within_tests": len(communication_within),
        "structured_holm_family": (
            "All overall and condition-stratified between-model Wilcoxon tests."
        ),
        "communication_holm_family": (
            "All overall and condition-stratified between-model McNemar tests."
        ),
        "strict_communication_rule": "Both judge iterations must be positive.",
    }
    report_path = args.output_root / "final_analysis" / "pairwise_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
