"""Reproduce Table 5a from released derived scores without API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PAIR = ["case_id", "patient_attitude", "response_index"]
EXPECTED_PAIRS = 320
EXPECTED_CASES = 20


def analyze(frame: pd.DataFrame, resamples: int = 50000) -> pd.DataFrame:
    keys = ["model_label", "system_prompt", *PAIR]
    if frame.duplicated(keys).any():
        message = "Duplicate response identifiers"
        raise ValueError(message)
    rows = []
    for outcome in ["structured_score", "communication_score"]:
        family = []
        for model, group in frame.groupby("model_label", sort=True):
            matrix = group[group.system_prompt.isin(["none", "minimal"])].pivot(  # noqa: PD010
                index=PAIR, columns="system_prompt", values=outcome
            )
            if len(matrix) != EXPECTED_PAIRS or matrix[["none", "minimal"]].isna().any().any():
                message = f"Incomplete matched data for {model}"
                raise ValueError(message)
            delta = matrix["minimal"] - matrix["none"]
            cases = delta.groupby(level="case_id").mean()
            if len(cases) != EXPECTED_CASES:
                message = "Expected 20 medication-indication cases"
                raise ValueError(message)
            rng = np.random.default_rng(20260920)
            draws = rng.choice(cases.to_numpy(), size=(resamples, len(cases)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            # Preserve ties rather than ranking floating-point round-off.
            differences = np.round(cases.to_numpy(), 12)
            p = 1.0 if not np.any(differences) else float(stats.wilcoxon(differences, method="auto").pvalue)
            family.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "n_pairs": len(matrix),
                    "n_cases": len(cases),
                    "none_mean": matrix["none"].mean(),
                    "minimal_mean": matrix["minimal"].mean(),
                    "difference": delta.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": p,
                }
            )
        corrected = multipletests([row["p_value"] for row in family], method="holm")[1]
        for row, adjusted in zip(family, corrected, strict=True):
            row["p_holm"] = adjusted
        rows.extend(family)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/final_analysis/final_response_level_scores.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/none_minimal/supplementary_table_5a.csv")
    args = parser.parse_args()
    result = analyze(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} contrasts to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
