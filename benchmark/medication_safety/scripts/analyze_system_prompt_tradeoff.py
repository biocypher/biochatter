from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "final_analysis"
    / "summary_model_system_prompt_trajectory.csv"
)
MODEL_ORDER = [
    "Claude Sonnet 4.6",
    "GPT-5.4",
    "GPT-OSS-120B",
    "Med42-8B",
    "Gemini 3.5 Flash",
    "GLM 4.5",
    "Llama 4 Maverick",
    "Llama 3.1 8B Instruct",
]
PROMPT_ORDER = [
    "none",
    "minimal",
    "role_encouraging",
    "role_attitude_sensitive",
]


def load_and_center(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model_label",
        "system_prompt",
        "structured_score",
        "communication_score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["model_label"] = pd.Categorical(
        frame["model_label"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    frame["system_prompt"] = pd.Categorical(
        frame["system_prompt"],
        categories=PROMPT_ORDER,
        ordered=True,
    )
    if frame["model_label"].isna().any() or frame["system_prompt"].isna().any():
        raise ValueError("Input contains an unknown model or system prompt.")

    frame = frame.sort_values(["model_label", "system_prompt"]).reset_index(drop=True)
    if frame.duplicated(["model_label", "system_prompt"]).any():
        raise ValueError("Input contains duplicate model and system prompt rows.")
    frame["structured_centered"] = frame["structured_score"] - frame.groupby(
        "model_label",
        observed=True,
    )["structured_score"].transform("mean")
    frame["communication_centered"] = frame["communication_score"] - frame.groupby(
        "model_label",
        observed=True,
    )["communication_score"].transform("mean")
    return frame


def permutation_p_value(
    frame: pd.DataFrame,
    observed_r: float,
    *,
    permutations: int = 100_000,
    seed: int = 20260712,
) -> float:
    rng = np.random.default_rng(seed)
    structured = frame["structured_centered"].to_numpy(float)
    communication = frame["communication_centered"].to_numpy(float)
    groups = [
        indices.to_numpy()
        for _, indices in frame.groupby("model_label", observed=True).groups.items()
    ]
    exceedances = 0
    for _ in range(permutations):
        permuted = communication.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_r = np.corrcoef(structured, permuted)[0, 1]
        if abs(permuted_r) >= abs(observed_r):
            exceedances += 1
    return (exceedances + 1) / (permutations + 1)


def analyze(
    path: str | Path,
    *,
    permutations: int = 100_000,
    seed: int = 20260712,
) -> dict[str, int | float]:
    frame = load_and_center(path)
    observed_r = float(
        np.corrcoef(
            frame["structured_centered"],
            frame["communication_centered"],
        )[0, 1]
    )
    return {
        "n_models": int(frame["model_label"].nunique()),
        "n_model_prompt_points": int(len(frame)),
        "pearson_r_model_centered": observed_r,
        "within_model_permutation_p": permutation_p_value(
            frame,
            observed_r,
            permutations=permutations,
            seed=seed,
        ),
        "n_permutations": permutations,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the combined system prompt tradeoff analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(
        args.input,
        permutations=args.permutations,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
