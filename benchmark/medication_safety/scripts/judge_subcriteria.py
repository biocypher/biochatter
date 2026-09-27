from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "prompts" / "llm_judge_prompts.yaml"
METRICS = {
    "understandability",
    "usefulness",
    "patient_attitude_responsiveness",
}


def load_protocol(path: str | Path = DEFAULT_PROMPTS) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def render_prompt(
    metric: str,
    *,
    system_messages: str,
    prompt: str,
    patient_attitude: str,
    response: str,
    path: str | Path = DEFAULT_PROMPTS,
) -> str:
    protocol = load_protocol(path)
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    template = protocol["prompts"][metric]["prompt"]
    return template.format(
        system_messages=system_messages,
        prompt=prompt,
        patient_attitude=patient_attitude,
        response=response,
    )


def parse_subcriteria(raw: str) -> dict[str, str]:
    text = raw.strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge output does not contain a JSON object")
    parsed = json.loads(match.group(0))
    expected = {f"q{index}" for index in range(1, 6)}
    if set(parsed) != expected:
        raise ValueError(f"Unexpected subcriterion keys: {sorted(parsed)}")
    values = {key: str(value).strip().lower() for key, value in parsed.items()}
    if any(value not in {"yes", "no", "not_applicable"} for value in values.values()):
        raise ValueError(f"Unexpected subcriterion values: {values}")
    return values


def threshold_label(
    values: dict[str, str],
    metric: str,
    patient_attitude: str,
    path: str | Path = DEFAULT_PROMPTS,
) -> int:
    protocol = load_protocol(path)
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    expected_not_applicable: set[str] = set()
    if metric == "patient_attitude_responsiveness":
        expected_not_applicable = {
            "q3" if patient_attitude in {"anxious", "very_anxious"} else "q2"
        }
    actual_not_applicable = {
        key for key, value in values.items() if value == "not_applicable"
    }
    if actual_not_applicable != expected_not_applicable:
        raise ValueError(
            "Unexpected not_applicable assignment: "
            f"expected {expected_not_applicable}, found {actual_not_applicable}"
        )

    applicable = [value for value in values.values() if value != "not_applicable"]
    expected_count = protocol["prompts"][metric]["applicable_subcriteria"]
    if len(applicable) != expected_count:
        raise ValueError(f"Expected {expected_count} applicable subcriteria")
    threshold = protocol["prompts"][metric]["threshold"]
    return int(sum(value == "yes" for value in applicable) >= threshold)
