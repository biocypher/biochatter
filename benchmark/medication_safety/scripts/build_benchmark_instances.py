from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from benchmark.medication_safety.scripts.medication_safety_utils import (
    build_user_prompt,
    load_benchmark_cases,
    load_patient_attitudes,
    load_system_prompts,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "benchmark_medication_safety_data.yaml"

PROMPT_LABELS = {
    "none": "none",
    "minimal": "minimal",
    "role_encouraging": "role_encouraging",
    "role_attitude_sensitive": "patient_attitude_sensitive",
}


def one_line(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\\n")


def iter_benchmark_instances(data_path: Path):
    cases = load_benchmark_cases(data_path)
    system_prompts = load_system_prompts(data_path)
    patient_attitudes = load_patient_attitudes(data_path)

    for case in cases:
        context = case["input"]["medication_context"]
        for system_prompt_key, system_prompt_lines in system_prompts.items():
            for patient_attitude in patient_attitudes:
                yield {
                    "case_id": case["case_id"],
                    "medication": context["medicine"],
                    "indication": context["indication"],
                    "system_prompt_key": system_prompt_key,
                    "system_prompt_label": PROMPT_LABELS.get(system_prompt_key, system_prompt_key),
                    "patient_attitude": patient_attitude,
                    "system_prompt": one_line("\n".join(system_prompt_lines)),
                    "user_prompt": one_line(
                        build_user_prompt(case, patient_attitude, patient_attitudes)
                    ),
                }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 320 medication safety benchmark instances."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Benchmark YAML file.")
    parser.add_argument("--output", type=Path, help="Optional output CSV path. Defaults to stdout.")
    args = parser.parse_args()

    rows = list(iter_benchmark_instances(args.data))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", encoding="utf-8", newline="")
    else:
        handle = sys.stdout

    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.output:
        print(f"Wrote {len(rows)} benchmark instances to {args.output}")


if __name__ == "__main__":
    main()
