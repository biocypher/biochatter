from __future__ import annotations

import argparse
import ast
import csv
import logging
import sys
from contextlib import nullcontext, suppress
from pathlib import Path

from benchmark.medication_safety.scripts.conservative_term_matching import (
    MedicationSafetyScorer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "benchmark_medication_safety_data.yaml"
DEFAULT_REPLACEMENTS = ROOT / "results" / "term_matching" / "normalization_replacements.csv"
DEFAULT_SYNONYMS = ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv"
SUBTASK_COMPONENTS = 3


logger = logging.getLogger(__name__)


def expand_response_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Expand BioChatter response-list rows into one row per iteration."""
    expanded = []
    for row in rows:
        value = row["response"]
        is_biochatter_row = {"subtask", "iterations", "md5_hash"}.issubset(row)
        parsed = value
        if is_biochatter_row:
            with suppress(SyntaxError, ValueError):
                parsed = ast.literal_eval(value)
        responses = parsed if isinstance(parsed, list) else [value]

        for index, response in enumerate(responses, start=1):
            expanded_row = {**row, "response": str(response)}
            if not expanded_row.get("response_iteration"):
                expanded_row["response_iteration"] = str(index)

            subtask = expanded_row.get("subtask", "")
            if subtask and (not expanded_row.get("system_prompt") or not expanded_row.get("patient_attitude")):
                subtask_parts = subtask.rsplit(":", 2)
                if len(subtask_parts) == SUBTASK_COMPONENTS:
                    _, system_prompt, patient_attitude = subtask_parts
                    expanded_row["system_prompt"] = system_prompt
                    expanded_row["patient_attitude"] = patient_attitude
            expanded.append(expanded_row)
    return expanded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Apply the conservative medication safety term matcher to response text.",
    )
    parser.add_argument("input", type=Path, help="CSV with case_id and response columns.")
    parser.add_argument("--output", type=Path, help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        msg = "The input CSV is empty."
        raise ValueError(msg)
    required = {"case_id", "response"}
    missing = required - set(rows[0])
    if missing:
        msg = f"Missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    rows = expand_response_rows(rows)

    scorer = MedicationSafetyScorer.from_files(
        args.data,
        args.replacements,
        args.synonyms,
    )
    scored_rows = [{**row, **scorer.score(row["case_id"], row["response"])} for row in rows]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_context = args.output.open("w", encoding="utf-8", newline="")
    else:
        output_context = nullcontext(sys.stdout)
    with output_context as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)

    if args.output:
        logger.info("Wrote %s scored responses to %s", len(scored_rows), args.output)


if __name__ == "__main__":
    main()
