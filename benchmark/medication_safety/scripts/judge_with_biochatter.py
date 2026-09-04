"""Apply the calibrated medication safety judge through BioChatter."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmark.medication_safety.biochatter_adapter import (
    load_medication_safety_instances,
    safe_model_name,
)
from benchmark.medication_safety.biochatter_provider import (
    DEFAULT_KEY_ENV,
    create_conversation,
)
from benchmark.medication_safety.scripts.judge_subcriteria import (
    METRICS,
    load_protocol,
    parse_subcriteria,
    render_prompt,
    threshold_label,
)
from benchmark.medication_safety.scripts.score_responses import expand_response_rows

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation


logger = logging.getLogger(__name__)


JUDGEMENT_COLUMNS = [
    "judge_model",
    "evaluated_model",
    "case_id",
    "subtask",
    "system_prompt",
    "patient_attitude",
    "response_iteration",
    "md5_hash",
    "metric",
    "judge_iteration",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "criterion_label",
]


def default_judgement_path(judge_model: str) -> Path:
    """Return a local path for subcriterion-level judge output."""
    name = safe_model_name(judge_model)
    return Path("benchmark/results") / f"medication_safety_judgements_{name}.csv"


def load_response_rows(path: Path) -> list[dict[str, str]]:
    """Load raw or standard BioChatter response rows in long form."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        msg = "The response file is empty."
        raise ValueError(msg)
    return expand_response_rows(rows)


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JUDGEMENT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (
                row["judge_model"],
                row["evaluated_model"],
                row["md5_hash"],
                row["response_iteration"],
                row["metric"],
                row["judge_iteration"],
            )
            for row in csv.DictReader(handle)
        }


def _judge_once(
    conversation: Conversation,
    system_message: str,
    prompt: str,
    max_attempts: int,
) -> dict[str, str]:
    error = None
    for _ in range(max_attempts):
        conversation.reset()
        conversation.append_system_message(system_message)
        raw, _, _ = conversation.query(prompt)
        try:
            return parse_subcriteria(raw)
        except (ValueError, TypeError) as exc:
            error = exc
    msg = f"Judge returned no valid subcriterion JSON: {error}"
    raise RuntimeError(msg)


def summarize_judgements(
    judgement_path: Path,
    output_path: Path,
    expected_iterations: int = 2,
) -> Path:
    """Create descriptive and strict response-level communication scores."""
    if expected_iterations < 1:
        msg = "expected_iterations must be at least 1"
        raise ValueError(msg)

    rows = pd.read_csv(judgement_path)
    if not rows["criterion_label"].isin([0, 1]).all():
        msg = "criterion_label must contain only binary values"
        raise ValueError(msg)

    group_columns = [
        "judge_model",
        "evaluated_model",
        "case_id",
        "subtask",
        "system_prompt",
        "patient_attitude",
        "response_iteration",
        "md5_hash",
        "metric",
    ]
    scores = rows.groupby(group_columns, as_index=False).agg(
        descriptive_score=("criterion_label", "mean"),
        judge_iterations=("criterion_label", "count"),
        positive_iterations=("criterion_label", "sum"),
    )
    incomplete = scores["judge_iterations"] != expected_iterations
    if incomplete.any():
        count = int(incomplete.sum())
        msg = f"Found {count} response-metric groups without exactly {expected_iterations} judge iterations."
        raise ValueError(msg)
    scores["strict_binary_label"] = (scores["positive_iterations"] == expected_iterations).astype(int)
    scores = scores.drop(columns="positive_iterations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Judge medication safety responses through BioChatter.",
    )
    parser.add_argument("input", type=Path, help="BioChatter response CSV.")
    parser.add_argument("--provider", choices=sorted(DEFAULT_KEY_ENV), required=True)
    parser.add_argument("--model", required=True, help="Judge model identifier.")
    parser.add_argument("--api-key-env")
    parser.add_argument("--base-url")
    parser.add_argument("--judge-iterations", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Judge only the first N responses.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scores-output", type=Path)
    args = parser.parse_args()

    protocol = load_protocol()
    judge_iterations = args.judge_iterations or protocol["protocol"]["judge_iterations"]
    system_message = protocol["judge_system_message"]
    conversation = create_conversation(
        provider=args.provider,
        model_name=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )

    instances = load_medication_safety_instances()
    by_hash = {instance["hash"]: instance for instance in instances}
    response_rows = load_response_rows(args.input)
    if args.limit is not None:
        response_rows = response_rows[: args.limit]

    output_path = args.output or default_judgement_path(args.model)
    completed = _completed_keys(output_path)
    for row_index, row in enumerate(response_rows, start=1):
        instance = by_hash.get(row.get("md5_hash", ""))
        if instance is None:
            msg = f"No expanded benchmark instance matches row {row_index}."
            raise ValueError(msg)

        for metric in sorted(METRICS):
            prompt = render_prompt(
                metric,
                system_messages="\n".join(instance["input"]["system_messages"]),
                prompt=instance["input"]["prompt"],
                patient_attitude=instance["patient_attitude"],
                response=row["response"],
            )
            for judge_iteration in range(1, judge_iterations + 1):
                key = (
                    args.model,
                    row["model_name"],
                    instance["hash"],
                    row["response_iteration"],
                    metric,
                    str(judge_iteration),
                )
                if key in completed:
                    continue
                values = _judge_once(
                    conversation,
                    system_message,
                    prompt,
                    args.max_attempts,
                )
                label = threshold_label(
                    values,
                    metric,
                    instance["patient_attitude"],
                )
                _append_row(
                    output_path,
                    {
                        "judge_model": args.model,
                        "evaluated_model": row["model_name"],
                        "case_id": instance["case_id"],
                        "subtask": instance["case"],
                        "system_prompt": instance["system_prompt"],
                        "patient_attitude": instance["patient_attitude"],
                        "response_iteration": row["response_iteration"],
                        "md5_hash": instance["hash"],
                        "metric": metric,
                        "judge_iteration": judge_iteration,
                        **values,
                        "criterion_label": label,
                    },
                )
                completed.add(key)
        logger.info(
            "[%s/%s] judged %s",
            row_index,
            len(response_rows),
            instance["case"],
        )

    scores_output = args.scores_output or output_path.with_name(
        f"{output_path.stem}_scores.csv",
    )
    summarize_judgements(
        output_path,
        scores_output,
        expected_iterations=judge_iterations,
    )
    logger.info("Judgements written to %s", output_path)
    logger.info("Response-level scores written to %s", scores_output)


if __name__ == "__main__":
    main()
