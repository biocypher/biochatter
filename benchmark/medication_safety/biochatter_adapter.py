"""Run medication safety cases through BioChatter conversations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from benchmark.benchmark_utils import write_responses_to_file
from benchmark.load_dataset import get_benchmark_dataset

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation


TASK_NAME = "medication_safety_response_generation"
DEFAULT_ITERATIONS = 4
RESPONSE_COLUMNS = [
    "model_name",
    "case_id",
    "subtask",
    "age",
    "prompt",
    "response",
    "expected_answer",
    "summary",
    "key_words",
    "type",
    "iterations",
    "md5_hash",
    "datetime",
    "biochatter_version",
]


def load_medication_safety_instances() -> list[dict[str, Any]]:
    """Load the 320 native BioChatter benchmark instances."""
    data = get_benchmark_dataset()
    return data["drug_adverse_effect_assessment"]


def safe_model_name(model_name: str) -> str:
    """Return a model name that is safe to use in a result filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def default_response_path(model_name: str) -> Path:
    """Return the standard local response path for a model."""
    filename = f"{TASK_NAME}_{safe_model_name(model_name)}_response.csv"
    return Path("benchmark") / "results" / filename


def ensure_response_file(path: Path) -> None:
    """Create an empty response file using BioChatter's response schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=RESPONSE_COLUMNS).to_csv(path, index=False)


def response_already_recorded(path: Path, model_name: str, md5_hash: str) -> bool:
    """Check whether a model-instance response is already present."""
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    return bool(
        ((rows["model_name"] == model_name) & (rows["md5_hash"] == md5_hash)).any(),
    )


def _remove_existing_record(path: Path, model_name: str, md5_hash: str) -> None:
    """Remove one exact model-instance record before an explicit rerun."""
    rows = pd.read_csv(path)
    keep = ~((rows["model_name"] == model_name) & (rows["md5_hash"] == md5_hash))
    rows.loc[keep].to_csv(path, index=False)


def generate_responses(
    conversation: Conversation,
    instance: dict[str, Any],
    iterations: int = DEFAULT_ITERATIONS,
) -> list[str]:
    """Generate repeated responses for one expanded benchmark instance."""
    if iterations < 1:
        msg = "iterations must be at least 1"
        raise ValueError(msg)

    responses = []
    for _ in range(iterations):
        conversation.reset()
        for message in instance["input"]["system_messages"]:
            conversation.append_system_message(message)
        response, _, _ = conversation.query(instance["input"]["prompt"])
        responses.append(response)
    return responses


def expected_concepts(instance: dict[str, Any]) -> list[str]:
    """Flatten the curated case concepts for the standard response record."""
    expected = instance["expected"]
    adverse_effects = expected.get("adverse_effects", {})
    concepts = [concept for frequency_terms in adverse_effects.values() for concept in frequency_terms]
    concepts.extend(expected.get("contraindications", []))
    return concepts


def run_and_record_instance(  # noqa: PLR0913
    conversation: Conversation,
    model_name: str,
    instance: dict[str, Any],
    *,
    output_path: Path | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    force: bool = False,
) -> Path:
    """Generate and store one case using BioChatter's response CSV schema."""
    path = output_path or default_response_path(model_name)
    ensure_response_file(path)

    if response_already_recorded(path, model_name, instance["hash"]):
        if not force:
            return path
        _remove_existing_record(path, model_name, instance["hash"])

    responses = generate_responses(conversation, instance, iterations)
    expected = instance["expected"]
    write_responses_to_file(
        model_name=model_name,
        case_id=instance["case_id"],
        subtask=instance["case"],
        individual=expected.get("individual", ""),
        prompt=instance["input"]["prompt"],
        responses=responses,
        expected_answer=expected.get("answer", [""])[0],
        summary=expected.get("summary", ""),
        key_words=expected_concepts(instance),
        type=instance["type"],
        iterations=str(iterations),
        md5_hash=instance["hash"],
        file_path=str(path),
    )
    return path
