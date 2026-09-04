from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Any

import yaml


NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_benchmark_cases(path: str | Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    rows = data["drug_adverse_effect_assessment"]
    return [row for row in rows if "case_id" in row]


def load_patient_attitudes(path: str | Path) -> dict[str, str]:
    data = load_yaml(path)
    for row in data["drug_adverse_effect_assessment"]:
        if "patient_attitude_templates" in row:
            return row["patient_attitude_templates"]
    raise ValueError("No patient attitude templates found.")


def load_system_prompts(path: str | Path) -> dict[str, list[str]]:
    data = load_yaml(path)
    for row in data["drug_adverse_effect_assessment"]:
        if "general_system_messages" in row:
            return row["general_system_messages"]
    raise ValueError("No system prompt definitions found.")


def build_user_prompt(
    case: dict[str, Any],
    patient_attitude: str,
    attitude_map: dict[str, str],
) -> str:
    paragraph = case["input"]["medication_context"]["paragraph"]
    return f"{paragraph}\n\n{attitude_map[patient_attitude]}"


def load_replacements(path: str | Path) -> dict[str, str]:
    replacements: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_form", "").strip().lower()
            target = row.get("normalized_form", "").strip().lower()
            if source and target:
                replacements[source] = target
    return replacements


def normalize_text(text: str, replacements: dict[str, str] | None = None) -> str:
    normalized = text.lower()
    for source, target in (replacements or {}).items():
        normalized = normalized.replace(source, target)
    normalized = NON_ALNUM.sub(" ", normalized)
    return " ".join(normalized.split())


def load_synonym_groups(path: str | Path) -> list[set[str]]:
    groups: list[set[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variants = {
                variant.strip()
                for variant in row.get("variants", "").split(";")
                if variant.strip()
            }
            if variants:
                groups.append(variants)
    return groups


def synonym_aware_match(
    expected_term: str,
    response: str,
    replacements: dict[str, str],
    groups: list[set[str]],
) -> bool:
    normalized_response = normalize_text(response, replacements)
    normalized_expected = normalize_text(expected_term, replacements)
    variants = {normalized_expected}
    for group in groups:
        normalized_group = {normalize_text(term, replacements) for term in group}
        if normalized_expected in normalized_group:
            variants.update(normalized_group)
    padded_response = f" {normalized_response} "
    return any(variant and f" {variant} " in padded_response for variant in variants)
