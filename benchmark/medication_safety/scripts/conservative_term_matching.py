from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Iterable

from benchmark.medication_safety.scripts.medication_safety_utils import (
    load_benchmark_cases,
    load_replacements,
    load_synonym_groups,
    normalize_text,
)


MAX_TOKEN_WINDOW = 12
STRICT_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "because",
    "caused",
    "in",
    "of",
    "or",
    "patient",
    "relevant",
    "the",
    "to",
    "use",
    "with",
}
PHRASE_ONLY_TERMS = {
    "abnormal coordination",
    "kidney infection",
    "musculoskeletal chest pain",
    "stomach ulcer",
    "weight decrease",
    "weight decreased",
    "weight gain",
    "weight increase",
    "weight increased",
    "weight loss",
}
CONCEPT_EXCLUSION_PHRASES = {
    "pollakiuria": {"frequent urge", "frequent urges"},
}

LIST_ITEM = re.compile(r"^(?P<indent>\s*)(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.*)$")
MARKDOWN_PREFIX = re.compile(r"^\s*(?:#{1,6}\s*|>\s*)")
MARKDOWN_WRAPPERS = re.compile(r"[*_`]")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Variant:
    phrase: str
    required_tokens: frozenset[str]


def _clean_line(line: str) -> str:
    value = MARKDOWN_PREFIX.sub("", line.strip())
    return MARKDOWN_WRAPPERS.sub("", value).strip()


def _is_heading(raw_line: str, cleaned: str) -> bool:
    if not cleaned:
        return False
    raw = raw_line.strip()
    if raw.startswith("#"):
        return True
    if cleaned.endswith(":") and len(cleaned.split()) <= 18:
        return True
    if raw.startswith("**") and raw.endswith("**") and len(cleaned.split()) <= 18:
        return True
    lowered = cleaned.lower().rstrip(":")
    heading_terms = (
        "very common adverse",
        "common adverse",
        "uncommon adverse",
        "contraindication",
        "references",
        "side effects",
        "adverse reactions",
    )
    return len(cleaned.split()) <= 12 and any(term in lowered for term in heading_terms)


def structured_segments(response: str) -> list[str]:
    """Split text into local matching contexts without joining list siblings."""

    segments: list[str] = []
    heading = ""
    paragraph: list[str] = []
    current_item: list[str] = []
    parent_items: list[tuple[int, str]] = []

    def add_text(value: str, *, split_sentences: bool) -> None:
        value = value.strip()
        if not value:
            return
        parts = SENTENCE_BREAK.split(value) if split_sentences else [value]
        segments.extend(part.strip() for part in parts if part.strip())

    def flush_paragraph() -> None:
        if paragraph:
            add_text(f"{heading} {' '.join(paragraph)}".strip(), split_sentences=True)
            paragraph.clear()

    def flush_item() -> None:
        if current_item:
            add_text(" ".join(current_item), split_sentences=False)
            current_item.clear()

    for raw_line in response.splitlines():
        if not raw_line.strip():
            flush_item()
            flush_paragraph()
            continue

        list_match = LIST_ITEM.match(raw_line)
        if list_match:
            flush_item()
            flush_paragraph()
            indent = len(list_match.group("indent").replace("\t", "    "))
            content = _clean_line(list_match.group("text"))
            while parent_items and parent_items[-1][0] >= indent:
                parent_items.pop()
            context = [heading] if heading else []
            context.extend(parent for _, parent in parent_items)
            context.append(content)
            current_item.extend(part for part in context if part)
            if content.endswith(":") and len(content.split()) <= 18:
                parent_items.append((indent, content))
            continue

        cleaned = _clean_line(raw_line)
        if _is_heading(raw_line, cleaned):
            flush_item()
            flush_paragraph()
            heading = cleaned.rstrip(":")
            parent_items.clear()
            add_text(heading, split_sentences=False)
            continue

        if "|" in raw_line and raw_line.count("|") >= 2:
            flush_item()
            flush_paragraph()
            add_text(f"{heading} {cleaned}".strip(), split_sentences=False)
            continue

        if current_item:
            current_item.append(cleaned)
        else:
            paragraph.append(cleaned)

    flush_item()
    flush_paragraph()
    return segments or [response]


def _allergy_variants(term: str) -> set[str]:
    target = ""
    if term.startswith("allergy to "):
        target = term.removeprefix("allergy to ").strip()
    elif term.startswith("hypersensitivity to "):
        target = term.removeprefix("hypersensitivity to ").strip()
    elif term.endswith(" allergy"):
        target = term.removesuffix(" allergy").strip()
    elif term.endswith(" hypersensitivity"):
        target = term.removesuffix(" hypersensitivity").strip()
    if not target:
        return set()
    return {
        f"allergy to {target}",
        f"hypersensitivity to {target}",
        f"allergic to {target}",
        f"{target} allergy",
        f"{target} hypersensitivity",
    }


def _minimal_token_window(position_map: dict[str, list[int]], required: set[str]) -> int | None:
    if not required or any(token not in position_map for token in required):
        return None
    events = sorted(
        (position, token)
        for token in required
        for position in position_map[token]
    )
    counts: Counter[str] = Counter()
    have = 0
    left = 0
    best: int | None = None
    for right_position, token in events:
        counts[token] += 1
        if counts[token] == 1:
            have += 1
        while have == len(required):
            left_position, left_token = events[left]
            width = right_position - left_position + 1
            best = width if best is None else min(best, width)
            counts[left_token] -= 1
            if counts[left_token] == 0:
                have -= 1
            left += 1
    return best


class ConceptMatcher:
    """Match canonical concepts using phrases and local 12-token windows."""

    def __init__(
        self,
        expected_terms: Iterable[str],
        replacements: dict[str, str],
        synonym_groups: list[set[str]],
    ) -> None:
        self.replacements = replacements
        normalized_expected = {
            normalize_text(term, replacements)
            for term in expected_terms
            if term
        }

        parent: dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for term in normalized_expected:
            find(term)
        for group in synonym_groups:
            members = sorted(
                {
                    normalize_text(term, replacements)
                    for term in group
                    if term
                }
            )
            if not members:
                continue
            for member in members:
                find(member)
            for member in members[1:]:
                union(members[0], member)

        components: dict[str, set[str]] = defaultdict(set)
        for term in parent:
            components[find(term)].add(term)

        self.term_to_concept = {term: find(term) for term in normalized_expected}
        expected_by_concept: dict[str, set[str]] = defaultdict(set)
        for term, concept in self.term_to_concept.items():
            expected_by_concept[concept].add(term)

        self.display: dict[str, str] = {}
        self.variants: dict[str, list[Variant]] = {}
        for concept, concept_expected in expected_by_concept.items():
            self.display[concept] = sorted(
                concept_expected,
                key=lambda value: (len(value), value),
            )[0]
            phrases = set(components[concept])
            for phrase in list(phrases):
                phrases.update(
                    normalize_text(alias, replacements)
                    for alias in _allergy_variants(phrase)
                )
            self.variants[concept] = [
                Variant(
                    phrase=phrase,
                    required_tokens=frozenset(
                        token
                        for token in phrase.split()
                        if token not in STRICT_GENERIC_TOKENS
                    ),
                )
                for phrase in sorted(phrases)
                if phrase
            ]

        token_frequency: Counter[str] = Counter()
        for variants in self.variants.values():
            for variant in variants:
                token_frequency.update(variant.required_tokens)
        self.phrase_anchors: dict[str, list[tuple[str, Variant]]] = defaultdict(list)
        self.token_anchors: dict[str, list[tuple[str, Variant]]] = defaultdict(list)
        for concept, variants in self.variants.items():
            for variant in variants:
                phrase_tokens = variant.phrase.split()
                if phrase_tokens:
                    self.phrase_anchors[phrase_tokens[0]].append((concept, variant))
                if len(variant.required_tokens) >= 2:
                    anchor = min(
                        variant.required_tokens,
                        key=lambda token: (token_frequency[token], token),
                    )
                    self.token_anchors[anchor].append((concept, variant))

    @property
    def concepts(self) -> set[str]:
        return set(self.variants)

    def concept_for(self, term: str) -> str:
        return self.term_to_concept[normalize_text(term, self.replacements)]

    def match(self, response: str) -> set[str]:
        found: set[str] = set()
        for raw_segment in structured_segments(response):
            segment = normalize_text(raw_segment, self.replacements)
            if not segment:
                continue
            tokens = segment.split()
            positions: dict[str, list[int]] = defaultdict(list)
            for index, token in enumerate(tokens):
                positions[token].append(index)
            token_set = set(positions)
            padded = f" {segment} "

            phrase_candidates = {
                candidate
                for token in token_set
                for candidate in self.phrase_anchors.get(token, [])
            }
            for concept, variant in phrase_candidates:
                if concept in found or f" {variant.phrase} " not in padded:
                    continue
                display = self.display[concept]
                exclusions = CONCEPT_EXCLUSION_PHRASES.get(display, set())
                if any(f" {phrase} " in padded for phrase in exclusions):
                    continue
                found.add(concept)

            token_candidates = {
                candidate
                for token in token_set
                for candidate in self.token_anchors.get(token, [])
            }
            for concept, variant in token_candidates:
                if concept in found:
                    continue
                display = self.display[concept]
                if display in PHRASE_ONLY_TERMS:
                    continue
                exclusions = CONCEPT_EXCLUSION_PHRASES.get(display, set())
                if any(f" {phrase} " in padded for phrase in exclusions):
                    continue
                if not variant.required_tokens.issubset(token_set):
                    continue
                width = _minimal_token_window(positions, set(variant.required_tokens))
                if width is not None and width <= MAX_TOKEN_WINDOW:
                    found.add(concept)
        return found


class MedicationSafetyScorer:
    """Score adverse reaction and contraindication retrieval for benchmark cases."""

    def __init__(
        self,
        cases: list[dict],
        replacements: dict[str, str],
        synonym_groups: list[set[str]],
    ) -> None:
        adverse_terms = {
            term
            for case in cases
            for terms in case["expected"]["adverse_effects"].values()
            for term in (terms or [])
        }
        contraindication_terms = {
            term
            for case in cases
            for term in (case["expected"].get("contraindications") or [])
        }
        self.adverse_matcher = ConceptMatcher(adverse_terms, replacements, synonym_groups)
        self.contraindication_matcher = ConceptMatcher(
            contraindication_terms,
            replacements,
            synonym_groups,
        )
        self.cases: dict[str, dict[str, object]] = {}
        for case in cases:
            adverse_by_frequency = {
                frequency: {
                    self.adverse_matcher.concept_for(term)
                    for term in (terms or [])
                }
                for frequency, terms in case["expected"]["adverse_effects"].items()
            }
            expected_adverse = set().union(*adverse_by_frequency.values())
            expected_contraindications = {
                self.contraindication_matcher.concept_for(term)
                for term in (case["expected"].get("contraindications") or [])
            }
            self.cases[str(case["case_id"])] = {
                "adverse_by_frequency": adverse_by_frequency,
                "expected_adverse": expected_adverse,
                "expected_contraindications": expected_contraindications,
            }

    @classmethod
    def from_files(
        cls,
        benchmark_path: str | Path,
        replacements_path: str | Path,
        synonym_groups_path: str | Path,
    ) -> "MedicationSafetyScorer":
        return cls(
            load_benchmark_cases(benchmark_path),
            load_replacements(replacements_path),
            load_synonym_groups(synonym_groups_path),
        )

    @staticmethod
    def _ratio(numerator: int, denominator: int, *, empty: float = nan) -> float:
        return empty if denominator == 0 else numerator / denominator

    def score(self, case_id: str | int, response: str) -> dict[str, float]:
        case = self.cases[str(case_id)]
        adverse_by_frequency = case["adverse_by_frequency"]
        expected_adverse = case["expected_adverse"]
        expected_contraindications = case["expected_contraindications"]
        assert isinstance(adverse_by_frequency, dict)
        assert isinstance(expected_adverse, set)
        assert isinstance(expected_contraindications, set)

        recognized_adverse = self.adverse_matcher.match(response)
        recognized_contraindications = self.contraindication_matcher.match(response)
        matched_adverse = recognized_adverse & expected_adverse
        matched_contraindications = recognized_contraindications & expected_contraindications

        result = {
            "adverse_effects_recall": self._ratio(len(matched_adverse), len(expected_adverse)),
            "adverse_effects_specificity": self._ratio(
                len(matched_adverse),
                len(recognized_adverse),
                empty=0.0,
            ),
            "contraindications_recall": self._ratio(
                len(matched_contraindications),
                len(expected_contraindications),
            ),
            "contraindications_specificity": self._ratio(
                len(matched_contraindications),
                len(recognized_contraindications),
                empty=0.0,
            ),
        }
        for frequency in ("very_common", "common", "uncommon"):
            expected = adverse_by_frequency.get(frequency, set())
            result[f"{frequency}_adverse_effects_coverage"] = self._ratio(
                len(matched_adverse & expected),
                len(expected),
            )
        return result
