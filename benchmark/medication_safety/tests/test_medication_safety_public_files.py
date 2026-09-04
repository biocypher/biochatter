from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from benchmark.load_dataset import (
    _expand_medication_safety_test_cases,
    _get_yaml_data,
)
from benchmark.medication_safety.biochatter_adapter import (
    RESPONSE_COLUMNS,
    expected_concepts,
    generate_responses,
    run_and_record_instance,
)
from benchmark.medication_safety.scripts.analyze_system_prompt_tradeoff import (
    load_and_center,
)
from benchmark.medication_safety.scripts.analyze_final_scores import (
    analyze_communication_within,
    analyze_structured_within,
)
from benchmark.medication_safety.scripts.build_benchmark_instances import (
    iter_benchmark_instances,
)
from benchmark.medication_safety.scripts.conservative_term_matching import (
    MedicationSafetyScorer,
)
from benchmark.medication_safety.scripts.judge_subcriteria import (
    parse_subcriteria,
    render_prompt,
    threshold_label,
)
from benchmark.medication_safety.scripts.judge_with_biochatter import (
    JUDGEMENT_COLUMNS,
    summarize_judgements,
)
from benchmark.medication_safety.scripts.medication_safety_utils import (
    build_user_prompt,
    load_benchmark_cases,
    load_patient_attitudes,
    load_replacements,
    load_synonym_groups,
    load_system_prompts,
    synonym_aware_match,
)
from benchmark.medication_safety.scripts.score_responses import expand_response_rows
from benchmark.medication_safety.scripts.update_file_manifest import manifest_payload


ROOT = Path(__file__).resolve().parents[1]


class FakeConversation:
    """Minimal BioChatter conversation used by adapter tests."""

    def __init__(self) -> None:
        self.messages = []
        self.queries = []

    def reset(self) -> None:
        self.messages = []

    def append_system_message(self, message: str) -> None:
        self.messages.append(message)

    def query(self, prompt: str):
        self.queries.append((list(self.messages), prompt))
        return (f"response {len(self.queries)}", None, None)


def test_benchmark_yaml_loads_with_20_cases() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    cases = load_benchmark_cases(data_path)
    assert len(cases) == 20
    assert all(case["expected"]["contraindications"] is not None for case in cases)


def test_prompt_matrix_definitions_load() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    prompts = load_system_prompts(data_path)
    attitudes = load_patient_attitudes(data_path)
    assert set(prompts) == {
        "none",
        "minimal",
        "role_encouraging",
        "role_attitude_sensitive",
    }
    assert set(attitudes) == {"very_anxious", "anxious", "neutral", "confident"}
    first_prompt = build_user_prompt(
        load_benchmark_cases(data_path)[0], "anxious", attitudes
    )
    assert "What are the adverse reactions and contraindications?" in first_prompt


def test_benchmark_instance_builder_creates_320_instances() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    rows = list(iter_benchmark_instances(data_path))
    assert len(rows) == 320
    assert rows[0]["system_prompt_label"] in {
        "none",
        "minimal",
        "role_encouraging",
        "patient_attitude_sensitive",
    }


def test_native_biochatter_expansion_creates_the_same_320_instances() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    with data_path.open("r", encoding="utf-8") as handle:
        raw_rows = yaml.safe_load(handle)["drug_adverse_effect_assessment"]

    rows = _expand_medication_safety_test_cases(raw_rows)
    assert len(rows) == 320
    assert len({row["case"] for row in rows}) == 320
    assert {row["system_prompt"] for row in rows} == {
        "none",
        "minimal",
        "role_encouraging",
        "patient_attitude_sensitive",
    }
    assert {row["patient_attitude"] for row in rows} == {
        "very_anxious",
        "anxious",
        "neutral",
        "confident",
    }
    assert all("medication_context" not in row["input"] for row in rows)


def test_biochatter_adapter_applies_messages_and_iterations() -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    with data_path.open("r", encoding="utf-8") as handle:
        raw_rows = yaml.safe_load(handle)["drug_adverse_effect_assessment"]
    instance = next(
        row
        for row in _expand_medication_safety_test_cases(raw_rows)
        if row["case_id"] == "1" and row["system_prompt"] == "minimal"
    )
    conversation = FakeConversation()

    responses = generate_responses(conversation, instance, iterations=2)

    assert responses == ["response 1", "response 2"]
    assert len(conversation.queries) == 2
    assert conversation.queries[0][0] == instance["input"]["system_messages"]
    assert "pioglitazone" in conversation.queries[0][1]
    assert "feel very worried" in conversation.queries[0][1]


def test_biochatter_adapter_writes_standard_response_schema(tmp_path) -> None:
    data_path = ROOT / "data" / "benchmark_medication_safety_data.yaml"
    with data_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    instance = _get_yaml_data(raw)["drug_adverse_effect_assessment"][0]
    output = tmp_path / "responses.csv"

    run_and_record_instance(
        conversation=FakeConversation(),
        model_name="test/model",
        instance=instance,
        output_path=output,
        iterations=2,
    )

    rows = pd.read_csv(output)
    assert list(rows.columns) == RESPONSE_COLUMNS
    assert len(rows) == 1
    assert rows.loc[0, "model_name"] == "test/model"
    assert rows.loc[0, "iterations"] == 2
    assert "pioglitazone" in rows.loc[0, "prompt"]
    assert "edema with insulin combination" in rows.loc[0, "key_words"]
    assert expected_concepts(instance)


def test_scoring_expands_standard_biochatter_response_rows() -> None:
    rows = expand_response_rows(
        [
            {
                "case_id": "1",
                "subtask": "drug:pioglitazone:none:anxious",
                "response": "['First response', 'Second response']",
                "iterations": "2",
                "md5_hash": "abc123",
            },
        ],
    )

    assert [row["response"] for row in rows] == [
        "First response",
        "Second response",
    ]
    assert [row["response_iteration"] for row in rows] == ["1", "2"]
    assert all(row["system_prompt"] == "none" for row in rows)
    assert all(row["patient_attitude"] == "anxious" for row in rows)


def test_scoring_preserves_list_like_text_in_a_simple_input_row() -> None:
    rows = expand_response_rows(
        [{"case_id": "1", "response": "['Nausea', 'Headache']"}],
    )

    assert len(rows) == 1
    assert rows[0]["response"] == "['Nausea', 'Headache']"
    assert rows[0]["response_iteration"] == "1"


def test_judge_summary_retains_split_scores_and_uses_strict_labels(
    tmp_path,
) -> None:
    judgement_path = tmp_path / "judgements.csv"
    score_path = tmp_path / "scores.csv"
    base = {
        "judge_model": "judge/model",
        "evaluated_model": "response/model",
        "case_id": "1",
        "subtask": "drug:pioglitazone:none:anxious",
        "system_prompt": "none",
        "patient_attitude": "anxious",
        "response_iteration": 1,
        "md5_hash": "abc123",
        "metric": "understandability",
        "q1": "yes",
        "q2": "yes",
        "q3": "yes",
        "q4": "yes",
        "q5": "no",
    }
    rows = [
        {**base, "judge_iteration": 1, "criterion_label": 1},
        {**base, "judge_iteration": 2, "criterion_label": 0},
    ]
    pd.DataFrame(rows, columns=JUDGEMENT_COLUMNS).to_csv(
        judgement_path,
        index=False,
    )

    summarize_judgements(judgement_path, score_path)

    summary = pd.read_csv(score_path).iloc[0]
    assert summary["descriptive_score"] == 0.5
    assert summary["strict_binary_label"] == 0
    assert summary["judge_iterations"] == 2


def test_judge_summary_rejects_incomplete_repeated_judgements(tmp_path) -> None:
    judgement_path = tmp_path / "judgements.csv"
    score_path = tmp_path / "scores.csv"
    row = {
        column: "yes" if column.startswith("q") else "value"
        for column in JUDGEMENT_COLUMNS
    }
    row.update(
        {
            "judge_iteration": 1,
            "criterion_label": 1,
            "response_iteration": 1,
        },
    )
    pd.DataFrame([row], columns=JUDGEMENT_COLUMNS).to_csv(
        judgement_path,
        index=False,
    )

    with pytest.raises(ValueError, match="without exactly 2 judge iterations"):
        summarize_judgements(judgement_path, score_path)


def test_public_term_matching_rules_detect_common_variants() -> None:
    replacements = load_replacements(
        ROOT / "results" / "term_matching" / "normalization_replacements.csv"
    )
    groups = load_synonym_groups(
        ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv"
    )
    response = (
        "The patient reported nosebleeds, joint pain, low blood sugar, and dyspnoea."
    )
    assert synonym_aware_match("epistaxis", response, replacements, groups)
    assert synonym_aware_match("arthralgia", response, replacements, groups)
    assert synonym_aware_match("hypoglycemia", response, replacements, groups)
    assert synonym_aware_match("dyspnea", response, replacements, groups)


def test_conservative_scorer_uses_local_token_windows() -> None:
    scorer = MedicationSafetyScorer.from_files(
        ROOT / "data" / "benchmark_medication_safety_data.yaml",
        ROOT / "results" / "term_matching" / "normalization_replacements.csv",
        ROOT / "results" / "term_matching" / "canonical_synonym_groups.csv",
    )
    reordered = scorer.score(
        1,
        "Common adverse reactions:\n"
        "- Insulin combination was associated with heart failure.",
    )
    assert reordered["common_adverse_effects_coverage"] > 0

    separated = scorer.score(1, "Common adverse reactions:\n- Weight\n- gain")
    assert separated["common_adverse_effects_coverage"] == 0


def test_response_model_settings_include_primary_and_additional_models() -> None:
    with (ROOT / "model_settings" / "response_model_settings.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["Cohort"] for row in rows} == {
        "Original response-model cohort",
        "Expanded response-model cohort",
    }
    gpt = next(row for row in rows if row["Response model"] == "GPT-5.4")
    assert float(gpt["Temperature"]) == 0.0


def test_final_judge_prompt_thresholds_match_manual_validation() -> None:
    with (ROOT / "prompts" / "llm_judge_prompts.yaml").open(
        "r",
        encoding="utf-8",
    ) as handle:
        prompt_data = yaml.safe_load(handle)
    prompts = prompt_data["prompts"]
    assert prompts["understandability"]["threshold"] == 4
    assert prompts["usefulness"]["threshold"] == 5
    assert prompts["patient_attitude_responsiveness"]["threshold"] == 4
    assert prompts["patient_attitude_responsiveness"]["applicable_subcriteria"] == 4

    rendered = render_prompt(
        "usefulness",
        system_messages="No explicit system instructions.",
        prompt="Medication question",
        patient_attitude="neutral",
        response="Generated response",
    )
    assert "Medication question" in rendered
    assert '"q1":"yes"' in rendered

    understandable = parse_subcriteria(
        '{"q1":"yes","q2":"yes","q3":"yes","q4":"yes","q5":"no"}'
    )
    assert threshold_label(understandable, "understandability", "neutral") == 1

    responsive = parse_subcriteria(
        '{"q1":"yes","q2":"yes","q3":"not_applicable",'
        '"q4":"yes","q5":"yes"}'
    )
    assert threshold_label(
        responsive,
        "patient_attitude_responsiveness",
        "very_anxious",
    ) == 1


def test_manual_validation_summaries_use_512_response_subset() -> None:
    summary_path = (
        ROOT / "results" / "manual_validation" / "manual_validation_by_metric.csv"
    )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "understandability",
        "usefulness",
        "patient_attitude_responsiveness",
    }
    assert {int(row["n"]) for row in rows} == {512}

    report_path = (
        ROOT / "results" / "manual_validation" / "manual_validation_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["unique_validated_responses_with_at_least_one_complete_metric"] == 512
    assert report["incomplete_replacement_rows"] == 0


def test_final_score_corpus_is_complete_and_unique() -> None:
    score_path = (
        ROOT / "results" / "final_analysis" / "final_response_metric_scores.csv"
    )
    scores = pd.read_csv(score_path)
    key = [
        "model_label",
        "metric",
        "case_id",
        "system_prompt",
        "patient_attitude",
        "response_index",
    ]
    assert len(scores) == 102_400
    assert scores["model_label"].nunique() == 8
    assert scores["metric"].nunique() == 10
    assert not scores.duplicated(key).any()
    assert scores["score"].dropna().between(0, 1).all()

    report = json.loads(
        (ROOT / "results" / "final_analysis" / "pairwise_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["structured_tests"] == 1764
    assert report["communication_tests"] == 756

    structured_pairwise = pd.read_csv(
        ROOT
        / "results"
        / "structured_metrics"
        / "final_structured_pairwise_wilcoxon.csv"
    )
    communication_pairwise = pd.read_csv(
        ROOT
        / "results"
        / "llm_judge_metrics"
        / "final_communication_pairwise_mcnemar.csv"
    )
    assert len(structured_pairwise) == 1764
    assert len(communication_pairwise) == 756

    structured_within = pd.read_csv(
        ROOT
        / "results"
        / "structured_metrics"
        / "final_structured_within_friedman.csv"
    )
    communication_within = pd.read_csv(
        ROOT
        / "results"
        / "llm_judge_metrics"
        / "final_communication_within_cochran_q.csv"
    )
    assert len(structured_within) == 112
    assert len(communication_within) == 48

    model_summary = pd.read_csv(
        ROOT
        / "results"
        / "final_analysis"
        / "summary_model_structured_communication.csv"
    ).set_index("model_label")
    assert np.isclose(
        model_summary.loc["GPT-5.4", "structured_score"],
        0.6646336028711517,
    )
    assert np.isclose(
        model_summary.loc["Gemini 3.5 Flash", "communication_score"],
        0.9516927083333334,
    )


def test_temperature_sensitivity_uses_complete_paired_gpt54_runs() -> None:
    report = json.loads(
        (
            ROOT
            / "results"
            / "temperature_sensitivity"
            / "temperature0_vs_temperature1_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["temperature_0_rows"] == 1280
    assert report["temperature_1_rows"] == 1280
    structured = next(
        row
        for row in report["paired_metric_summary"]
        if row["metric"] == "structured_score"
    )
    assert np.isclose(structured["temperature_0_mean"], 0.6646336028711517)


def test_combined_system_prompt_source_data_reproduces_correlation() -> None:
    result_dir = ROOT / "results" / "final_analysis"
    frame = load_and_center(
        result_dir / "summary_model_system_prompt_trajectory.csv"
    )
    correlation = float(
        np.corrcoef(
            frame["structured_centered"],
            frame["communication_centered"],
        )[0, 1]
    )
    assert len(frame) == 32
    assert np.isclose(correlation, -0.6272343655209744, rtol=0, atol=1e-12)


def test_within_model_omnibus_outputs_use_complete_paired_units() -> None:
    scores = pd.read_csv(
        ROOT / "results" / "final_analysis" / "final_response_metric_scores.csv"
    )
    released_structured = pd.read_csv(
        ROOT / "results" / "structured_metrics" / "final_structured_within_friedman.csv"
    )
    released_communication = pd.read_csv(
        ROOT / "results" / "llm_judge_metrics" / "final_communication_within_cochran_q.csv"
    )
    reproduced_structured = analyze_structured_within(
        scores[scores["metric_group"].eq("structured")]
    )
    reproduced_communication = analyze_communication_within(
        scores[scores["metric_group"].eq("judge")]
    )
    pd.testing.assert_frame_equal(
        released_structured,
        reproduced_structured,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        released_communication,
        reproduced_communication,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    incomplete_frequency_row = released_structured[
        released_structured["comparison_axis"].eq("patient_attitude")
        & released_structured["evaluated_model"].eq("Claude Sonnet 4.6")
        & released_structured["metric"].eq("very_common_adverse_effects_coverage")
    ].iloc[0]
    assert incomplete_frequency_row["n_units"] == 224
    assert np.isclose(incomplete_frequency_row["p_value"], 0.2488560739)


def test_published_summaries_and_figure_source_data_recompute() -> None:
    final_root = ROOT / "results" / "final_analysis"
    score_rows = pd.read_csv(final_root / "final_response_metric_scores.csv")
    released_levels = pd.read_csv(final_root / "final_response_level_scores.csv")
    keys = [
        "model_label",
        "case_id",
        "system_prompt",
        "patient_attitude",
        "response_index",
    ]
    structured_metrics = [
        "adverse_effects_recall",
        "very_common_adverse_effects_coverage",
        "common_adverse_effects_coverage",
        "uncommon_adverse_effects_coverage",
        "adverse_effects_specificity",
        "contraindications_recall",
        "contraindications_specificity",
    ]
    communication_metrics = [
        "understandability",
        "usefulness",
        "patient_attitude_responsiveness",
    ]
    wide = score_rows.pivot(index=keys, columns="metric", values="score").reset_index()
    recomputed_levels = wide[keys].copy()
    recomputed_levels["structured_score"] = wide[structured_metrics].mean(axis=1)
    recomputed_levels["communication_score"] = wide[communication_metrics].mean(axis=1)
    merged = released_levels.merge(
        recomputed_levels,
        on=keys,
        suffixes=("_released", "_recomputed"),
        validate="one_to_one",
    )
    assert len(merged) == 10_240
    assert np.allclose(
        merged["structured_score_released"], merged["structured_score_recomputed"]
    )
    assert np.allclose(
        merged["communication_score_released"], merged["communication_score_recomputed"]
    )

    for axis, filename in [
        ("system_prompt", "summary_model_system_prompt_trajectory.csv"),
        ("patient_attitude", "summary_model_patient_attitude_trajectory.csv"),
    ]:
        released = pd.read_csv(final_root / filename).sort_values(
            ["model_label", axis]
        ).reset_index(drop=True)
        recomputed = (
            released_levels.groupby(["model_label", axis], as_index=False)[
                ["structured_score", "communication_score"]
            ]
            .mean()
            .sort_values(["model_label", axis])
            .reset_index(drop=True)
        )
        assert np.allclose(released["structured_score"], recomputed["structured_score"])
        assert np.allclose(released["communication_score"], recomputed["communication_score"])
        assert released["n"].eq(320).all()

    figure_root = ROOT / "results" / "figure_source_data"
    figure_4 = pd.read_csv(figure_root / "figure_4_model_profiles_source_data.csv")
    figure_levels = figure_4[figure_4["panel"].eq("a_b_response_level")]
    figure_merged = figure_levels.merge(
        released_levels,
        on=keys,
        suffixes=("_figure", "_released"),
        validate="one_to_one",
    )
    assert len(figure_merged) == 10_240
    assert np.allclose(
        figure_merged["structured_score_figure"],
        figure_merged["structured_score_released"],
    )
    assert np.allclose(
        figure_merged["communication_score_figure"],
        figure_merged["communication_score_released"],
    )

    within_source = pd.read_csv(
        figure_root / "supplementary_figure_s3_within_model_effects_source_data.csv"
    ).sort_values(["comparison_axis", "evaluated_model", "metric"]).reset_index(drop=True)
    within_tables = pd.concat(
        [
            pd.read_csv(
                ROOT
                / "results"
                / "structured_metrics"
                / "final_structured_within_friedman.csv"
            ),
            pd.read_csv(
                ROOT
                / "results"
                / "llm_judge_metrics"
                / "final_communication_within_cochran_q.csv"
            ),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["comparison_axis", "evaluated_model", "metric"]).reset_index(drop=True)
    assert len(within_source) == len(within_tables) == 160
    for column in ["n_units", "p_value", "p_value_holm"]:
        assert np.allclose(
            within_source[column], within_tables[column], equal_nan=True
        )


def test_public_release_excludes_private_and_raw_outputs() -> None:
    forbidden_name_parts = {
        "answer_key",
        "manual_validation_response_level",
        "private",
        "raw_output",
        "responses_long",
    }
    relative_paths = [
        path.relative_to(ROOT).as_posix().lower() for path in ROOT.rglob("*")
    ]
    assert not [
        path
        for path in relative_paths
        if any(part in path for part in forbidden_name_parts)
    ]

    text_extensions = {".csv", ".json", ".md", ".py", ".txt", ".yaml"}
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in text_extensions
            and path.resolve() != Path(__file__).resolve()
        ):
            content = path.read_text(encoding="utf-8", errors="ignore")
            assert "C:\\Users\\johan" not in content
            assert "OPENAI_API_KEY=" not in content
            assert "DEEPSEEK_API_KEY=" not in content


def test_file_manifest_is_complete_and_current() -> None:
    manifest_path = ROOT / "file_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["relative_path"]: row for row in csv.DictReader(handle)}

    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(manifest) == expected_paths

    for relative_path, row in manifest.items():
        content = manifest_payload(ROOT / relative_path)
        assert int(row["bytes"]) == len(content)
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
