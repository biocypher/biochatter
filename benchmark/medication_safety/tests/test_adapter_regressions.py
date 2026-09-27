"""Offline regression checks for new BioChatter runs."""

# ruff: noqa: INP001, PLR2004
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark.medication_safety import biochatter_provider as provider
from benchmark.medication_safety.scripts import analyze_final_scores as analysis
from benchmark.medication_safety.scripts import judge_with_biochatter as judge


@pytest.mark.parametrize("field", ["responses", "protocol", "settings", "model", "base_url"])
def test_resume_rejects_changed_identity(tmp_path, field):
    path = tmp_path / "judgements.csv"
    identity = dict.fromkeys(["responses", "protocol", "settings", "model", "base_url"], "original")
    judge.validate_resume(path, identity)
    judge.validate_resume(path, identity)
    with pytest.raises(ValueError, match="identity changed"):
        judge.validate_resume(path, {**identity, field: "changed"})
    assert "original" not in path.with_suffix(".csv.identity.json").read_text()


def test_resume_rejects_untracked_old_output(tmp_path):
    path = tmp_path / "judgements.csv"
    path.write_text("old output")
    with pytest.raises(ValueError, match="lack resume provenance"):
        judge.validate_resume(path, {})


@pytest.mark.parametrize("ids", [[1, 1], [1, 3]])
def test_judge_rejects_duplicate_or_wrong_iterations(tmp_path, ids):
    path = tmp_path / "judgements.csv"
    row = dict.fromkeys(judge.JUDGEMENT_COLUMNS, "value")
    row["criterion_label"] = 1
    pd.DataFrame([{**row, "judge_iteration": iteration} for iteration in ids]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="iteration identifiers"):
        judge.summarize_judgements(path, tmp_path / "scores.csv")


def test_fresh_analysis_output_directory(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    pd.DataFrame({"metric_group": ["structured", "judge"]}).to_csv(path, index=False)
    for name in [
        "analyze_structured",
        "analyze_communication",
        "analyze_structured_within",
        "analyze_communication_within",
    ]:
        monkeypatch.setattr(analysis, name, lambda _frame: pd.DataFrame({"example": [1]}))
    output = tmp_path / "new_output"
    monkeypatch.setattr(sys, "argv", ["analyze", "--scores", str(path), "--output-root", str(output)])
    analysis.main()
    assert (output / "final_analysis/pairwise_report.json").exists()


def test_provider_binds_explicit_settings_without_network(monkeypatch):
    captured = {}

    class FakeChat:
        def generate(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def invoke(self, **kwargs: object) -> None:
            captured.update(kwargs)

    conversation = SimpleNamespace(chat=FakeChat(), set_api_key=lambda *_args, **_kwargs: True)
    monkeypatch.setattr(provider, "GptConversation", lambda *_args, **_kwargs: conversation)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-placeholder")
    provider.create_conversation("deepseek", "test-model", max_tokens=4096, reasoning_effort="high", thinking="enabled")
    conversation.chat.generate()
    assert captured == {
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert "offline-test-placeholder" not in str(captured)
    expected = captured.copy()
    captured.clear()
    conversation.chat.invoke()
    assert captured == expected
