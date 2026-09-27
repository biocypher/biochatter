# Snapshot counts are explicit expectations, not analysis parameters.
# ruff: noqa: INP001, PLR2004
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.medication_safety.scripts.analyze_none_minimal import analyze

ROOT = Path(__file__).resolve().parents[1]


def test_none_minimal_table_reproduces_with_case_bootstrap():
    frame = pd.read_csv(ROOT / "results/final_analysis/final_response_level_scores.csv")
    actual = analyze(frame)
    saved = pd.read_csv(ROOT / "results/none_minimal/supplementary_table_5a.csv")
    pd.testing.assert_frame_equal(actual, saved, check_exact=False, atol=1e-12, rtol=1e-12)
    assert len(actual) == 16
    assert actual.p_holm.lt(0.05).all()
    assert actual[actual.outcome.eq("structured_score")].difference.gt(0).all()
    assert actual[actual.outcome.eq("communication_score")].difference.lt(0).all()
    gpt = actual[actual.model.eq("GPT-5.4") & actual.outcome.eq("communication_score")].iloc[0]
    assert gpt.difference == pytest.approx(-0.4864583333333333)


def test_alternative_judge_selection_counts_and_means():
    counts = pd.read_csv(ROOT / "results/judge_sensitivity/final_run_counts.csv")
    assert len(counts) == 15
    assert counts.criterion_level_ratings.sum() == 107520
    assert counts.responses.eq(1280).all()
    assert counts.included_runs.eq(1).sum() == 2
    data = pd.read_csv(ROOT / "results/figure_source_data/supplementary_figure_s7_alternative_judges_source_data.csv")
    assert len(data) == 180
    assert set(data.model_label) == {"Claude Sonnet 4.6", "GPT-OSS-120B", "Med42-8B"}
    expected = data.pivot_table(index="judge", columns="system_prompt", values="mean_response_score", aggfunc="mean")
    expected.loc["Mean across alternative judges"] = expected.mean(axis=0)
    saved = pd.read_csv(ROOT / "results/judge_sensitivity/supplementary_table_9.csv", index_col=0)
    np.testing.assert_allclose(saved, expected.reindex(index=saved.index, columns=saved.columns))
    assert ((saved["minimal"] - saved["none"]).iloc[:-1] < 0).sum() == 4


def test_final_subcriterion_denominators_and_snapshot_figures():
    data = pd.read_csv(ROOT / "results/figure_source_data/supplementary_figure_s4_subcriteria_source_data.csv")
    conditional = data.metric.eq("patient_attitude_responsiveness") & data.subcriterion.isin(["q2", "q3"])
    assert conditional.sum() == 2
    assert data.loc[conditional, "n_compared"].eq(256).all()
    assert data.loc[~conditional, "n_compared"].eq(512).all()
    assert len(list((ROOT / "figures").glob("*.png"))) == 13
    validation = pd.read_csv(ROOT / "results/none_minimal/supplementary_table_5bc_validation.csv")
    assert len(validation) == 16
    assert not any(column.startswith("p_") for column in validation.columns)
