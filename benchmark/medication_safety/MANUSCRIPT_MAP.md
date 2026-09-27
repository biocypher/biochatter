# Manuscript snapshot: 24 September 2026

This snapshot aligns the released derived results with the current manuscript
and Supplement. It is not a claim that editorial review or peer review is complete.
No new provider inference was performed for this update. The exploratory newer
512-response judge run without the generation system prompt is not included in
the primary analysis or in these manuscript figures.

## Figures

The PNGs in `figures/` are the manuscript-embedded images checked on 23 September,
with S7 regenerated to make its prompt labels black. Data, trajectories and axes
are unchanged. S1/S2 each have two image parts. These are presentation exports,
not a claim that every original plotting script is packaged here.

| Figure | Source in results/figure_source_data |
| --- | --- |
| 1 | Workflow schematic; definitions in data/ and prompts/ |
| 2 | figure_2_system_prompt_trajectories_source_data.csv |
| 3 | figure_3_manual_validation_diagnostics_source_data.csv |
| 4 | figure_4_model_profiles_source_data.csv |
| S1 | supplementary_figure_s1_system_prompt_profiles_source_data.csv |
| S2 | supplementary_figure_s2_patient_attitude_profiles_source_data.csv |
| S3 | supplementary_figure_s3_within_model_effects_source_data.csv |
| S4 | supplementary_figure_s4_subcriteria_source_data.csv |
| S5 | supplementary_figure_s5_matched_validation_source_data.csv |
| S6 | supplementary_figure_s6_validation_source_data.csv and supplementary_figure_s6_full_corpus_source_data.csv |
| S7 | supplementary_figure_s7_alternative_judges_source_data.csv |

S3 contains the corrected 112 Friedman and 48 Cochran-Q tests. S4 uses the final
manual subcriteria and primary DeepSeek labels: responsiveness q2/q3 each have
256 applicable responses, all other subcriteria have 512. S5 uses averaged
criterion scores; S6 uses strict positive rates. Fig. 2/S5 share axis limits;
S7 deliberately retains its separate zoom and judge-model centering. Model colors
are consistent; manual/judge colors encode a different variable.

The file named `supplementary_figure_s10_patient_attitude_responsiveness_source_data.csv`
predates the final numbering and is not a current supplementary figure.

## Supplementary tables

Tables 1 and 3 describe the curated benchmark in `data/`; Table 2 is described by
`model_settings/response_model_settings.csv`; Table 4 describes the primary row
of `model_settings/llm_judge_settings.csv`. Table 3 counts term entries before
canonical merging, not necessarily the concept denominators used by the scorer.

Table 5 is in `results/none_minimal/`. Table 6 is response word counts; its source
text is intentionally not public, and its aggregate export is provided under
`results/response_length/`. Tables 7 and 8 correspond to the aggregate manual
validation summaries in `results/manual_validation/`. Table 9 is
`results/judge_sensitivity/supplementary_table_9.csv`.

## Definitions and limitations

The primary judge contributes two binary labels per response and criterion.
Mean scores use 0/0.5/1; positive rates, agreement and binary tests require two
positive runs. The None-to-Minimal aggregate-score contrast uses averaged scores.
Manual labels are binary. The structured composite is an unweighted mean over
applicable metrics; missing frequency categories are not treated as zero.

Precision counts distinct recognized concepts from the benchmark-wide vocabulary.
Unrecognized wording does not enter its denominator; repetitions do not create
extra distinct false positives. Full text-to-score reproduction requires the
excluded response text. Human validation used one reviewer and four cases, and
judge and human inputs differed; these results do not remove those limitations.

The older temperature-sensitivity and four-model judge-sensitivity results are
retained for provenance, not promoted to current manuscript claims. Unreported
exploratory analyses, raw responses, row-level manual ratings, comments and draft
manuscript PDFs are excluded.
