# Exploratory None-to-Minimal comparison (Supplementary Table 5)

`supplementary_table_5a.csv` is reproducible with
`python -m benchmark.medication_safety.scripts.analyze_none_minimal`.
It uses 320 matched response pairs per model, matched on medication-indication
case, patient attitude and generation round. Differences are Minimal minus None.
The two aggregate outcomes are the mean structured score over applicable metrics
and the communication score averaged across judge runs and criteria.

Two-sided Wilcoxon signed-rank tests use the 20 case-mean differences, with Holm
adjustment across eight response models separately for each outcome. Pointwise
95% percentile bootstrap intervals use 50,000 resamples of whole cases (seed
20260920); they are not multiplicity-adjusted. These tests concern signed-rank
differences, not specifically the arithmetic mean. This is a post hoc analysis.

`supplementary_table_5bc_validation.csv` contains descriptive aggregate results
for manual and primary DeepSeek scores on the same 512 validation responses.
Each prompt contributes 16 responses per model (four cases by four attitudes).
Generation rounds need not match across prompts. No validation-subset CIs or
p-values are released here; the sample contains only four cases. Individual
manual labels are not included.
