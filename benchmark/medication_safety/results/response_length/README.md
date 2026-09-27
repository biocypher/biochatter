# Response length (Supplementary Table 6)

The descriptive table contains unrounded means and medians independently
recomputed from the final 10,240-response manifest. Word counts use matches of
the regular expression `\b[\w'-]+\b`. None uses 320 responses per model;
the structured mean uses all 960 responses from the other three prompts.
The structured median is the median of the 320 per-unit averages across the
three structured prompts, matched by case, attitude and generation round.

`score_length_correlations.csv` contains Spearman correlations for the full
corpus and the 512-response manual sample. Raw response text is excluded, so
these length aggregates cannot be recomputed from this release alone. These
files report descriptive statistics, not independent-response inferential CIs.
