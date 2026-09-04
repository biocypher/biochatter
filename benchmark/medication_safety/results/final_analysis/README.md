# Final primary analysis

These files contain the final eight-model analysis. GPT-5.4 responses were
generated at temperature 0. Communication scores were derived from two
subcriteria-based DeepSeek V4 Flash judgements per response.

- `final_response_metric_scores.csv` contains one derived score per response
  and metric. It contains no response text or raw judge output.
- `final_response_level_scores.csv` contains the unweighted mean of the seven
  structured metrics and the unweighted mean of the three communication
  metrics for each response.
- `summary_model_*` files contain aggregate model and condition means.
- `validation_report.json` records corpus-size and uniqueness checks.
- `pairwise_report.json` records the final between-model test families.
- `primary_judge_run_summary.json` records the final judge run settings and
  completion counts without provider credentials or raw outputs.
