# Medication safety benchmark

This folder contains the public release of a benchmark for large language model
responses about adverse reactions and contraindications. It includes 20
synthetic adult medication indication cases based on European Medicines Agency
Product Information.

Each case was combined with four system prompts and four patient attitude
conditions. Eight response models were evaluated across four repeated response
iterations, yielding 10,240 responses. GPT-5.4 was rerun at temperature 0 for
the final analysis so that all response models used temperature 0.

The final primary communication analysis used DeepSeek V4 Flash with explicit
subcriteria. The judge returned five item-level decisions for each criterion.
Binary labels were derived with the same thresholds as the manual assessment:
at least 4/5 items for understandability, 5/5 for usefulness, and 4/4 applicable items
for patient attitude responsiveness. Each response was judged twice per criterion,
yielding 61,440 criterion-level ratings. Descriptive criterion scores average the
two binary run labels (0, 0.5 or 1). Positive rates, agreement and binary-outcome
tests use a strict label, positive only when both runs are positive. The additional
None-to-Minimal aggregate-score test uses averaged scores, not strict labels.

## Contents

- `data/benchmark_medication_safety_data.yaml`: benchmark cases, EMA source
  metadata, prompts, patient attitude statements, curated adverse reactions,
  and contraindications.
- `prompts/response_generation_prompts.yaml`: response generation prompts.
- `prompts/llm_judge_prompts.yaml`: final subcriteria-based primary judge
  protocol.
- `prompts/llm_judge_prompts_legacy_binary.yaml`: direct binary protocol used
  for the alternative-judge sensitivity analysis.
- `model_settings/`: response-model and judge settings without credentials.
- `scripts/`: benchmark construction, term matching, scoring, trade-off, and
  inferential analysis utilities.
- `results/final_analysis/`: final response-level scores, aggregate trajectory
  data, and validation reports for all eight models.
- `results/structured_metrics/`: final structured metric summaries and tests.
- `results/llm_judge_metrics/`: final DeepSeek communication summaries and
  tests.
- `results/manual_validation/`: aggregate summaries for the 512-response
  blinded manual validation subset.
- `results/temperature_sensitivity/`: aggregate GPT-5.4 temperature 0 versus
  temperature 1 sensitivity results.
- `results/judge_sensitivity/`: the final five-judge, three-response-model
  analysis (3,840 responses, 107,520 criterion-level ratings), plus explicitly
  documented older sensitivity summaries that are not manuscript results.
- `results/none_minimal/`: Supplementary Table 5 aggregate-score contrasts.
- `results/figure_source_data/`: final source tables used for manuscript
  figures.
- `results/term_matching/`: normalization replacements and canonical synonym
  groups.
- `environment/python_pip_freeze.txt`: analysis environment snapshot.
- `figures/`: the manuscript figure images, including S7 with black prompt labels.
- `MANUSCRIPT_MAP.md`: current figure/table numbering, definitions and exclusions.

## Not included

The public release excludes full generated responses, raw judge outputs,
row-level manual labels, manual free-text notes, provider logs, API keys, and
local runtime configuration. Released response-level files contain only
benchmark identifiers and derived numeric scores.

## Basic usage

Run the focused validation tests from the repository root:

```bash
python -m pytest benchmark/medication_safety/tests \
  --confcutdir=benchmark/medication_safety
```

Build the 320 benchmark instances:

```bash
python -m benchmark.medication_safety.scripts.build_benchmark_instances
```

Run the expanded instances through the model matrix configured by BioChatter's
benchmark fixtures:

```bash
pytest benchmark/test_medication_safety.py
```

The native test uses four response iterations by default. Set
`BIOCHATTER_MEDICATION_SAFETY_ITERATIONS` to use another value. Responses are
written with BioChatter's standard response schema under `benchmark/results/`.
These local raw-response files are ignored by Git.

For a model that is not part of the central benchmark matrix, use a
BioChatter provider directly. This example runs one OpenRouter instance as a
smoke test before starting the complete 320-instance matrix:

```bash
python -m benchmark.medication_safety.scripts.run_with_biochatter \
  --provider openrouter \
  --model google/gemini-3.5-flash \
  --limit 1
```

The runner reads credentials from the provider's environment variable and
does not accept API keys as command-line values. Supported providers are
OpenAI, Anthropic, Gemini, OpenRouter, and OpenAI-compatible endpoints.

Apply the published subcriterion judge protocol to a standard BioChatter
response file. This direct DeepSeek example reads `DEEPSEEK_API_KEY` from the
environment:

```bash
python -m benchmark.medication_safety.scripts.judge_with_biochatter \
  benchmark/results/medication_safety_response_generation_MODEL_response.csv \
  --provider deepseek \
  --model deepseek-v4-flash \
  --temperature 0 --max-tokens 4096 --thinking enabled --reasoning-effort high
```

The judge runs twice per response and criterion by default. It records the five
subcriteria locally, then derives the descriptive response score and the strict
binary label. A strict label is positive only when both judge iterations are
positive. Local raw response and judgement files are ignored by Git.

Resume validates a SHA-256 identity of the complete selected response input,
benchmark instances, rubric, provider, endpoint and requested run settings.
Changing any of these requires a new output path. Old judgement CSVs without
the identity sidecar are rejected rather than silently reused. The sidecar
contains only a digest, not prompts, responses or credentials. Duplicate or
unexpected judge-round identifiers are rejected during summarization.

Both standalone runners accept explicit `--temperature`, `--max-tokens`,
`--reasoning-effort` and `--thinking` options. Temperature defaults to zero;
unset optional values retain provider defaults. Thinking/reasoning overrides
are limited to OpenAI-compatible providers. Use the documented model-specific
settings rather than assuming one configuration fits all models. The native
pytest model matrix still uses the central BioChatter fixture configuration.

These runners support new executions, not an exact historical replay: they do
not automatically import settings CSVs, reproduce historical context formatting,
or implement the historical 8192-token rescue procedure. The DeepSeek example
explicitly sets the primary 4096-token configuration, but live provider support
must be checked separately. Offline tests do not establish API compatibility.

Score either a simple CSV containing `case_id` and `response` columns or a
standard BioChatter response file:

```bash
python -m benchmark.medication_safety.scripts.score_responses responses.csv \
  --output scored_responses.csv
```

Reproduce the final between-model and within-model tests and system prompt
trade-off analysis:

```bash
python -m benchmark.medication_safety.scripts.analyze_final_scores
python -m benchmark.medication_safety.scripts.analyze_system_prompt_tradeoff
python -m benchmark.medication_safety.scripts.analyze_none_minimal
```

## Reproducibility scope

### Optional live integration check

This command makes paid API calls with your own DeepSeek credential from the
environment. It is never invoked by CI. Use a fresh output directory:

```bash
python -m benchmark.medication_safety.scripts.live_smoke \
  --output-dir benchmark/results/medication_safety_smoke
```

It exercises one None-prompt/neutral-attitude instance, four response generations,
structured scoring, two judge runs for each of three criteria (24 judgements), and resume
without additional inference. The response model uses thinking disabled; the
judge uses thinking enabled/high reasoning, both at temperature 0. Token limits
are 8192 for generation and 4096 for judging. This is an execution check, not a
new scientific validation or a replay of the historical eight-model study.
Raw outputs stay in the ignored directory.
The generated report contains counts/settings but no credentials or response text.
Individual raw completions are saved locally before validation so a failed check
does not discard its diagnostic evidence. A length-limited response fails the test.
Use `--system-prompt minimal` to test the longer structured-output condition.
`--generation-max-tokens` changes only the smoke-test response budget; it does
not change the judge budget or any published study settings.

### Historical reproducibility

The released data support audit of the benchmark definitions, prompt design,
term matching rules, derived scores, statistical summaries, and figure source
data. Repeating provider inference requires access to the listed models. Full
recomputation from generated text is not possible from this public package
because raw model responses and raw judge outputs are intentionally excluded.
The public scripts reproduce the benchmark protocol through BioChatter, but an
exact repeat of historical inference also depends on continued access to the
listed provider snapshots and provider-specific decoding options.

The YAML retains the technical prompt key `role_attitude_sensitive`; manuscript
text refers to this condition as the Patient-attitude sensitive prompt. The other
display labels are None, Minimal and Role-encouraging. Technical keys are retained
unchanged for compatibility.
