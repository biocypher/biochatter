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
4/5 items for understandability, 5/5 for usefulness, and 4/4 applicable items
for patient attitude responsiveness. Each response was judged twice.

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
- `results/judge_sensitivity/`: alternative-judge results for the common
  5,120-response corpus from the four original response models.
- `results/figure_source_data/`: final source tables used for manuscript
  figures.
- `results/term_matching/`: normalization replacements and canonical synonym
  groups.
- `environment/python_pip_freeze.txt`: analysis environment snapshot.

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
  --model deepseek-v4-flash
```

The judge runs twice per response and criterion by default. It records the five
subcriteria locally, then derives the descriptive response score and the strict
binary label. A strict label is positive only when both judge iterations are
positive. Local raw response and judgement files are ignored by Git.

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
```

## Reproducibility scope

The released data support audit of the benchmark definitions, prompt design,
term matching rules, derived scores, statistical summaries, and figure source
data. Repeating provider inference requires access to the listed models. Full
recomputation from generated text is not possible from this public package
because raw model responses and raw judge outputs are intentionally excluded.
The public scripts reproduce the benchmark protocol through BioChatter, but an
exact repeat of historical inference also depends on continued access to the
listed provider snapshots and provider-specific decoding options.

The YAML retains the technical prompt key `role_attitude_sensitive`; manuscript
text refers to this condition as the patient attitude sensitive prompt.
