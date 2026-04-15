# BioCypher Worked Example

This is an example application of the generic Croissant benchmark migration
workflow to one benchmark family in this repository.

## Recommended Slice

Start with the `biocypher_query_generation` benchmark family and keep the first
scope intentionally small.

- KG schema: `gene_kg`
- Cases: `simple`, `single_word`, `multi_word`, `complex`
- Preserve as independent benchmark targets:
  - `query_generation`
  - `entity_selection`
  - `relationship_selection`
  - `property_selection`

This subset is preferable because it already supports multiple benchmark targets
over shared inputs and structured outputs. The relevant source files are:

- `benchmark/data/benchmark_query_test_data.yaml`
- `benchmark/data/benchmark_kg_schema_data.yaml`
- `benchmark/test_biocypher_query_generation.py`
- `biochatter/prompts.py`

## Why This Is A Strong First Migration

- The benchmark cases are fixed and explicit.
- The KG schema can be treated as a named input artifact.
- Intermediate outputs are structured enough to describe at a high level.
- The task can be reimplemented independently without reproducing the BioChatter
  internals line by line.
- The benchmark supports multiple conceptually meaningful targets over the same
  cases, which can be represented as `croissant:subTask`s without prescribing a
  required internal pipeline.
- The top-level object is better interpreted as a benchmark suite than as one
  `TaskProblem` with one output contract.

## Suggested Croissant Framing

Use one top-level Croissant task or benchmark-suite object for the benchmark
slice:

- name: something like `BioCypher Query Generation Benchmark Slice`
- input:
  - benchmark cases dataset
  - KG schema dataset
- implementation:
  - reference implementation artifacts, if useful
- subTask:
  - one task problem per preserved benchmark target

Avoid:

- making the top-level object a `TaskProblem` whose evaluation mixes several
  different target types without one shared output contract
- requiring a fixed internal planning representation for query generation

## Mapping Table

| BioChatter concept | Croissant representation | Notes |
| --- | --- | --- |
| `biocypher_query_generation` family | top-level Croissant task or benchmark suite | Treat this as the benchmark slice to reproduce conceptually. |
| YAML case list in `benchmark_query_test_data.yaml` | `croissant:input` dataset | Each record should expose case id, prompt, schema id, and gold structures. |
| `gene_kg` schema in `benchmark_kg_schema_data.yaml` | `croissant:input` dataset | Keep as a referenced schema/context artifact, not as code. |
| entity selection target | `croissant:subTask` | Output schema can emit one record per selected entity. |
| relationship selection target | `croissant:subTask` | Output schema should emit one record per relationship binding. |
| property selection target | `croissant:subTask` | Output schema should emit one record per selected property binding. |
| query generation target | `croissant:subTask` | Output schema can expose `query_text`. |
| BioChatter implementation files | `croissant:implementation` | Useful as reference implementations, but not normative task structure. |
| expected entities, relationships, properties | benchmark input fields or gold labels in companion dataset | Keep these available for evaluation but do not confuse them with model outputs. |
| regex checks in `parts_of_query` | companion evaluation protocol | Croissant can name the metric, but the regex semantics remain external. |

## Evaluation Guidance

At the Croissant level, declare only the high-level metrics, for example:

- entity selection accuracy
- relationship selection accuracy
- property selection accuracy
- query-pattern coverage

Keep these details outside the JSON-LD:

- the exact boolean-vector scoring logic
- regex rules for query validation
- how scores are aggregated into `score/max`
- result-file formats and hash-based skipping

## What To Defer

These pieces are still related, but they are less suitable for the first
Croissant proof of concept:

- `naive_query_generation_using_schema`
  - This is useful for comparison but less central to the main benchmark
    targets in the first suite draft.
- `property_exists`
  - This is closer to a hallucination-check metric than a core task stage.
- normative intermediate planning inputs for query generation
  - Avoid these in the Croissant contract unless the benchmark explicitly wants
    to evaluate a fixed planning representation.

## Conceptual Reproducibility Goal

The target is not byte-for-byte equivalence with the BioChatter implementation.
The target is that an independent team can read the Croissant task description,
recreate the benchmark slice, and obtain the same kind of stage-level outcomes
and aggregate conclusions on the same cases.
