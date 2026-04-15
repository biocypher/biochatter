# BioCypher Worked Example

This is an example application of the generic Croissant benchmark migration
workflow to one benchmark family in this repository.

## Recommended Slice

Start with the `biocypher_query_generation` benchmark family and keep the first
scope intentionally small.

- KG schema: `gene_kg`
- Cases: `simple`, `single_word`, `complex`
- Preserve as semantic subtasks:
  - `entity_selection`
  - `relationship_selection`
  - `property_selection`
  - `query_generation`

This subset is preferable because it already behaves like a task graph with
shared inputs and structured intermediate outputs. The relevant source files are:

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
- The benchmark naturally decomposes into semantic stages, which matches
  `croissant:subTask`.

## Suggested Croissant Framing

Use one top-level `croissant:TaskProblem` for the benchmark slice:

- name: something like `BioCypher Query Generation Benchmark Slice`
- input:
  - benchmark cases dataset
  - KG schema dataset
- output:
  - a query artifact, or per-stage structured outputs for the subtasks
- evaluation:
  - expected metrics describing stage accuracy and query-pattern coverage
- subTask:
  - one task problem per preserved stage

## Mapping Table

| BioChatter concept | Croissant representation | Notes |
| --- | --- | --- |
| `biocypher_query_generation` family | top-level `croissant:TaskProblem` | Treat this as the benchmark slice to reproduce conceptually. |
| YAML case list in `benchmark_query_test_data.yaml` | `croissant:input` dataset | Each record should expose case id, prompt, schema id, and gold structures. |
| `gene_kg` schema in `benchmark_kg_schema_data.yaml` | `croissant:input` dataset | Keep as a referenced schema/context artifact, not as code. |
| entity selection stage | `croissant:subTask` | Output schema can be a repeated string field such as `entities`. |
| relationship selection stage | `croissant:subTask` | Output schema can expose relationship labels plus source and target roles. |
| property selection stage | `croissant:subTask` | Output schema can expose entity or relationship property selections. |
| query generation stage | `croissant:subTask` | Output schema can expose `query_text`. |
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
  - This is useful for comparison but less central to the semantic pipeline.
- `property_exists`
  - This is closer to a hallucination-check metric than a core task stage.
- `end_to_end_query_generation`
  - Valuable later, but it bundles the whole pipeline and reduces visibility
    into where reimplementations agree or differ.

## Conceptual Reproducibility Goal

The target is not byte-for-byte equivalence with the BioChatter implementation.
The target is that an independent team can read the Croissant task description,
recreate the benchmark slice, and obtain the same kind of stage-level outcomes
and aggregate conclusions on the same cases.
