---
name: croissant-benchmark-migration
description: Convert an existing benchmark or evaluation pipeline into a Croissant task description for conceptual reproducibility. Use when selecting a migration subset, mapping benchmark inputs, outputs, subtasks, and evaluation into Croissant TaskProblem or TaskSolution structures, and documenting where the benchmark exceeds the current Croissant task spec.
---

# Croissant Benchmark Migration

Use this skill when the goal is to express an existing benchmark in Croissant task
format at a high level, without making the JSON-LD itself executable. Favor
conceptual reproducibility over implementation lock-in.

## Outcome

Produce these artifacts or decisions:

1. A recommended benchmark subset to migrate first.
2. A source-to-Croissant mapping for inputs, outputs, subtasks, and metrics.
3. A short boundary note explaining what belongs in the Croissant JSON-LD and
   what must remain in companion evaluation or execution documentation.
4. A limitations note describing which parts of the benchmark exceed the
   current Croissant task model.

## Workflow

1. Read the benchmark code and data, not just published results.
2. Separate five layers before writing any Croissant object:
   - benchmark cases or datasets
   - semantic task stages
   - execution harness details
   - evaluation semantics
   - reporting artifacts such as traces, failure modes, or confidence files
3. Choose a first migration target that is stable and reimplementable:
   - fixed case set
   - structured inputs and outputs
   - explicit or deterministic scoring
   - little hidden runtime state
   - minimal dependence on external services or judge models
4. Map the chosen subset into Croissant:
   - top-level benchmark slice -> `croissant:TaskProblem`
   - semantic stages worth preserving -> `croissant:subTask`
   - required data and context -> `croissant:input`
   - expected output structure -> `croissant:output`
   - expected high-level metrics -> `croissant:evaluation`
5. Keep the JSON-LD high level:
   - describe the task contract, not the exact executor
   - preserve freedom for independent reimplementation
   - avoid encoding repo-specific code paths as if they were normative
6. Write an explicit boundary note for anything that cannot be captured cleanly
   in the current Croissant task spec.

## Selection Rubric

Strong first candidates usually have all or most of these properties:

- a small frozen subset of benchmark cases
- a clear task objective
- inputs that can be named as datasets or structured records
- outputs that can be described with a simple schema
- evaluation that can be summarized as one or a few metrics
- subtasks that are semantic stages rather than implementation accidents

The following are not disqualifiers, but they are signals that the migration
will need either a companion protocol note or a frozen materialization of the
benchmark procedure:

- another LLM judge to decide correctness
- tool-call traces as part of the benchmark meaning
- random sampling during evaluation
- external infrastructure state
- many prompt-variant expansions or execution matrices
- pipeline bookkeeping that is more elaborate than the underlying task

For a first migration, prefer one of these strategies:

1. Freeze the variable procedure into explicit benchmark inputs.
2. Move the procedure into a short companion protocol note.
3. Narrow the task slice until the Croissant object cleanly captures the task
   contract.

Read [references/task-vs-protocol-boundary.md](references/task-vs-protocol-boundary.md)
when deciding whether something belongs in the Croissant JSON-LD or in
supporting documentation.

## Croissant Mapping Guidance

- Use `croissant:TaskProblem` for the benchmark slice that another team should
  be able to reproduce conceptually.
- Use `croissant:subTask` only for stages that are meaningful to preserve in an
  independent reimplementation.
- Put datasets, schemas, fixed examples, and reference context in
  `croissant:input` when they are part of the task definition.
- Put the predicted object shape in `croissant:output`; do not confuse this
  with gold labels or implementation internals.
- Use `croissant:EvaluationSpec` to name expected metrics, but keep detailed
  scoring rules in companion documentation when they are more specific than the
  ontology can express.
- Treat `croissant:TaskSolution` as optional for the manuscript phase unless
  you are also documenting a concrete implementation run.

## Boundaries To State Explicitly

Always state these distinctions when writing the migration:

- The Croissant JSON-LD describes the task, not the runner.
- Execution matrices across models, prompts, quantizations, or iterations are
  benchmark harness details unless they are part of the task identity.
- LLM-as-judge procedures, regex scoring, ROUGE scoring, trace capture, and
  failure-mode taxonomies may need to remain outside the JSON-LD.
- Hash-based caching, skip logic, and result file layouts belong to the harness,
  not the Croissant task definition.

## Example References

The core skill is generic. Use repository-specific references only as worked
examples.

- Read [references/task-vs-protocol-boundary.md](references/task-vs-protocol-boundary.md)
  for generic guidance on what Croissant task objects describe well and what
  usually remains in companion protocol text.
- Read [references/biocypher-worked-example.md](references/biocypher-worked-example.md)
  only when you want a concrete BioChatter example of the workflow.
- Read [references/biochatter-fit-assessment.md](references/biochatter-fit-assessment.md)
  only when you want an example of benchmark-family fit analysis in one codebase.
