# Task vs Protocol Boundary

This note explains a recurring migration decision: whether a benchmark feature
belongs in the Croissant task object itself or in companion protocol text.

## Core Principle

Croissant task objects are strongest at describing a task contract:

- what the task is
- what goes in
- what comes out
- what benchmark targets or conceptual subproblems matter
- what high-level metrics are expected

They are weaker as a declarative language for benchmark procedure:

- orchestration logic
- repeated sampling protocols
- judge-model prompts
- execution matrices
- caching and skip behavior
- stateful infrastructure setup

This does not mean those features cannot be mentioned. It means they often
cannot be captured precisely enough to carry the whole benchmark meaning inside
the JSON-LD alone.

## Practical Test

Ask this question:

Would an independent team understand the benchmark identity from the Croissant
task object alone, or would they still need substantial procedural details?

If they still need substantial procedural details, the missing parts belong in a
short companion protocol note.

## Common Cases

### Benchmark targets vs implementation stages

This is the most important distinction to get right.

Sometimes a benchmark exposes several real targets over the same inputs, for
example:

- final answer generation
- label selection
- retrieval ranking
- attribute extraction

If each of these is a legitimate thing that the benchmark evaluates in its own
right, each can be modeled as its own `croissant:TaskProblem`, often grouped
under a top-level `croissant:Task` suite.

By contrast, if a system happens to use internal steps such as:

- plan generation
- candidate pruning
- intermediate serialization
- prompt chaining

those steps should usually not be modeled as Croissant subtasks unless the
benchmark itself treats them as benchmark targets.

Migration strategy:

- preserve benchmark targets
- omit implementation-only pipeline steps
- use `croissant:implementation` for reference code when helpful

### LLM-as-judge evaluation

Croissant can express that evaluation exists and can name expected metrics. It
does not provide a rich formal language for:

- judge model identity
- judge prompt wording
- number of judging rounds
- agreement or aggregation rules

Migration strategy:

- keep the benchmark task in Croissant
- document the judge protocol separately
- if possible, freeze judged labels for a first migration

### Tool-call traces

Croissant can mention execution or trace artifacts at a high level. It does not
make tool protocol semantics first-class.

Migration strategy:

- keep task inputs and outputs in Croissant
- treat trace semantics as companion protocol unless the traces are merely
  supplementary artifacts

### Random sampling

Croissant can record execution metadata such as a seed or configuration note,
but it does not define a rich sampling-procedure language.

Examples of details that usually remain external:

- sampling universe
- replacement policy
- stratification
- number of repeated draws
- pre- and post-filtering order
- score aggregation across samples

Migration strategy:

- best: freeze the sampled subset as an explicit input dataset
- acceptable: document the sampling protocol in a short note

### External infrastructure state

Croissant can point to implementations and environments. It does not fully
capture stateful runtime conditions such as:

- specific server state
- pre-populated vector stores
- active tool registries
- remote service versions with hidden state

Migration strategy:

- treat infrastructure as environment or implementation context
- freeze benchmark-relevant state where possible
- avoid making hidden state part of the task identity

### Prompt matrices and execution matrices

Croissant can represent the realized tasks or datasets. It does not provide a
first-class declarative language for generating benchmark matrices from prompt
variants, model variants, quantizations, and iteration schedules.

Migration strategy:

- materialize the realized cases into an explicit dataset, or
- document the expansion logic separately

### Pipeline bookkeeping

Many benchmarks contain extra orchestration:

- generate intermediate outputs
- persist them
- reload them later
- run downstream evaluation
- cache by hash
- skip completed runs

Migration strategy:

- keep the semantic task and semantic subtasks in Croissant
- keep bookkeeping in the harness or protocol description

## Rule Of Thumb

Put something in the Croissant task object when it helps define the task that an
independent implementation should reproduce.

Put something in companion protocol text when it mainly explains how one
particular benchmark harness executes, samples, caches, or judges the task.

Put something in `croissant:subTask` when it is itself a benchmark target or a
conceptual subproblem worth reproducing, not just a step that happened to exist
inside one implementation.
