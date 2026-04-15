# BioChatter Fit Assessment

This note assesses which BioChatter benchmark families fit the current
Croissant task model cleanly, which ones only fit partially, and which ones
surpass the present task spec unless substantial information is kept outside the
JSON-LD. It is an example of repository-specific fit analysis, not part of the
generic skill itself.

## Benchmark-Family Fit

| Benchmark family | Fit | Rationale | Main Croissant boundary |
| --- | --- | --- | --- |
| `biocypher_query_generation` | strong | Structured cases, explicit semantic stages, schema-as-input, mostly transparent evaluation | Detailed regex scoring remains external |
| `medical_exam` | moderate to strong | Fixed QA cases with explicit expected answers | Exact-match and regex scoring details are not first-class in the task spec |
| `rag_interpretation` | moderate | Binary relevance tasks are simple and bounded | Implicit cases use another evaluator model, which moves key semantics outside the task description |
| `text_extraction` | moderate | Clear input captions and output intents | Heterogeneous extraction requests plus ROUGE-based scoring exceed what `EvaluationSpec` can express cleanly |
| `api_calling` | moderate | Inputs and outputs are structured enough to describe as code or query generation tasks | API-specific query fragments and builder semantics remain companion protocol, not pure Croissant structure |
| `text_image_multimodality` | moderate | Multimodal input and binary output are conceptually clear | Random sampling and confidence bookkeeping are harness-level behavior |
| `mcp_edam_qa` | weak | The benchmark objective is understandable | Tool schemas, native tool-calling requirements, traces, and execution behavior are central but only partially representable |
| `longevity_judge_responses_simultan` | weak | High-level response-and-judgment task can be described | The actual benchmark is a multi-stage, judge-heavy pipeline with stored responses and secondary evaluation loops |
| `vectorstore_semantic_search` | weak | Retrieval is a recognizable task class | External vector DB state, chunking, embedding models, and environment setup dominate the benchmark identity |

## Families That Best Match A Croissant First Pass

These are the best places to start:

1. `biocypher_query_generation`
2. a reduced `medical_exam` subset
3. explicit `rag_interpretation` yes or no tasks

These are usually second-pass candidates:

1. `text_extraction`
2. `api_calling`
3. `text_image_multimodality`

These are the clearest examples of current limits:

1. `mcp_edam_qa`
2. `longevity_judge_responses_simultan`
3. `vectorstore_semantic_search`

## Concrete Limits Surfaced By BioChatter

### 1. Evaluation semantics are richer than `EvaluationSpec`

Croissant can declare expected metrics, but BioChatter often depends on
evaluation rules that are more specific than a metric name:

- exact string equality
- regex coverage over generated queries
- boolean-vector scoring with partial credit
- ROUGE-based comparison
- secondary LLM judgment

These can be documented, but not fully normalized into the current task spec.

### 2. Prompt matrices and case expansion are benchmark harness behavior

BioChatter expands prompt variants and multi-input dictionaries into a larger
test matrix in `benchmark/load_dataset.py`. Croissant can describe the resulting
task data, but it does not currently offer a first-class declarative language
for benchmark-case expansion logic.

### 3. Tool-calling benchmarks need more than inputs and outputs

The MCP EDAM benchmark depends on:

- tool availability
- native tool-calling capability
- tool schemas
- traces of calls and results
- failure modes tied to tool execution

These are not impossible to mention in prose, but they are not cleanly captured
as the central benchmark semantics in current Croissant task objects.

### 4. Multi-stage judge pipelines exceed a clean task contract

The longevity benchmark is not just a single task. It is a pipeline:

- generate responses
- persist them
- load them later
- run a separate judge model
- repeat judgments
- aggregate results by metric

A Croissant description can summarize this at a high level, but much of the
actual benchmark meaning would still live in external protocol text.

### 5. Execution environment can be central, not incidental

Some benchmark families rely on environment-specific infrastructure such as:

- external APIs
- MCP servers
- vector databases
- multimodal model support
- specific tool-calling capabilities

Croissant provides places to mention implementation and environment, but not a
rich execution-contract language for all of these dependencies.

## Practical Rule For The Paper

If independent reimplementers could reconstruct the benchmark from a Croissant
task plus a short evaluation protocol, the benchmark is a good candidate for
Croissant migration.

If independent reimplementers would still need substantial hidden execution
logic, judge prompts, infrastructure behavior, or trace semantics to understand
what the benchmark really is, then the benchmark currently exceeds the sweet
spot of the Croissant task spec.
