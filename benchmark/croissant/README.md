# Croissant Task Drafts

This directory contains draft Croissant task artifacts for expressing selected
BioChatter benchmark slices as conceptually reproducible tasks.

These files are intentionally high level. They are not meant to make the
JSON-LD executable by itself. Instead, they separate:

- generic migration guidance in the local `SKILL.md`
- task semantics captured in the Croissant problem description
- fixed input artifacts that an independent implementation can consume
- protocol details that may remain outside the JSON-LD

## Contents

- `SKILL.md`
  - generic guidance for migrating benchmarks into Croissant task format
- `references/`
  - worked examples and notes on the task-versus-protocol boundary

## Current Draft

- `biocypher-query-benchmark-suite.jsonld`
  - a Croissant task suite for a reduced BioCypher query benchmark slice
- `biocypher-query-subset.json`
  - the fixed benchmark cases used by the draft task
- `gene-kg-subset.json`
  - the reduced schema context required by those cases

## Scope Of The Current Draft

The first draft freezes:

- benchmark family: `biocypher_query_generation`
- KG schema slice: `gene_kg`
- cases: `simple`, `single_word`, `multi_word`, `complex`

The task description preserves four independent benchmark targets as subtasks:

- end-to-end query generation
- entity selection
- relationship selection
- property selection

Detailed regex-based query checking and score aggregation remain protocol-level
evaluation details rather than being fully encoded in the JSON-LD.
