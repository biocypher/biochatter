# API Agent Reference

Here we handle the connection to external software tools via the
parameterisation of API calls by the LLM.

## Base classes

### The abstract base classes

::: biochatter.api_agent.base.agent_abc

### The API Agent

::: biochatter.api_agent.base.api_agent

## Web APIs

### The BLAST tool

::: biochatter.api_agent.web.blast

### The OncoKB tool

::: biochatter.api_agent.web.oncokb

### The bio.tools API

::: biochatter.api_agent.web.bio_tools

## Python APIs

### Generic Python API ingestion

::: biochatter.api_agent.python.generic_agent

::: biochatter.api_agent.python.autogenerate_model

### Scanpy modules

::: biochatter.api_agent.python.anndata_agent

::: biochatter.api_agent.python.scanpy_pl_full

::: biochatter.api_agent.python.scanpy_pp_full

::: biochatter.api_agent.python.scanpy_pl_reduced

::: biochatter.api_agent.python.scanpy_pp_reduced

## Utility functions

### Formatters to parse the calls

::: biochatter.api_agent.base.formatters

