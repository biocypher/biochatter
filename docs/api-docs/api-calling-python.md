# API Calling: Python APIs

API calling for large Python APIs is currently experimental. In particular, we
observe a decrease in stability with increasing number of total parameters
offered to the LLM. Due to this limitation, we recommend benchmarking the
stability of the calls using our benchmarking framework. If you're interested in
the performance of a specific API / LLM combination, don't hesitate to get in
touch.

## Generic Python API ingestion

Using Pydantic parsing, we autogenerate API descriptions for tool bindings.
While this allows better scaling (given suitable structure of the ingested code,
particularly with respect to the docstrings), it offers less control than the
manual implementation of API descriptions. For instance, it is much harder to
reduce the set of parameters to the essentials.

::: biochatter.api_agent.python.generic_agent

::: biochatter.api_agent.python.autogenerate_model

## Scanpy modules

We manually define the API descriptions for select Scanpy modules.

::: biochatter.api_agent.python.anndata_agent

::: biochatter.api_agent.python.scanpy_pl_full

::: biochatter.api_agent.python.scanpy_pp_full

::: biochatter.api_agent.python.scanpy_pl_reduced

::: biochatter.api_agent.python.scanpy_pp_reduced

# API Calling: Utility functions

## Formatters to parse the calls

::: biochatter.api_agent.base.formatters

