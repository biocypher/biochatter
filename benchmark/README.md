# The BioChatter Benchmark Module

Here we collect procedures and data for the living BioChatter benchmark. The
benchmark follows a standard Pytest procedure (configured in `conftest.py`). We
create a matrix of test cases, tested models, and parameters, and run the tests
periodically. For more information, see the corresponding pages in the
BioChatter docs:

- Description: https://biochatter.org/benchmarking/

- Developer docs: https://biochatter.org/benchmark-developer/

- Results: https://biochatter.org/benchmark/

The medication safety extension under `benchmark/medication_safety/` adds 20
EMA-grounded cases that expand to 320 combinations of system prompts and
patient attitudes. It can be run through BioChatter with
`pytest benchmark/test_medication_safety.py`; focused usage and reproducibility
details are documented in the extension's README.
