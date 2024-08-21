# Benchmark Results - Overview

Here we collect the results of the living BioChatter benchmark. For an
explanation, see the [benchmarking documentation](../features/benchmark.md) and the
[developer docs](developer.md) for further reading.

## Scores per model

Table sorted by median score in descending order.
Click the column names to reorder.

{{ read_csv('benchmark/results/processed/overview-model.csv', colalign=("left","right")) }}

![Scatter Quantisation Name](../images/scatter-per-quantisation-name.png)
![Boxplot Model](../images/stripplot-per-model.png)

## Scores per quantisation

Table sorted by median score in descending order.
Click the column names to reorder.

{{ read_csv('benchmark/results/processed/overview-quantisation.csv', colalign=("left","right")) }}

![Boxplot Quantisation](../images/boxplot-per-quantisation.png)

## Scores of all tasks

Wide table; you may need to scroll horizontally to see all columns.
Table sorted by median score in descending order.
Click the column names to reorder.

{{ read_csv('benchmark/results/processed/overview.csv', colalign=("left","right")) }}
