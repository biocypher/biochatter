"""

Load all CSVs from the benchmark/results folder that do not have either
"confidence" or "failure" in their name, and count the number of individual
tests that have been run by summing up the "iterations" column from all CSVs.

"""

import os
import pandas as pd

results_folder = "benchmark/results"
files = os.listdir(results_folder)
individual_tests = 0

for file in files:
    # check for directory, recursively
    if os.path.isdir(f"{results_folder}/{file}"):
        continue
    if "confidence" in file or "failure" in file:
        continue
    df = pd.read_csv(f"{results_folder}/{file}")
    individual_tests += df["iterations"].sum()

print(individual_tests)
