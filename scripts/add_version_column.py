import os

import pandas as pd


def _add_version_column():
    """Go through all benchmark result files and add a column for the version of
    biochatter, `biochatter_version`, to the results. Also add the value `0.4.10`
    as a default value to all rows.
    """
    result_files = [f"benchmark/results/{file}" for file in os.listdir("benchmark/results") if file.endswith(".csv")]

    for file in result_files:
        result_file = pd.read_csv(file, header=0)
        if "biochatter_version" not in result_file.columns:
            result_file["biochatter_version"] = "0.4.10"
            result_file.to_csv(file, index=False)


if __name__ == "__main__":
    _add_version_column()
