import os
import pytest

RESULT_FILES = [
    "benchmark/results/biocypher_query_generation.csv",
    "benchmark/results/vectorstore.csv",
    "benchmark/results/numeric_qa.csv",
]


@pytest.fixture(autouse=True, scope="session")
def delete_csv_files():
    """
    Reset benchmark output each time pytest is run.

    Todo:

        Probably not the most economic way to delete everything every time,
        should be extended to only overwrite the tests that have changed or add
        models that were not present before.

    """
    for f in RESULT_FILES:
        if os.path.exists(f):
            os.remove(f)

    # create blank CSV files
    for f in RESULT_FILES:
        with open(f, "w") as f:
            f.write("")


def calculate_test_score(vector: list[bool]):
    score = sum(vector)
    max = len(vector)
    return f"{score}/{max}"
