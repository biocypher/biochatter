# Get paths for images and text from the folders in benchmark/data/source_data_clip
# There is one folder for each test case, containing a jpg and a txt file with the same name
# The txt file contains the figure caption
# The jpg file contains the figure
# The expected answer is whether the figure caption actually belongs to the figure panel

# Load the data
import os
import pytest
import inspect
import hashlib
import numpy as np
from benchmark.conftest import calculate_bool_vector_score
from .benchmark_utils import (
    get_confidence_file_path,
    skip_if_already_run,
    get_result_file_path,
    write_confidence_to_file,
    write_results_to_file,
)


# Load the data
@pytest.fixture
def data_list():
    data_path = "benchmark/data/source_data_clip"
    return [
        os.path.join(data_path, folder)
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ]


# Run benchmark
def test_multimodal_answer(
    data_list, model_name, conversation, multiple_testing
):
    """
    Select randomly from the list of folders in data_list:
    - n examples with true positives (figure and caption match)
    - n examples with true negatives (figure and caption do not match)

    For each example, ask the model to determine whether the figure caption
    belongs to the figure panel. Also ask the model to give an estimate of its
    confidence. Record the answer (yes/no) and the confidence score. Write the
    results to a file.

    """
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    md5_hash = hashlib.md5(str(data_list).encode()).hexdigest()
    skip_if_already_run(model_name=model_name, task=task, md5_hash=md5_hash)

    # Skip if model has no multimodal capabilities
    if model_name not in [
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
    ]:
        pytest.skip("Model does not support multimodal input")

    # Set number of examples
    n = 2

    # True positives: list of tuples containing the same file name twice
    true_positives = [
        (f, f) for f in np.random.choice(data_list, n, replace=False)
    ]
    # True negatives: list of tuples containing different file names
    # Check that the randomly selected names are different
    true_negatives = [
        (f1, f2)
        for f1, f2 in [
            (np.random.choice(data_list), np.random.choice(data_list))
            for _ in range(n)
        ]
        if f1 != f2
    ]
    assert len(true_positives) == n
    assert len(true_negatives) == n

    def run_test():
        results = []
        for f1, f2 in true_positives + true_negatives:
            conversation.reset()
            # Load the image and the caption
            with open(os.path.join(f1, f1.split("/")[-1] + ".txt"), "r") as f:
                caption = f.read()
            image_path = os.path.join(f2, f2.split("/")[-1] + ".jpg")

            # Ask the model to determine whether the caption belongs to the figure
            response, _, _ = conversation.query(
                f"Does this caption describe the figure in the image? {caption}"
                "Answer with 'yes' or 'no' and give a confidence score between 0 and 10. "
                "Answer in the format 'yes, 8' or 'no, 2'.",
                image_url=image_path,
            )

            # Remove full stop from response
            response = response.replace(".", "")

            # Extract the answer and the confidence score
            answer = response.split(",")[0].strip().lower()
            try:
                confidence = response.split(",")[1].strip()
            except IndexError:
                confidence = None

            # Record the answer and the confidence
            results.append(
                {
                    "file": f1.split("/")[-1],
                    "answer": answer,
                    "confidence": confidence,
                }
            )

        # Consume results and write to file
        score = []
        correct_confidence = []
        incorrect_confidence = []
        # We expect the model to answer 'yes' for the first n examples and 'no' for the last n
        for i, result in enumerate(results):
            if i < n:
                score.append(result["answer"] == "yes")
            else:
                score.append(result["answer"] == "no")

            # collect confidence scores for correct and incorrect answers
            if not result["confidence"]:
                continue
            if score[-1]:
                correct_confidence.append(result["confidence"])
            else:
                incorrect_confidence.append(result["confidence"])

        # Record confidence scores
        if not correct_confidence:
            correct_confidence = ["None"]
        if not incorrect_confidence:
            incorrect_confidence = ["None"]
        write_confidence_to_file(
            model_name,
            "multimodal_answer",
            ";".join(correct_confidence),
            ";".join(incorrect_confidence),
            md5_hash,
            get_confidence_file_path(task),
        )

        return calculate_bool_vector_score(score)

    scores, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        "multimodal_answer",
        f"{scores}/{max}",
        f"{n_iterations}",
        md5_hash,
        get_result_file_path(task),
    )
