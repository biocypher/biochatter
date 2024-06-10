import re
import inspect

import nltk
import pytest

from biochatter._misc import ensure_iterable
from .conftest import calculate_bool_vector_score
from .benchmark_utils import (
    categorize_failure_modes,
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
    get_failure_mode_file_path,
    write_failure_modes_to_file,
)


def test_correctness_of_answers(
    model_name,
    test_data_correctness,
    conversation,
    multiple_testing,
):
    # Downloads the naturale language synonym toolkit, just need to be done once per device
    # nltk.download()

    yaml_data = test_data_correctness
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    failure_mode = "other"
    actual_answer = ""
    expected_answer = ""

    def run_test():
        nonlocal actual_answer
        nonlocal expected_answer
        nonlocal failure_mode
        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        # lower case, remove punctuation
        response = (
            response.lower().replace(".", "").replace("?", "").replace("!", "")
        ).strip()

        print(yaml_data["case"])
        print(response)
        # print(get_result_file_path(task))

        # calculate score of correct answers
        score = []

        # calculate for answers without regex and save response if not exactly
        # the same as expected (pretty much impossible for open questions)
        if "regex" not in yaml_data["case"]:
            expected_answer = yaml_data["expected"]["answer"]
            is_correct = response == expected_answer
            score.append(is_correct)
            if not is_correct:
                actual_answer = response
                failure_mode = categorize_failure_modes(
                    actual_answer, expected_answer
                )

        # calculate for answers with regex
        else:
            expected_word_pairs = yaml_data["expected"]["words_in_response"]
            for pair in expected_word_pairs:
                regex = "|".join(pair)
                expected_answer = regex
                if re.search(regex, response, re.IGNORECASE):
                    # print(f"Expected words '{pair}' found in response: {response}")
                    score.append(True)
                else:
                    score.append(False)
                    actual_answer = actual_answer + response
                    failure_mode = categorize_failure_modes(
                        actual_answer, expected_answer, True
                    )

        return calculate_bool_vector_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )
    if actual_answer != "":
        write_failure_modes_to_file(
            model_name,
            yaml_data["case"],
            actual_answer,
            expected_answer,
            failure_mode,
            yaml_data["hash"],
            get_failure_mode_file_path(task),
        )
