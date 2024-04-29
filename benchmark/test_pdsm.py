import inspect
import re

import pytest

from biochatter._misc import ensure_iterable
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def test_correctness_of_answers(
    model_name,
    test_data_pdsm,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_pdsm
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    # Wieder einkommentieren, wenn benötigt
    '''skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )'''


    def run_test():
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

        print(response)
        print(get_result_file_path(task))

        score = []
        if "regex" not in yaml_data["case"]:
            score.append(response == yaml_data["expected"]["answer"])
        else:
            expected_word_pairs = yaml_data["expected"]["words_in_response"]
            for pair in expected_word_pairs:
                regex = "|".join(pair)
                if re.search(regex, response, re.IGNORECASE):
                    # print(f"Expected words '{pair}' found in response: {response}")
                    score.append(True)
                else:
                    score.append(False)

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )

'''
def test_correctness_with_regex(
        model_name,
        test_data_pdsm,
        conversation,
        multiple_testing
):
    yaml_data = test_data_pdsm
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    # Skip if the test doesn't need regex
    if "regex" not in yaml_data["case"]:
        pytest.skip(
            f"test case {yaml_data['case']} not supported for {task} benchmark"
        )

    def run_test():

        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]

        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        response = (
            response.lower().replace(".", "").replace("?", "").replace("!", "")
        ).strip()

        score = []
        print(response)

        expected_word_pairs = yaml_data["expected"]["words_in_response"]
        for pair in expected_word_pairs:
            regex = "|".join(pair)
            if re.search(regex, response, re.IGNORECASE):
                #print(f"Expected words '{pair}' found in response: {response}")
                score.append(True)
            else:
                score.append(False)
                #print(f"Expected words '{pair}' not found in response: {response}")

        return calculate_test_score(score)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )
    
    '''


