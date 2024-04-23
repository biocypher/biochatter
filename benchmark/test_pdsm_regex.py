import re
import inspect

from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def test_asymmetry_calculations(
        model_name,
        test_data_pdsm_regex,
        conversation,
        multiple_testing
):
    yaml_data = test_data_pdsm_regex
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    def run_test():

        conversation.reset()  # needs to be reset for each test
        [
            conversation.append_system_message(m)
            for m in yaml_data["input"]["system_messages"]
        ]
        response, _, _ = conversation.query(yaml_data["input"]["prompt"])

        score = []

        expected_word_pairs = yaml_data["expected"]["words_in_response"]
        for pair in expected_word_pairs:
            regex = "|".join(pair)
            if re.search(regex, response, re.IGNORECASE):
                score.append(True)
            else:
                score.append(False)
                print(f"Expected words '{pair}' not found in response: {response}")

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



