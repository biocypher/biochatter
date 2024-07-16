import inspect

import pytest

from biochatter._misc import ensure_iterable
from biochatter.api_agent.oncokb import (
    OncoKBQueryBuilder,
    OncoKBFetcher,
)
from .conftest import calculate_bool_vector_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


def test_api_calling(
    model_name,
    test_data_api_calling,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_api_calling
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )
    if False:
        pytest.skip(
            f"test case {yaml_data['case']} not supported for {task} benchmark"
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        builder = OncoKBQueryBuilder()
        parameters = builder.parameterise_query(
            question=yaml_data["input"]["prompt"],
            conversation=conversation,
        )

        fetcher = OncoKBFetcher()
        api_query = fetcher.submit_query(parameters)

        score = []
        for expected_part in ensure_iterable(
            yaml_data["expected"]["parts_of_query"]
        ):
            if expected_part in api_query:
                score.append(True)
            else:
                score.append(False)

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
