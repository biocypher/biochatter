import inspect
import re

import pytest

from biochatter._misc import ensure_iterable
from biochatter.api_agent import (
    BioToolsQueryBuilder,
    OncoKBQueryBuilder,
    ScanpyPlQueryBuilder,
    ScanpyPlQueryBuilderReduced,
    AnnDataIOQueryBuilder,
    format_as_rest_call,
    format_as_python_call,
)

from .benchmark_utils import (
    get_result_file_path,
    skip_if_already_run,
    write_results_to_file,
)
from .conftest import calculate_bool_vector_score


def test_web_api_calling(
    model_name,
    test_data_api_calling,
    conversation,
    multiple_testing,
):
    yaml_data = test_data_api_calling
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    skip_if_already_run(
        model_name=model_name,
        task=task,
        md5_hash=yaml_data["hash"],
    )
    if "gpt-" not in model_name:
        pytest.skip(
            f"model {model_name} does not support API calling for {task} benchmark",
        )
    if "scanpy" in yaml_data["case"]:
        pytest.skip(
            "scanpy is not a web API",
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        if "oncokb" in yaml_data["case"]:
            builder = OncoKBQueryBuilder()
        elif "biotools" in yaml_data["case"]:
            builder = BioToolsQueryBuilder()
        parameters = builder.parameterise_query(
            question=yaml_data["input"]["prompt"],
            conversation=conversation,
        )

        api_query = format_as_rest_call(parameters[0])

        score = []
        for expected_part in ensure_iterable(
            yaml_data["expected"]["parts_of_query"],
        ):
            if re.search(expected_part, api_query):
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


def test_python_api_calling(
    model_name,
    test_data_api_calling,
    conversation,
    multiple_testing,
):
    """Test the Python API calling capability."""
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    yaml_data = test_data_api_calling

    skip_if_already_run(
        model_name=model_name,
        task=task,
        md5_hash=yaml_data["hash"],
    )

    if "scanpy" not in yaml_data["case"] and "anndata" not in yaml_data["case"]:
        pytest.skip(
            "Function to be tested is not a Python API",
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        if "scanpy:pl" in yaml_data["case"]:
            builder = ScanpyPlQueryBuilder()
        elif "anndata" in yaml_data["case"]:
            builder = AnnDataIOQueryBuilder()
        parameters = builder.parameterise_query(
            question=yaml_data["input"]["prompt"],
            conversation=conversation,
        )

        method_call = format_as_python_call(parameters[0]) if parameters else ""

        score = []
        for expected_part in ensure_iterable(
            yaml_data["expected"]["parts_of_query"],
        ):
            if re.search(expected_part, method_call):
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

def test_python_api_calling_reduced(
    model_name,
    test_data_api_calling,
    conversation,
    multiple_testing,
):
    """Test the Python API calling capability with reduced Scanpy plotting class."""
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    yaml_data = test_data_api_calling

    skip_if_already_run(
        model_name=model_name,
        task=task,
        md5_hash=yaml_data["hash"],
    )

    if "scanpy:pl" not in yaml_data["case"]:
        pytest.skip(
            "Function to be tested is not a Scanpy plotting API",
        )

    def run_test():
        conversation.reset()  # needs to be reset for each test
        builder = ScanpyPlQueryBuilderReduced()
        parameters = builder.parameterise_query(
            question=yaml_data["input"]["prompt"],
            conversation=conversation,
        )

        method_call = format_as_python_call(parameters[0]) if parameters else ""

        score = []
        for expected_part in ensure_iterable(
            yaml_data["expected"]["parts_of_query"],
        ):
            if re.search(expected_part, method_call):
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