from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
)
from biochatter._misc import ensure_iterable
import os
import inspect
import pytest
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)
import evaluate
from datasets import load_metric
import ast

def test_sourcedata_info_extraction(
    model_name,
    test_data_text_extraction,
    conversation,
    multiple_testing,
):
    """Test sourcedata info extraction by the model.
    The user input is a figure caption and what the model should extract.
    The system prompt has the guidelines given to the professional curators
    of SourceData, and the expected answer is the information that the model
    should extract from the figure caption. The test is successful if the
    extracted information matches the expected answer.
    """
    yaml_data = test_data_text_extraction
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    def run_test():
        conversation.reset()

        # Define the system prompt
        [conversation.append_system_message(yaml_data["input"]["system_messages"])]

        rouge_scores = []
        for caption in ensure_iterable(yaml_data["input"]["caption"]):
            for query, format_, answer in zip(
                ensure_iterable(yaml_data["input"]["query"]),
                ensure_iterable(yaml_data["input"]["format"]),
                ensure_iterable(yaml_data["expected"]["answer"])
            ):
                response, _, _ = conversation.query(
                    f"FIGURE CAPTION: {caption} ##\n\n## QUERY: {query} ##\n\n## ANSWER FORMAT: {format_}"
                )
                
                rouge_score = evaluate_response(response, answer)

                rouge_scores.append(rouge_score)

        return calculate_test_score(rouge_scores)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )

def check_if_correct_format(answer, format_):
    try:
        # Try to safely evaluate the string
        result = ast.literal_eval(answer)
        # Check the type of the result
        if isinstance(result, format_):
            return True
        else:
            return False
    except:
        raise ValueError("The answer is not in the correct format")

def evaluate_response(response, expected):
    # Check if the response is the expected one
    rouge = evaluate.load('rouge')

    return rouge.compute(
        predictions=[response],
        references=[expected]
        )["rouge1"]
