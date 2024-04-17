import os
import ast
import inspect

from datasets import load_metric
import pytest
import evaluate

from biochatter._misc import ensure_iterable
from biochatter.vectorstore import DocumentReader, DocumentEmbedder
from .conftest import calculate_test_score
from .benchmark_utils import (
    skip_if_already_run,
    get_result_file_path,
    write_results_to_file,
)


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

        avg_rouge_scores = []
        for caption, answers in zip(ensure_iterable(yaml_data["input"]["caption"]), ensure_iterable(yaml_data["expected"]["answer"])):
            rouge_scores = []
            for query, format_, answer in zip(
                ensure_iterable(yaml_data["input"]["query"]),
                ensure_iterable(yaml_data["input"]["format"]),
                ensure_iterable(answers)
            ):
                response, _, _ = conversation.query(
                    f"FIGURE CAPTION: {caption} ##\n\n## QUERY: {query} ##\n\n## ANSWER FORMAT: {format_}"
                )
                rouge_score = evaluate_response(response, answer)

                rouge_scores.append(rouge_score)
            avg_rouge_scores.append(sum(rouge_scores) / len(rouge_scores))

        return calculate_test_score(avg_rouge_scores)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )

def evaluate_response(response, expected):
    """Application of the ROUGE metric to evaluate the response of the model."""
    rouge = evaluate.load('rouge')

    return rouge.compute(
        predictions=[response],
        references=[expected]
        )["rouge1"]
