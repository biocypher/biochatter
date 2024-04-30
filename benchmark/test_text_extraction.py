import inspect

import evaluate

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
    skip_if_already_run(
        model_name=model_name, task=task, md5_hash=yaml_data["hash"]
    )

    def run_test():
        conversation.reset()
        # Define the system prompt
        [
            conversation.append_system_message(
                yaml_data["input"]["system_messages"]
            )
        ]

        response, _, _ = conversation.query(
            f"FIGURE CAPTION: {yaml_data['input']['caption']} ##\n\n"
            f"## QUERY: {yaml_data['input']['query']} ##\n\n"
            f"## ANSWER FORMAT: {yaml_data['input']['format']}"
        )
        rouge_score = calculate_rouge_score(
            response, yaml_data["expected"]["answer"]
        )

        return (rouge_score, 1)

    mean_score, max, n_iterations = multiple_testing(run_test)

    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{mean_score}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )


def calculate_rouge_score(response, expected):
    """Application of the ROUGE metric to evaluate the response of the model."""
    rouge = evaluate.load("rouge")

    return rouge.compute(predictions=[response], references=[expected])[
        "rouge1"
    ]
