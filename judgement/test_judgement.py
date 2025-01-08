import inspect

from judgement_utils import (
    skip_if_already_run,
    write_judgement_to_file,
    get_prompt_binary,
)
from conftest import calculate_bool_vector_score

def test_judgement(
    test_judgement,
    create_conversation,
    model_name,
    multiple_testing,
):
    """
    Tests the judgement process for a given model and data set, evaluating correctness.

    This function evaluates a model's ability to judge responses using a benchmark 
    dataset. It generates prompts, interacts with the model multiple times, calculates 
    a performance score, and logs the results.

    Args:
        test_judgement (dict): A dictionary containing test-specific data, including:
            - model_name (str): The name of the model being evaluated.
            - md5_hash (str): A unique hash representing the test case.
            - prompt (str): The user-level input prompt for the test.
            - summary (str): A summary of the test case context.
            - key_words (list[str]): Key words associated with the test case.
            - response (str): The response from the evaluated model.
            - expected_answer (str): The expected correct answer for the test.
            - subtask (str): The subtask or category for the test case.
            - individual (str): Metadata, specific for the longevity and geriatrics benchmark example.
        create_conversation (callable): A function for creating model interactions.
        model_name (str): The name of the model used as the judge.
        multiple_testing (callable): A function for running tests multiple times and aggregating results.
    """

    metric = "correctness"
    # task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

    skip_if_already_run(
        judge_model = model_name,
        evaluated_model = test_judgement["model_name"],
        metric = metric,
        md5_hash = test_judgement["md5_hash"],
    )

    def run_test():
        prompt_template = get_prompt_binary(
            "./judgement/prompts/prompts.yaml",
            metric,
        )
        success, failure = "correct", "not correct"
        prompt = prompt_template.format(
            success = success,
            failure = failure,
            prompt = test_judgement["prompt"],
            summary = test_judgement["summary"],
            keywords = test_judgement["key_words"],
            response = test_judgement["response"],
            expected_answer = test_judgement["expected_answer"],
        )

        model = model_name
        base_url = "https://api.openai.com/v1"

        score = []
        ITERATIONS = 3
        for iter in range(ITERATIONS):
            message = create_conversation(
                model, 
                base_url,
                "As a research assistant, your task is to assess the processing of a scientific question and the transmitted answer as carried out by another LLM.",
                prompt,
            )
            if message.lower().replace(".", "") == success:
                score.append(True)
            elif message.lower().replace(".", "") == failure:
                score.append(False)
        
        print(score)
        return calculate_bool_vector_score(score)
    
    scores, max, n_iterations = multiple_testing(run_test)

    if "list" in test_judgement["subtask"]:
        prompt_type = "list"
    elif "freetext" in test_judgement["subtask"]:
        prompt_type = "freetext"

    if ":appendix:" in test_judgement["subtask"]:
        clause = "appendix"
    elif "no_appendix" in test_judgement["subtask"]:
        clause = "no_appendix"

    if "simple" in test_judgement["subtask"]:
        system_prompt = "simple"
    elif "detailed" in test_judgement["subtask"]:
        system_prompt = "detailed"
    elif "explicit" in test_judgement["subtask"]:
        system_prompt = "explicit"

    write_judgement_to_file(
        judge_model = model_name,
        evaluated_model = test_judgement["model_name"],
        iterations = f"{n_iterations}",
        metric = metric,
        case_id = test_judgement["case_id"],
        subtask = test_judgement["subtask"],
        individual = test_judgement["individual"],
        md5_hash = test_judgement["md5_hash"],
        prompt = test_judgement["prompt"],
        system_prompt = system_prompt,
        prompt_type = prompt_type,
        is_appendix = clause,
        responses = test_judgement["response"],
        expected_answer = test_judgement["expected_answer"],
        rating = f"{scores}/{1}",
        path = f"./judgement/model_eval/{test_judgement['model_name']}_{metric}.csv",
    )
