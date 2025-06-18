import re
from datetime import datetime
import os
import yaml

import importlib_metadata
import pandas as pd
import pytest
from nltk.corpus import wordnet


def benchmark_already_executed(
    model_name: str,
    task: str,
    md5_hash: str,
    mode: str | None = False,
    metric: str | None = False,
    judge_name: str | None = False,
) -> bool:
    """Check if the benchmark task and subtask test case for the model_name have already been executed.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        task (str): The benchmark task, e.g. "biocypher_query_generation"

        md5_hash (str): The md5 hash of the test case, e.g.,
            "72434e7a340a3f6dd047b944988491b7". It is created from the
            dictionary representation of the test case.

    Returns:
    -------
        bool: True if the benchmark case for the model_name has already been
            run, False otherwise

    """

    def task_executed(task_results, filters):
        if task_results.empty:
            return False
        return task_results.query(filters).shape[0] > 0

    if mode is False and md5_hash != "?":
        task_results = return_or_create_result_file(task)

        # check if failure group csv already exists
        return_or_create_failure_mode_file(task)

        # check if confidence group csv already exists
        return_or_create_confidence_file(task)

        return task_executed(
            task_results=task_results,
            filters=f"model_name == '{model_name}' and md5_hash == '{md5_hash}'",
        )
    elif mode == "response":
        task_results = return_or_create_response_file(task, model_name)

        return task_executed(
            task_results=task_results,
            filters=f"model_name == '{model_name}' and md5_hash == '{md5_hash}'",
        )
    elif mode == "judge":
        task_results = return_or_create_judge_file(task, model_name)

        return task_executed(
            task_results=task_results,
            filters=f"metric == '{metric}' and model_name == '{model_name}' and judge == '{judge_name}' and md5_hash == '{md5_hash}'",
        )

    return False


def skip_if_already_run(
    model_name: str,
    task: str,
    md5_hash: str,
    mode: str | None = False,
    metric: str | None = False,
    judge_name: str | None = False,
) -> None:
    """Check if the test case is already executed.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        task (str): The benchmark task, e.g. "biocypher_query_generation"

        md5_hash (str): The md5 hash of the test case, e.g.,
            "72434e7a340a3f6dd047b944988491b7". It is created from the
            dictionary representation of the test case.

    """
    if benchmark_already_executed(model_name, task, md5_hash, mode, metric, judge_name):
        message = f"{mode} mode with {metric}" if mode and metric else "Benchmark"
        pytest.skip(
            f"{message} for {task} with hash {md5_hash} with {model_name} already executed",
        )


def return_or_create_result_file(
    task: str,
):
    """Return the result file for the task or create it if it does not exist.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        pd.DataFrame: The result file for the task

    """
    file_path = get_result_file_path(task)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "subtask",
                "score",
                "iterations",
                "md5_hash",
                "datetime",
                "biochatter_version",
            ],
        )
        results.to_csv(file_path, index=False)
    return results


def return_or_create_failure_mode_file(task: str):
    """Return the failure mode file for the task or create it if it does not exist.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        pd.DataFrame: The failure mode recording file for the task

    """
    file_path = get_failure_mode_file_path(task)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "subtask",
                "actual_answer",
                "expected_answer",
                "failure_modes",
                "md5_hash",
                "datetime",
            ],
        )
        results.to_csv(file_path, index=False)
    return results


def return_or_create_confidence_file(task: str):
    """Return the confidence file for the task or create it if it does not exist.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        pd.DataFrame: The confidence recording file for the task

    """
    file_path = get_confidence_file_path(task)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "subtask",
                "correct_confidence",
                "incorrect_confidence",
                "md5_hash",
                "datetime",
            ],
        )
        results.to_csv(file_path, index=False)
    return results


def return_or_create_response_file(task: str, model: str):
    """Return the result file for the task or create it if it does not exist.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        pd.DataFrame: The judgement file for the judgment task

    """
    file_path = get_response_mode_file_path(task, model)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "case_id",
                "subtask",
                "age",
                "prompt",
                "response",
                "expected_answer",
                "summary",
                "key_words",
                "type",
                "iterations",
                "md5_hash",
                "datetime",
                "biochatter_version",
            ]
        )
        results.to_csv(file_path, index=False)
    return results


def return_or_create_rag_response_file(task: str, model: str):
    """Return the result file for the task or create it if it does not exist.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        pd.DataFrame: The judgement file for the judgment task

    """
    file_path = get_rag_response_mode_file_path(task, model)
    try:
        results = pd.read_csv(file_path, header=0)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = pd.DataFrame(
            columns=[
                "model_name",
                "case_id",
                "subtask",
                "individual",
                "prompt",
                "response",
                "expected_answer",
                "summary",
                "key_words",
                "iterations",
                "md5_hash",
                "datetime",
                "biochatter_version",
            ]
        )
        results.to_csv(file_path, index=False)
    return results


def get_confidence_file_path(task: str) -> str:
    """Return the path to the confidence recording file.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        str: The path to the confidence file

    """
    return f"benchmark/results/{task}_confidence.csv"


def get_failure_mode_file_path(task: str) -> str:
    """Return the path to the failure mode recording file.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        str: The path to the failure mode file

    """
    return f"benchmark/results/{task}_failure_modes.csv"


def get_response_mode_file_path(task: str, model: str) -> str:
    """Return the path to the response mode recording file.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        str: The path to the response mode file

    """
    # return f"benchmark/LLM_as_a_Judge/responses/{task}_{model}_response.csv"
    return f"benchmark/results/{task}_{model}_response.csv"


def get_rag_response_mode_file_path(task: str, model: str) -> str:
    """Return the path to the failure mode recording file.

    Args:
    ----
        task (str): The benchmark task, e.g. "biocypher_query_generation"

    Returns:
    -------
        str: The path to the response mode file

    """
    return f"benchmark/LLM_as_a_Judge/responses/{task}_{model}_rag_response.csv"


def write_results_to_file(
    model_name: str,
    subtask: str,
    score: str,
    iterations: str,
    md5_hash: str,
    file_path: str,
):
    """Write the benchmark results for the subtask to the result file.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        subtask (str): The benchmark subtask test case, e.g. "entities"
        score (str): The benchmark score, e.g. "5"
        iterations (str): The number of iterations, e.g. "7"
        md5_hash (str): The md5 hash of the test case
        file_path (str): The path to the result file

    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bc_version = importlib_metadata.version("biochatter")
    new_row = pd.DataFrame(
        [[model_name, subtask, score, iterations, md5_hash, now, bc_version]],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"],
    )
    results.to_csv(file_path, index=False)


def write_confidence_to_file(
    model_name: str,
    subtask: str,
    correct_confidence: str,
    incorrect_confidence: str,
    md5_hash: str,
    file_path: str,
):
    """Write the confidence scores for a given response to a subtask to the given file path.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        subtask (str): The benchmark subtask test case, e.g. "multimodal_answer"

        correct_confidence (str): The confidence scores for correct answers

        incorrect_confidence (str): The confidence scores for incorrect answers

        md5_hash (str): The md5 hash of the test case

        file_path (str): The path to the result file

    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame(
        [
            [
                model_name,
                subtask,
                correct_confidence,
                incorrect_confidence,
                md5_hash,
                now,
            ],
        ],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"],
    )
    results.to_csv(file_path, index=False)


def write_failure_modes_to_file(
    model_name: str,
    subtask: str,
    actual_answer: str,
    expected_answer: str,
    failure_modes: str,
    md5_hash: str,
    file_path: str,
):
    """Write the failure modes identified for a given response to a subtask to the given file path.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"

        subtask (str): The benchmark subtask test case, e.g. "entities"

        actual_answer (str): The actual response given to the subtask question

        expected_answer (str): The expected response for the subtask

        failure_modes (str): The mode of failure, e.g. "Wrong word count",
        "Formatting", etc.

        md5_hash (str): The md5 hash of the test case

        file_path (str): The path to the result file

    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame(
        [
            [
                model_name,
                subtask,
                actual_answer,
                expected_answer,
                failure_modes,
                md5_hash,
                now,
            ],
        ],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "subtask"],
    )
    results.to_csv(file_path, index=False)


def categorize_failure_modes(
    actual_answer,
    expected_answer,
    regex=False,
) -> str:
    """Categorise the mode of failure for a given response to a subtask.

    Args:
    ----
        actual_answer (str): The actual response given to the subtask question

        expected_answer (str): The expected response for the subtask

        regex (bool): Whether the expected answer is a regex expression

    Returns:
    -------
        str: The mode of failure, e.g. "Case Sensitivity", "Partial Match",
            "Format Error", "Synonym", "Words Missing", "Entire Answer Incorrect",
            "Other"

    """
    if not regex:
        # Check if the answer is right, but the case sensitivity was wrong (e.g. a / A)
        if actual_answer.lower() == expected_answer.lower():
            return "Case Sensitivity"

        # Check if the wrong answer contains the expected answer followed by ")"
        elif actual_answer.strip() == expected_answer + ")":
            return "Format Error"

        # Check if some of the answer is partially right, but only if it's more than one letter
        elif len(expected_answer) > 1 and (actual_answer in expected_answer or expected_answer in actual_answer):
            return "Partial Match"

        # Check if the format of the answer is wrong, but the answer otherwise is right (e.g. "a b" instead of "ab")
        elif re.sub(r"\s+", "", actual_answer.lower()) == re.sub(
            r"\s+",
            "",
            expected_answer.lower(),
        ):
            return "Format Error"

        # Check if the answer is a synonym with nltk (e.g. Illness / Sickness)
        elif is_synonym(actual_answer, expected_answer):
            return "Synonym"

        # Check if the format of the answer is wrong due to numerical or alphabetic differences (e.g. "123" vs "one two three")
        elif (
            re.search(r"\w+", actual_answer)
            and re.search(r"\w+", expected_answer)
            and any(char.isdigit() for char in actual_answer) != any(char.isdigit() for char in expected_answer)
        ):
            return "Format Error"

        # Check if partial match with case sensitivity
        elif actual_answer.lower() in expected_answer.lower() or expected_answer.lower() in actual_answer.lower():
            return "Partial Match / case Sensitivity"

        # Else the answer may be completely wrong
        else:
            return "Other"

    elif all(word in expected_answer for word in actual_answer.split()):
        return "Words Missing"

    # Check if some words in actual_answer are incorrect (present in actual_answer but not in expected_answer)
    # elif any(word not in expected_answer for word in actual_answer.split()):
    #   return "Incorrect Words"

    # Check if the entire actual_answer is completely different from the expected_answer
    else:
        return "Entire Answer Incorrect"


def is_synonym(word1, word2):
    """Test if the input arguments word1 and word2 are synonyms of each other.

    If yes, the function returns True, False otherwise.
    """
    if word2.lower() in ["yes", "no", "ja", "nein"]:
        return False

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.wup_similarity(synset2) is not None:
                return True
    return False


def write_responses_to_file(
    model_name: str,
    case_id: str,
    subtask: str,
    individual: str,
    prompt: str,
    responses: list,
    expected_answer: str,
    summary: str,
    key_words: list,
    type: str,
    iterations: str,
    md5_hash: str,
    file_path: str,
):
    """Write the benchmark responses for the subtask to the response file.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        subtask (str): The benchmark subtask test case, e.g. "entities"
        prompt (str): The prompt used for instructing the model.
        responses (list): The response(-s) which were generated by the model for a query/task.
        iterations (str): The number of iterations, e.g. "7"
        md5_hash (str): The md5 hash of the test case
        file_path (str): The path to the responses file

    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bc_version = importlib_metadata.version("biochatter")
    new_row = pd.DataFrame(
        [
            [
                model_name,
                case_id,
                subtask,
                individual,
                prompt,
                responses,
                expected_answer,
                summary,
                key_words,
                type,
                iterations,
                md5_hash,
                now,
                bc_version,
            ]
        ],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(by=["model_name", "subtask"])
    results.to_csv(file_path, index=False)


def write_rag_responses_to_file(
    model_name: str,
    case_id: str,
    subtask: str,
    individual: str,
    prompt: str,
    responses: list,
    contexts,
    expected_answer: str,
    summary: str,
    key_words: list,
    iterations: str,
    md5_hash: str,
    file_path: str,
):
    """Write the benchmark responses (RAG) for the subtask to the response (RAG) file.

    Args:
    ----
        model_name (str): The model name, e.g. "gpt-3.5-turbo"
        subtask (str): The benchmark subtask test case, e.g. "entities"
        prompt (str): The prompt used for instructing the model.
        responses (list): The response(-s) which were generated by the model for a query/task.
        iterations (str): The number of iterations, e.g. "7"
        md5_hash (str): The md5 hash of the test case
        file_path (str): The path to the response (RAG) file

    """
    results = pd.read_csv(file_path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bc_version = importlib_metadata.version("biochatter")
    new_row = pd.DataFrame(
        [
            [
                model_name,
                case_id,
                subtask,
                individual,
                prompt,
                responses,
                contexts,
                expected_answer,
                summary,
                key_words,
                iterations,
                md5_hash,
                now,
                bc_version,
            ]
        ],
        columns=results.columns,
    )
    results = pd.concat([results, new_row], ignore_index=True).sort_values(by=["model_name", "subtask"])
    results.to_csv(file_path, index=False)


# TODO should we use SQLite? An online database (REDIS)?
def get_result_file_path(file_name: str) -> str:
    """Return the path to the result file.

    Args:
    ----
        file_name (str): The name of the result file

    Returns:
    -------
        str: The path to the result file

    """
    return f"benchmark/results/{file_name}.csv"


###########################
#   FUNCTIONS JUDGEMENT   #
###########################


def list_files(path: str):
    """List all non-hidden files in a specified directory.

    This function iterates through the contents of a given directory and collects
    the names of files that are not hidden (files that do not start with a dot `.`).

    Args:
    ----
        path (str): The path to the directory to be scanned for files.

    Returns:
    -------
        list: A list of filenames (str) in the specified directory, excluding hidden files.

    """
    files = []
    for file in os.listdir(path):
        if not file.startswith(".") and file.endswith("_response.csv"):
            files.append(file)
    return files


def read_file(path: str):
    """Read a CSV file and return its contents as a Pandas DataFrame.

    This function uses Pandas to load the contents of a specified CSV file into
    a DataFrame for further analysis or processing.

    Args:
    ----
        path (str): The path to the CSV file to be read.

    Returns:
    -------
        pd.DataFrame: A DataFrame containing the contents of the CSV file.

    """
    df = pd.read_csv(path)
    return df


def return_or_create_judge_file(task: str, evaluated_model: str):
    """Create a judgement DataFrame and save it as a CSV file.

    Args:
    ----
        judged_model (str): Name of the judged model.
        evaluated_model (str): Name of the evaluated model.
        metric (str): Name of the metric.

    Returns:
    -------
        df: A dataframe.

    """
    path = f"./benchmark/results/{task}.csv"
    try:
        results = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = {
            "judge": [],
            "model_name": [],
            "iterations": [],
            "metric": [],
            "case_id": [],
            "subtask": [],
            "age": [],
            "md5_hash": [],
            "prompt": [],
            "system_prompt": [],
            "prompt_type": [],
            "is_distractor": [],
            "type": [],
            "responses": [],
            "expected_answer": [],
            "score": [],
            "datetime": [],
            "biochatter_version": [],
        }
        results = pd.DataFrame(results)
        results.to_csv(path, index=False)
    return results


def write_judgement_to_file(
    judge_model: str,
    evaluated_model: str,
    iterations: str,
    metric: str,
    case_id: str,
    subtask: str,
    individual: str,
    md5_hash: str,
    prompt: str,
    system_prompt: str,
    prompt_type: str,
    is_distractor: str,
    type_: str,
    responses: list,
    expected_answer: str,
    rating: str,
    path: str,
):
    """Append a new judgement entry to an existing CSV file and save the updated results.

    This function reads an existing CSV file containing judgement results, adds a new
    row with the provided data, sorts the results by specified columns, and writes the
    updated results back to the file.

    Args:
    ----
        judge_model (str): The name of the model performing the judgement.
        evaluated_model (str): The name of the model being evaluated.
        iterations (str): The number of iterations performed for the judgement.
        metric (str): The metric used for evaluation.
        subtask (str): The subtask or specific scenario being judged.
        individual (str): Metadata, specific for the longevity and geriatrics benchmark example.
        md5_hash (str): A unique hash representing the input data for the judgement.
        prompt (str): The prompt used to query the evaluated model.
        responses (list): The model's responses to the prompt.
        expected_answer (str): The expected answer for the task.
        rating (str): The rating or score assigned to the response.
        path (str): Path to the CSV file where the judgement results are stored.

    """
    results = pd.read_csv(path, header=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bc_version = importlib_metadata.version("biochatter")
    new_row = pd.DataFrame(
        [
            [
                judge_model,
                evaluated_model,
                iterations,
                metric,
                case_id,
                subtask,
                individual,
                md5_hash,
                prompt,
                system_prompt,
                prompt_type,
                is_distractor,
                type_,
                responses,
                expected_answer,
                rating,
                now,
                bc_version,
            ]
        ],
        columns=results.columns,
    )

    results = pd.concat([results, new_row], ignore_index=True).sort_values(
        by=["model_name", "metric"],
    )
    results.to_csv(path, index=False)


def read_prompts(path: str):
    """Read and load prompt configurations from a specified `yaml` file.

    This function opens a a specified `yaml` file at the specified path, reads its content,
    and extracts the "prompts" section into a Python dictionary.

    Args:
    ----
        path (str): The path to the `yaml` file containing the prompt configurations.

    Returns:
    -------
        dict: A dictionary containing the "prompts" section of the `yaml` file.

    """
    with open(path) as file:
        tasks = yaml.safe_load(file)["prompts"]
    return tasks


def get_prompt_binary(path: str):
    """Retrieve a specific prompt based on a given metric from the `prompt.yaml` configuration file.

    This function reads a set of prompts from the `prompt.yaml` configuration file and extracts the prompt
    that corresponds to the provided metric.

    Args:
    ----
        path (str): The path to the file containing the prompt configurations.
        metric (str): The evaluation metric for which the prompt is to be retrieved.

    Returns:
    -------
        str: The prompt string associated with the specified metric.

    """
    prompts = read_prompts(path)
    # prompt = prompts[metric]["prompt"]
    return prompts


def load_judgement_dataset(path: str):
    """Load and process judgement dataset files from a specified directory.

    This function scans a directory for files, reads them as CSV files,
    concatenates the contents if there are multiple files, and formats
    the data into a dictionary.

    Args:
    ----
        path (str): Path to the directory containing judgement (response) files.

    Returns:
    -------
        dict: A dictionary with a single key `"judgement"` containing a list of records
        from the loaded CSV files.

    Raises:
    ------
        ValueError: If no files are found in the specified directory.

    """
    files = list_files(path)

    latest_file = [max([f"{path}/{file}" for file in files], key=os.path.getmtime)]

    dfs = []
    for file in files:  # or files if each response file should be judged
        file_path = os.path.join(path, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as err:
            print(f"Error reading {file}: {err}")
            continue

        if len(dfs) > 1:
            concatenated_dfs = pd.concat(dfs, ignore_index=True)
        else:
            concatenated_dfs = dfs[0]

        result_data = concatenated_dfs.to_dict(orient="records")
        data_dict = {"judgement": result_data}

    return data_dict


def get_judgement_dataset(path: str):
    """Retrieve judgement dataset from the specified directory.

    This function loads the judgement dataset by transferring the task to
    `load_judgement_dataset` and returns the resulting data.

    Args:
    ----
        path (str): Path to the directory containing judgement files.

    Returns:
    -------
        dict: A dictionary containing the loaded judgement data.

    """
    test_data = load_judgement_dataset(path)
    return test_data
