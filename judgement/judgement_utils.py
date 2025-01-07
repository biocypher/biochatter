import os
import pandas as pd
import pytest
import yaml

def list_files(path: str):
    """
    Lists all non-hidden files in a specified directory.

    This function iterates through the contents of a given directory and collects 
    the names of files that are not hidden (files that do not start with a dot `.`).

    Args:
        path (str): The path to the directory to be scanned for files.

    Returns:
        list: A list of filenames (str) in the specified directory, excluding hidden files.
    """

    files = []
    for file in os.listdir(path):
        if not file.startswith("."):
            files.append(file)
    return files

def read_file(path: str):
    """
    Reads a CSV file and returns its contents as a Pandas DataFrame.

    This function uses Pandas to load the contents of a specified CSV file into 
    a DataFrame for further analysis or processing.

    Args:
        path (str): The path to the CSV file to be read.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of the CSV file.
    """

    df = pd.read_csv(path)
    return df

def judgement_already_executed(
    judge_model: str,
    evaluated_model: str,
    metric: str,
    md5_hash: str,
) -> bool:
    """
    Checks if a specific judgement task has already been executed.

    This function determines whether a judgement task, identified by the judge model,
    evaluated model, metric, and a unique hash, has already been run by examining the 
    corresponding task results.

    Args:
        judge_model (str): The name of the model performing the judgement.
        evaluated_model (str): The name of the model being evaluated.
        metric (str): The evaluation metric for the task.
        md5_hash (str): A unique hash representing the task input data.

    Returns:
        bool: `True` if the task has already been executed, `False` otherwise.
    """

    task_results = return_or_create_judge_file(
        judge_model = judge_model,
        evaluated_model = evaluated_model,
        metric = metric,
    )
    if task_results.empty:
        return False
    
    run = task_results[(task_results["judge"] == judge_model) & (task_results["md5_hash"] == md5_hash)].shape[0] > 0
    return run

def skip_if_already_run(
    judge_model: str,
    evaluated_model: str,
    metric: str,
    md5_hash: str,
):
    """
    Skips the execution of a test if the specified judgement task has already been executed.

    This function checks whether a given judgement task, identified by the judge model, 
    evaluated model, metric, and a unique data hash, has been previously executed. If the 
    task has already been run, it uses `pytest.skip` to skip the current test, providing 
    a descriptive message.

    Args:
        judge_model (str): The name of the model performing the judgement.
        evaluated_model (str): The name of the model being evaluated.
        metric (str): The metric used for evaluation.
        md5_hash (str): A unique hash representing the specific test case or input data.
    """

    if judgement_already_executed(
        judge_model = judge_model,
        evaluated_model = evaluated_model,
        metric = metric,
        md5_hash = md5_hash,
    ):
        pytest.skip(f"Judgement for {evaluated_model} with hash {md5_hash} with {judge_model} already executed")

def return_or_create_judge_file(judge_model: str, evaluated_model: str, metric: str):
    """
    Creates a judgement DataFrame and saves it as a CSV file.
    
    Args:
        judged_model (str): Name of the judged model.
        evaluated_model (str): Name of the evaluated model.
        metric (str): Name of the metric.
    
    Returns:
        df: A dataframe.
    """

    path = f"./judgement/model_eval/{evaluated_model}_{metric}.csv"
    try:
        results = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        results = {
            "judge": [],
            "evaluated_model": [],
            "iterations": [],
            "metric": [],
            "subtask": [],
            "individual": [],
            "md5_hash": [],
            "prompt": [],
            "responses": [],
            "expected_answer": [],
            "rating": [],
        }
        results = pd.DataFrame(results)
        results.to_csv(path, index = False)
    return results

def write_judgement_to_file(
    judge_model: str,
    evaluated_model: str,
    iterations: str,
    metric: str,
    subtask: str,
    individual: str,
    md5_hash: str,
    prompt: str,
    responses: list,
    expected_answer: str,
    rating: str,
    path: str,
):
    """
    Appends a new judgement entry to an existing CSV file and saves the updated results.

    This function reads an existing CSV file containing judgement results, adds a new 
    row with the provided data, sorts the results by specified columns, and writes the 
    updated results back to the file.

    Args:
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

    results = pd.read_csv(path, header = 0)
    new_row = pd.DataFrame([
        [judge_model, evaluated_model, iterations, metric, subtask,
         individual, md5_hash, prompt, responses, expected_answer, rating]
    ], columns = results.columns)

    results = pd.concat([results, new_row], ignore_index = True).sort_values(
        by = ["judge", "metric"],
    )
    results.to_csv(path, index = False)

def get_prompt_binary(path: str, metric: str):
    """
    Retrieves a specific prompt based on a given metric from the `prompt.yaml` configuration file.

    This function reads a set of prompts from the `prompt.yaml` configuration file and extracts the prompt 
    that corresponds to the provided metric.

    Args:
        path (str): The path to the file containing the prompt configurations.
        metric (str): The evaluation metric for which the prompt is to be retrieved.

    Returns:
        str: The prompt string associated with the specified metric.
    """

    prompts = read_prompts(path)
    prompt = prompts[metric]["prompt"]
    return prompt

def read_prompts(path: str):
    """
    Reads and loads prompt configurations from a specified `yaml` file.

    This function opens a a specified `yaml` file at the specified path, reads its content, 
    and extracts the "prompts" section into a Python dictionary.

    Args:
        path (str): The path to the `yaml` file containing the prompt configurations.

    Returns:
        dict: A dictionary containing the "prompts" section of the `yaml` file.
    """

    with open(path, "r") as file:
        tasks = yaml.safe_load(file)["prompts"]
    return tasks

def get_judgement_dataset(path: str):
    """
    Retrieves judgement dataset from the specified directory.

    This function loads the judgement dataset by transferring the task to 
    `load_judgement_dataset` and returns the resulting data.

    Args:
        path (str): Path to the directory containing judgement files.

    Returns:
        dict: A dictionary containing the loaded judgement data.
    """

    test_data = load_judgement_dataset(path)
    return test_data

def load_judgement_dataset(path: str):
    """
    Loads and processes judgement dataset files from a specified directory.

    This function scans a directory for files, reads them as CSV files, 
    concatenates the contents if there are multiple files, and formats 
    the data into a dictionary.

    Args:
        path (str): Path to the directory containing judgement (response) files.

    Returns:
        dict: A dictionary with a single key `"judgement"` containing a list of records 
        from the loaded CSV files.

    Raises:
        ValueError: If no files are found in the specified directory.
    """

    files = list_files(path)

    if not files:
        raise ValueError(f"No files found in directory: {path}")
    
    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as err:
            print(f"Error reading {file_path}: {err}")
            continue
    
    if len(dfs) > 1:
        concatenated_dfs = pd.concat(dfs, ignore_index = True)
    else:
        concatenated_dfs = dfs[0]
    
    result_data = concatenated_dfs.to_dict(orient = "records")
    data_dict = {
        "judgement": result_data
    }

    return data_dict