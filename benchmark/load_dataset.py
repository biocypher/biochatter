"""Module for loading and managing benchmark datasets.

Including both public test data from the repository and encrypted hold-out test
data.
"""

import copy
import hashlib
import itertools
import json
import os
from base64 import b64decode
from pathlib import Path

import pandas as pd
import rsa
import yaml
from cryptography.fernet import Fernet


def get_benchmark_dataset() -> dict[str, pd.DataFrame | dict[str, str]]:
    """Get benchmark dataset.

    - if the env variable HOLD_OUT_TEST_DATA_PRIVATE_KEY is set, the hold out
    test set is used

    - otherwise, the public test set from this repository is used

    Returns
    -------
        dict: keys are filenames and values are test data.

    """
    if os.environ.get("HOLD_OUT_TEST_DATA_PRIVATE_KEY"):
        test_data = _load_hold_out_test_dataset()
    else:
        test_data = _load_test_data_from_this_repository()

    return test_data


def _load_hold_out_test_dataset() -> dict[str, pd.DataFrame | dict[str, str]]:
    """Load hold out test dataset.

    Returns
    -------
        dict: keys are filenames and values are test data.

    """
    print("Use hold out test data for benchmarking")
    private_key = _get_private_key_from_env_variable()
    encrypted_test_data = _get_encrypted_test_data()
    return _decrypt_data(encrypted_test_data, private_key)


def _get_all_benchmark_files(directory_path: str) -> list[str]:
    """Get all files in the directory ending with _data.yaml.

    Args:
    ----
        directory_path (str): Path to the directory.

    Returns:
    -------
        list: List of file paths ending with _data.yaml.

    """
    directory = Path(directory_path)
    return [str(directory / f) for f in os.listdir(directory) if f.endswith("_data.yaml")]


def _load_test_data_from_this_repository() -> dict[str, pd.DataFrame | dict[str, str]]:
    """Load and combine test data from this repository.

    Returns
    -------
        dict: Combined test data from all YAML files.

    """
    print("Using public test data from this repository for benchmarking.")
    directory_path = "./benchmark/data"
    benchmark_files = _get_all_benchmark_files(directory_path)

    combined_data = {}
    for file_path in benchmark_files:
        with Path(file_path).open(encoding="utf-8", errors="ignore") as stream:
            try:
                yaml_data = yaml.safe_load(stream)
                if not isinstance(yaml_data, dict):
                    msg = f"Expected dict, got {type(yaml_data).__name__}"
                    raise TypeError(msg)
                yaml_data = _get_yaml_data(yaml_data)
                combined_data.update(yaml_data)
            except yaml.YAMLError as exc:
                print(exc)

    return combined_data


def _get_yaml_data(yaml_data: dict) -> dict:
    # expand multi-instruction tests
    yaml_data = _expand_multi_instruction(yaml_data)
    # generate hash for each case
    yaml_data = _hash_each_case(yaml_data)
    # delete benchmark results that have outdated hashes
    _delete_outdated_benchmark_results(yaml_data)
    return yaml_data


def _delete_outdated_benchmark_results(data_dict: dict) -> None:
    """Delete outdated benchmark results.

    Opens the corresponding result file for each test and deletes the results
    that have hashes that are not in the current test data.

    Args:
    ----
        data_dict (dict): The yaml data.

    """
    # get all current hashes for comparison
    current_hashes = [
        data_dict[key][i]["hash"]
        for key in data_dict
        if isinstance(data_dict[key], list)
        for i in range(len(data_dict[key]))
        if isinstance(data_dict[key][i], dict)
    ]

    # get all result files
    result_files = [f"benchmark/results/{file}" for file in os.listdir("benchmark/results") if file.endswith(".csv")]

    # delete outdated results
    for file in result_files:
        continue
        # turn off deletion for now

        if "multimodal_answer" in file:
            continue
        result_file = pd.read_csv(file, header=0)
        result_hashes = result_file["md5_hash"].to_list()
        for hash in result_hashes:
            if hash not in current_hashes:
                result_file = result_file[result_file["md5_hash"] != hash]
        result_file.to_csv(file, index=False)


def _hash_each_case(data_dict: dict) -> dict:
    """Create a hash for each test case.

    Create a hash in the test data to identify tests that have been run or
    modified.

    Args:
    ----
        data_dict (dict): The yaml data.

    Returns:
    -------
        dict: The yaml data with a hash for each case.

    """
    for key in data_dict:
        if isinstance(data_dict[key], list):
            for i in range(len(data_dict[key])):
                if isinstance(data_dict[key][i], dict):
                    data_dict[key][i]["hash"] = hashlib.md5(  # noqa: S324
                        json.dumps(data_dict[key][i]).encode("utf-8"),
                    ).hexdigest()

    return data_dict


def _expand_multi_instruction(data_dict: dict) -> dict:
    """Expand tests with input dictionaries that contain dictionaries.

    Args:
    ----
        data_dict (dict): The yaml data.

    Returns:
    -------
        dict: The expanded yaml data.

    """
    for module_key in data_dict:
        if "longevity" in module_key:
            data_dict[module_key] = _expand_longevity_test_cases(data_dict[module_key])
        if "kg_schemas" not in module_key:
            test_list = data_dict[module_key]
            expanded_test_list = []
            for test in test_list:
                dicts = {
                    key: value for key, value in test["input"].items() if isinstance(value, dict) and key != "format"
                }
                if not dicts:
                    expanded_test_list.append(test)
                    continue
                keys_lists = [list(value.keys()) for value in dicts.values()]
                for combination in itertools.product(*keys_lists):
                    query_type = None
                    new_case = copy.deepcopy(test)
                    new_case["case"] = ":".join(
                        [test["case"], *combination],
                    )
                    dict_keys = list(dicts)
                    key_value_combinations = zip(dict_keys, list(combination), strict=True)
                    for key, value in key_value_combinations:
                        new_case["input"][key] = dicts[key][value]
                        if key == "query":
                            new_case["input"]["format"] = test["input"]["format"][value]
                            query_type = value
                        elif key == "caption":
                            new_case["expected"]["answer"] = test["expected"]["answer"][value][query_type]

                    expanded_test_list.append(new_case)
            data_dict[module_key] = expanded_test_list

    return data_dict


def _expand_longevity_test_cases(data_dict: dict) -> dict:
    expanded_test_list = []

    message_dict = data_dict[0]["general_system_messages"]
    rag_message_dict = data_dict[1]["general_system_messages_rag"]

    for test in data_dict[2:]:
        prompt_dict = test["input"]["prompt"]
        if not isinstance(prompt_dict, dict):
            expanded_test_list.append(test)
            continue

        keys_lists = [list(value.keys()) for value in prompt_dict.values() if isinstance(value, dict)]
        for combination in itertools.product(*keys_lists):
            new_case = copy.deepcopy(test)
            new_case["case"] = ":".join([test["case"], *combination])

            # Update prompt values based on combination
            prompt_keys = [key for key, value in prompt_dict.items() if isinstance(value, dict)]
            for key, value in zip(prompt_keys, combination, strict=True):
                # First update individual prompt keys
                new_case["input"]["prompt"][key] = prompt_dict[key][value]

            new_case["input"]["prompt"] = " ".join(str(v) for v in new_case["input"]["prompt"].values())
            new_case["input"]["prompt"] = new_case["input"]["prompt"].strip()

            if "rag:" in new_case["case"]:
                messages = rag_message_dict
            else:
                messages = message_dict

            for key, value in messages.items():
                final_case = copy.deepcopy(new_case)
                final_case["input"]["system_messages"] = {key: value}
                expanded_test_list.append(final_case)

    return expanded_test_list


def _get_private_key_from_env_variable() -> rsa.PrivateKey:
    """Get the private key from an environment variable.

    Returns
    -------
        rsa.PrivateKey: The private key.

    """
    private_key_base64 = os.environ.get("HOLD_OUT_TEST_DATA_PRIVATE_KEY")
    private_key_str = b64decode(private_key_base64).decode("utf-8")
    return rsa.PrivateKey.load_pkcs1(private_key_str.encode("utf-8"))


def _get_encrypted_test_data() -> dict[str, dict[str, str]]:
    """Get encrypted test data.

    Currently from manually copied file benchmark/encrypted_llm_test_data.json
    TODO: automatically load test dataset (from github releases)?

    Returns
    -------
        dict: keys are filenames and values are encrypted test data.

    """
    json_file_path = "./benchmark/encrypted_llm_test_data.json"
    with Path(json_file_path).open() as json_file:
        return json.load(json_file)


def _decrypt_data(
    encrypted_test_data: dict,
    private_key: rsa.PrivateKey,
) -> dict:
    """Decrypt the test data.

    Args:
    ----
        encrypted_test_data (dict): keys are filenames and values are encrypted
            test data.

        private_key (rsa.PrivateKey): The private key.

    Returns:
    -------
        dict: keys are filenames and values are decrypted test data.

    Todo: genericise to eval automatically

    """
    decrypted_test_data = {}
    for key in encrypted_test_data:
        decrypted = _decrypt(encrypted_test_data[key], private_key)

        if key.endswith(".yaml"):
            try:
                yaml_data = yaml.safe_load(decrypted)
                if "_data" in key:
                    yaml_data = _get_yaml_data(yaml_data)

                decrypted_test_data[key] = yaml_data
            except yaml.YAMLError as exc:
                print(exc)

    return decrypted_test_data


def _decrypt(payload: dict[str, str], private_key: rsa.PrivateKey) -> str:
    """Decrypt a payload.

    Args:
    ----
        payload (Dict[str, str]): Payload with key and data to decrypt.
        private_key (rsa.PrivateKey): Private key to decrypt the payload.

    Returns:
    -------
        str: Decrypted data.

    """
    enc_symmetric_key = b64decode(payload["key"])
    enc_data = b64decode(payload["data"])
    symmetric_key = rsa.decrypt(enc_symmetric_key, private_key)
    f = Fernet(symmetric_key)
    return f.decrypt(enc_data).decode("utf-8")


if __name__ == "__main__":
    # just for debugging
    get_benchmark_dataset()
