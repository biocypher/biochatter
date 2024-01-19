import io
import json
import os
from ast import literal_eval
from base64 import b64decode

import pandas as pd
import rsa
import yaml
from cryptography.fernet import Fernet


def get_benchmark_dataset() -> dict[str, pd.DataFrame | dict[str, str]]:
    """

    Get benchmark dataset:

    - if the env variable HOLD_OUT_TEST_DATA_PRIVATE_KEY is set, the hold out
    test set is used

    - otherwise, the public test set from this repository is used

    Returns:
        dict: keys are filenames and values are test data.
    """
    if os.environ.get("HOLD_OUT_TEST_DATA_PRIVATE_KEY"):
        test_data = _load_hold_out_test_dataset()
    else:
        test_data = _load_test_data_from_this_repository()

    return test_data


def _load_hold_out_test_dataset() -> dict[str, pd.DataFrame | dict[str, str]]:
    """Load hold out test dataset.

    Returns:
        dict: keys are filenames and values are test data.
    """
    print("Use hold out test data for benchmarking")
    private_key = _get_private_key_from_env_variable()
    encrypted_test_data = _get_encrypted_test_data()
    decrypted_test_data = _decrypt_data(encrypted_test_data, private_key)
    return decrypted_test_data


def _load_test_data_from_this_repository():
    """Load test data from this repository.

    Returns:
        dict: keys are filenames and values are test data.
    """
    print("Using public test data from this repository for benchmarking.")
    test_data = {}
    directory_path = "./benchmark/data"
    files_in_directory = _get_all_files(directory_path)
    for file_path in files_in_directory:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, sep=";")
            _apply_literal_eval(
                df,
                [
                    "entities",
                    "relationships",
                    "relationship_labels",
                    "properties",
                    "parts_of_query",
                ],
            )
            test_data[file_path.replace("./benchmark/", "./")] = df
        elif file_path.endswith(".yaml"):
            test_data[file_path.replace("./benchmark/", "./")] = yaml.safe_load(
                file_path
            )
    return test_data


def _get_private_key_from_env_variable() -> rsa.PrivateKey:
    """Get the private key from an environment variable.

    Returns:
        rsa.PrivateKey: The private key.
    """
    private_key_base64 = os.environ.get("HOLD_OUT_TEST_DATA_PRIVATE_KEY")
    private_key_str = b64decode(private_key_base64).decode("utf-8")
    private_key = rsa.PrivateKey.load_pkcs1(private_key_str.encode("utf-8"))
    return private_key


def _get_encrypted_test_data() -> dict[str, dict[str, str]]:
    """Get encrypted test data.
    currently from manually copied file benchmark/encrypted_llm_test_data.json
    TODO: automatically load test dataset (from github releases)?

    Returns:
        dict: keys are filenames and values are encrypted test data.
    """
    json_file_path = "./benchmark/encrypted_llm_test_data.json"
    with open(json_file_path, "r") as json_file:
        encrypted_test_data = json.load(json_file)
    return encrypted_test_data


def _decrypt_data(
    encrypted_test_data: dict, private_key: rsa.PrivateKey
) -> dict:
    """Decrypt the test data.

    Args:
        encrypted_test_data (dict): keys are filenames and values are encrypted
            test data.

        private_key (rsa.PrivateKey): The private key.

    Returns:
        dict: keys are filenames and values are decrypted test data.

    Todo: genericise to eval automatically
    """
    decrypted_test_data = {}
    for key in encrypted_test_data.keys():
        decrypted = _decrypt(encrypted_test_data[key], private_key)
        if key.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decrypted), sep=";")
            _apply_literal_eval(
                df,
                [
                    "entities",
                    "relationships",
                    "relationship_labels",
                    "properties",
                    "parts_of_query",
                ],
            )
            decrypted_test_data[key] = df
        elif key.endswith(".yaml"):
            decrypted_test_data[key] = yaml.safe_load(decrypted)
    return decrypted_test_data


def _decrypt(payload: dict[str, str], private_key: rsa.PrivateKey) -> str:
    """Decrypt a payload.

    Args:
        payload (dict[str, str]): Payload with key and data to decrypt.
        private_key (rsa.PrivateKey): Private key to decrypt the payload.

    Returns:
        str: Decrypted data.
    """
    enc_symmetric_key = b64decode(payload["key"])
    enc_data = b64decode(payload["data"])
    symmetric_key = rsa.decrypt(enc_symmetric_key, private_key)
    f = Fernet(symmetric_key)
    data = f.decrypt(enc_data).decode("utf-8")
    return data


def _apply_literal_eval(df: pd.DataFrame, columns: list[str]):
    """Apply literal_eval to columns in a dataframe
    In the csv file lists and dicts are stored as strings.
    By calling literal_eval they are transformed to lists and dicts again.

    Args:
        df (pd.DataFrame): Dataframe.
        columns (list[str]): Columns to apply literal_eval to.
    """
    for col_name in columns:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(
                lambda x: literal_eval(x) if pd.notna(x) else {}
            )


def _get_all_files(directory: str) -> list[str]:
    """Get all files in a directory.

    Args:
        directory (str): Path to directory.

    Returns:
        list[str]: List of file paths.
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files
