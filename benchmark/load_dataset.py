import io
import json
import os
from ast import literal_eval
from base64 import b64decode

import pandas as pd
import rsa
import yaml
from cryptography.fernet import Fernet


def get_all_files(directory: str) -> list[str]:
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def decrypt(payload: dict, private_key: rsa.PrivateKey):
    enc_symmetric_key = b64decode(payload['key'])
    enc_data = b64decode(payload['data'])
    symmetric_key = rsa.decrypt(enc_symmetric_key, private_key)
    f = Fernet(symmetric_key)
    data = f.decrypt(enc_data).decode('utf-8')
    return data


def get_benchmark_dataset() -> dict:
    if os.environ.get('HOLD_OUT_TEST_DATA_PRIVATE_KEY'):
        test_data = load_hold_out_test_dataset()
    else:
        test_data = load_test_data_from_this_repository()

    return test_data


def load_hold_out_test_dataset():
    print("Use hold out test data for benchmarking")
    private_key = get_private_key_from_env_variable()
    encrypted_test_data = get_encrypted_test_data()
    decrypted_test_data = decrypt_data(encrypted_test_data, private_key)
    return decrypted_test_data


def get_encrypted_test_data():
    # TODO: automatically load test dataset (from github releases)?
    json_file_path = './benchmark/encrypted_llm_test_data.json'
    with open(json_file_path, 'r') as json_file:
        encrypted_test_data = json.load(json_file)
    return encrypted_test_data


def decrypt_data(encrypted_test_data, private_key):
    decrypted_test_data = {}
    for key in encrypted_test_data.keys():
        decrypted = decrypt(encrypted_test_data[key], private_key)
        if key.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decrypted), sep=";")
            apply_literal_eval(df)
            decrypted_test_data[key] = df
        elif key.endswith(".yaml"):
            decrypted_test_data[key] = yaml.safe_load(decrypted)
    return decrypted_test_data


def get_private_key_from_env_variable():
    private_key_base64 = os.environ.get('HOLD_OUT_TEST_DATA_PRIVATE_KEY')
    private_key_str = b64decode(private_key_base64).decode("utf-8")
    private_key = rsa.PrivateKey.load_pkcs1(private_key_str.encode("utf-8"))
    return private_key


def load_test_data_from_this_repository():
    print("Use public test data from this repository to test the benchmarking functionality")
    test_data = {}
    directory_path = "./benchmark/data"
    files_in_directory = get_all_files(directory_path)
    for file_path in files_in_directory:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, sep=";")
            apply_literal_eval(df)
            test_data[file_path.replace("./benchmark/", "./")] = df
        elif file_path.endswith(".yaml"):
            test_data[file_path.replace("./benchmark/", "./")] = yaml.safe_load(file_path)
    return test_data


def safe_literal_eval(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):
        return None


def apply_literal_eval(df):
    for col_name in ["entities", "relationships", "relationship_labels", "properties", "parts_of_query"]:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: safe_literal_eval(x) if pd.notna(x) else {})
