import os
import pandas as pd
import yaml
from benchmark.load_dataset import _get_all_files, _get_yaml_data


def _update_hashes_in_results():
    """
    This function replaces the md5_hash in the benchmark results with the
    corresponding hash from the reference_hashes, matching the case in the
    subtask column. Do this only if the test has not changed, but for some
    reason the hash has changed without affecting the test. This is purely to
    avoid rerunning expensive benchmarks.
    """
    directory_path = "./benchmark/data"
    files_in_directory = _get_all_files(directory_path)

    for file_path in files_in_directory:
        if file_path.endswith(".yaml"):
            with open(file_path, "r") as stream:
                try:
                    yaml_data = yaml.safe_load(stream)
                    if "_data" in file_path:
                        yaml_data = _get_yaml_data(yaml_data)

                    # get all current hashes for comparison
                    current_hashes = set()
                    reference_hashes = {}
                    for key in yaml_data.keys():
                        if isinstance(yaml_data[key], list):
                            for i in range(len(yaml_data[key])):
                                if isinstance(yaml_data[key][i], dict):
                                    reference_hashes[
                                        yaml_data[key][i]["case"]
                                    ] = yaml_data[key][i]["hash"]
                                    current_hashes.add(
                                        yaml_data[key][i]["hash"]
                                    )

                    # get all result files
                    result_files = [
                        f"benchmark/results/{file}"
                        for file in os.listdir("benchmark/results")
                        if file.endswith(".csv")
                    ]

                    # update hashes in results
                    for file in result_files:
                        result_file = pd.read_csv(file, header=0)
                        # go through the rows and replace the md5_hash with the
                        # corresponding hash from reference_hashes, matching the
                        # case in the subtask column
                        for index, row in result_file.iterrows():
                            result_file.at[index, "md5_hash"] = (
                                reference_hashes[row["subtask"]]
                            )
                        result_file.to_csv(file, index=False)

                except yaml.YAMLError as exc:
                    print(exc)


if __name__ == "__main__":
    _update_hashes_in_results()
