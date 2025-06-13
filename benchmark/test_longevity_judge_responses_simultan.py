import re
import time
import inspect
import ast
import os
import pandas as pd
from typing import List, Dict, Any, Tuple

import nltk
import pytest

from biochatter._misc import ensure_iterable
from .conftest import return_response, calculate_bool_vector_score, N_ITERATIONS
from .benchmark_utils import (
    benchmark_already_executed,
    skip_if_already_run,
    get_response_mode_file_path,
    write_responses_to_file,
    write_judgement_to_file,
    get_prompt_binary,
)

"""
TestLongevityResponseAndJudgement
================================

A test class that handles both response generation and judgment evaluation for 
longevity/geriatrics benchmark questions. Can run both stages or just judgment 
stage if responses already exist.

Flow:
1. Setup test data and configurations
2. Check if responses exist
   - If no: generate and store responses
   - If yes: load existing responses
3. Check if judgments exist
   - If no: run judgment evaluation
   - If yes: skip test
"""

class TestLongevityResponseAndJudgement:
    def setup_method(self):
        """
        Initialize test state:
        - Clear responses list
        - Set default iterations
        - etc.
        """
        # print("setup_method called")
        self.responses = []
        self.data = pd.DataFrame()
        self.ITERATIONS = 2 # defines number of rounds of judgement
    
    @pytest.mark.order(1)
    def test_generate_responses(
        self,
        model_name, 
        conversation,
        test_create_longevity_responses_simultaneously,
        multiple_responses,
    ):
        """
        Stage 1: Response Generation
        --------------------------
        - Skip if responses exist for (model, task, hash) combo
        - Generate multiple responses using conversation API
        - Store responses both in memory and to file
        - File format: CSV with metadata (model, task, timestamp, etc.)
        
        Returns: None (responses stored in self.responses and file)
        """
        print("test_generate_responses called")
        mode = "response"

        test_data = test_create_longevity_responses_simultaneously

        task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

        skip_if_already_run(
            model_name = model_name, 
            task = task, 
            md5_hash = test_data["hash"],
            mode = mode,
        )

        responses_run = []
        def run_test():
            conversation.reset()
            if "rag:" in test_data["case"]:
                pytest.skip(
                    f"test case {test_data['case']} not supported for {task} benchmark",
                )
            else:
                [
                    conversation.append_system_message(m)
                    for m in test_data["input"]["system_messages"]
                ]
                response, _, _ = conversation.query(test_data["input"]["prompt"])
                print(response)

            responses_run.append(response)

            return return_response(responses_run)

        n_iterations, responses = multiple_responses(run_test)
        self.responses.append(responses)

        write_responses_to_file(
            model_name,
            test_data["case_id"],
            test_data["case"],
            test_data["expected"]["individual"],
            test_data["input"]["prompt"],
            responses,
            test_data["expected"]["answer"][0],
            test_data["expected"]["summary"],
            test_data["expected"]["key_words"],
            test_data["type"],
            f"{n_iterations}",
            test_data["hash"],
            get_response_mode_file_path(task, model_name),
        )
        df = pd.read_csv(get_response_mode_file_path(task, model_name))
        self.data = df

    @pytest.mark.order(2)
    def test_generate_rag_responses(
        self,
        model_name, 
        conversation,
        test_create_longevity_responses_simultaneously,
        multiple_responses,
    ):
        """
        Stage 1: Response Generation
        --------------------------
        - Skip if responses exist for (model, task, hash) combo
        - Generate multiple responses using conversation API
        - Store responses both in memory and to file
        - File format: CSV with metadata (model, task, timestamp, etc.)
        
        Returns: None (responses stored in self.responses and file)
        """
        print("test_generate_responses called")
        mode = "response"

        test_data = test_create_longevity_responses_simultaneously

        task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

        skip_if_already_run(
            model_name = model_name, 
            task = task, 
            md5_hash = test_data["hash"],
            mode = mode,
        )

        responses_run = []
        def run_test():
            conversation.reset()
            if test_data["case"].startswith("rag:"):
                [
                    conversation.append_system_message(m.format(contexts = test_data["contexts"]))
                    for m in test_data["input"]["system_messages"]
                ]
                print(test_data["contexts"])
                response, _, _ = conversation.query(test_data["input"]["prompt"])
                print(response)
            else:
                pytest.skip(
                    f"test case {test_data['case']} not supported for {task} benchmark",
                )

            responses_run.append(response)

            return return_response(responses_run)

        n_iterations, responses = multiple_responses(run_test)
        self.responses.append(responses)

        write_responses_to_file(
            model_name,
            test_data["case_id"],
            test_data["case"],
            test_data["expected"]["individual"],
            test_data["input"]["prompt"],
            responses,
            test_data["expected"]["answer"][0],
            test_data["expected"]["summary"],
            test_data["expected"]["key_words"],
            test_data["type"],
            f"{n_iterations}",
            test_data["hash"],
            get_response_mode_file_path(task, model_name),
        )
        df = pd.read_csv(get_response_mode_file_path(task, model_name))
        self.data = df
        
    @pytest.mark.order(3)
    def test_judge_responses(
        self,
        judge_conversation, 
        judge_name, 
        judge_metric, 
    ):
        """
        Stage 2: Response Judgment
        ------------------------
        - Skip if judgments exist for (model, task, hash) combo
        - Load responses from file if not in memory
        - For each response:
            - Run multiple judgment iterations
            - Calculate agreement score
        - Store judgments to file
        
        Returns: None (judgments written to file)
        """
        print("test_judge_responses called")
        mode = "judge"

        if self.data.empty:
            self.data = self._load_responses(path="./benchmark/results/")
        test_data = self.data

        task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"

        system_message = "As a medical assistant, your task is to assess the  answering of a health-related query by an individual, where the answering is carried out by another LLM."
        judge_conversation.append_system_message(system_message)

        for data in test_data["judgement"]:
            if benchmark_already_executed(
                model_name = data["model_name"], 
                task = task, 
                md5_hash = data["md5_hash"],
                mode = mode,
                metric = judge_metric,
                judge_name = judge_name,
            ):
                print("\033[93mskipped\033[0m")
                continue
            print(f"\033[92m{judge_name} judges {data['model_name']} in {judge_metric} for {data['subtask']}\033[0m")

            all_time_scores = []
            for response in ast.literal_eval(data["response"]):

                params = {
                    "prompt": data["prompt"],
                    "summary": data["summary"],
                    "keywords": data["key_words"],
                    "response": response,
                }

                # if judge_metric == "correctness":
                #     params["expected_answer"] = data["expected_answer"]

                prompt, success, failure = self._format_judgment_prompt(
                    metric = judge_metric,
                    path = "./benchmark/prompts/prompts.yaml",
                    params = params,
                )

                scores = []
                for iter in range(self.ITERATIONS):
                    judge_conversation.reset() 
                    judgement, _, _ = judge_conversation.query(prompt)
                    # print(prompt)
                    print(judgement)
                    
                    if judgement.lower().replace(".", "") == success:
                        scores.append(True)
                    elif judgement.lower().replace(".", "") == failure:
                        scores.append(False)
                    else:
                        scores.append(False)
                all_time_scores.append(scores)
            score_string = self._calculate_judgment_scores(
                all_time_scores = all_time_scores,
            )
            
            _, prompt_type, is_distractor, system_prompt = data["subtask"].rsplit(":", 3)
            
            write_judgement_to_file(
                judge_model = judge_name,
                evaluated_model = data["model_name"],
                iterations = f"{N_ITERATIONS}",
                metric = judge_metric,
                case_id = data["case_id"],
                subtask = data["subtask"],
                individual = data["age"],
                md5_hash = data["md5_hash"],
                prompt = data["prompt"],
                system_prompt = system_prompt,
                prompt_type = prompt_type,
                is_distractor = is_distractor,
                type_ = data["type"],
                responses = data["response"],
                expected_answer = data["expected_answer"],
                rating = f"{score_string}/{1}",
                path = f"./benchmark/results/{task}.csv", # can be modified for separated file creation (one file for each tested model)
            )
            time.sleep(0.5)

    # Helper methods
    def _list_files(
        self, 
        path: str
    ) -> List[str]:
        """Lists files of the specified directory path"""
        files = []
        for file in os.listdir(path):
            if not file.startswith(".") and file.endswith("_response.csv"):
                files.append(file)
        return files

    def _load_responses(
        self, 
        path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load responses from file if they exist"""
        files = self._list_files(path = path)
        # latest_file = [max([f"{path}/{file}" for file in files], key = os.path.getmtime)]

        dfs = []
        for file in files: # or files if each response file should be judged
            file_path = os.path.join(path, file)
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as err:
                print(f"Error reading {file}: {err}")
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
        
    def _format_judgment_prompt(
        self, 
        metric: str,
        path: str,
        params: dict,
    ) -> Tuple[str, str, str]:
        """Format prompt for judge based on metric type"""
        prompts = get_prompt_binary(path = path)
        prompt = prompts[metric]["prompt"].format(
            **params,
        )
        success, failure = prompts[metric]["success"], prompts[metric]["failure"]
        return (prompt, success, failure)
        
    def _calculate_judgment_scores(
        self,
        all_time_scores,
    ) -> str:
        """Calculate agreement scores across iterations"""
        means = []
        for scores in all_time_scores:
            score = sum(scores)
            mean = score / len(scores)
            means.append(mean)
        score_string = ";".join([str(mean) for mean in means])
        return score_string

"""
Usage:
------
1. Full pipeline:
   pytest test_longevity_judge_responses_simultan.py

2. Just judgment (with existing responses):
   pytest test_longevity_judge_responses_simultan.py::TestLongevityResponseAndJudgement::test_judge_responses

Required changes to existing codebase:
------------------------------------
1. Modify skip_if_already_run() to handle separate response/judgment checks
2. Add response loading utility
3. Update file paths/naming conventions for better separation
4. Update test configurations for sequential running
"""