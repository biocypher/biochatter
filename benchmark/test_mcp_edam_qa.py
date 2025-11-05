"""Benchmark test for MCP-based question answering using EDAM ontology tools.

This test evaluates how well models can answer questions using MCP tools
from the EDAM ontology server.
"""

import inspect
import re

import pytest

from .benchmark_utils import (
    get_result_file_path,
    skip_if_already_run,
    write_results_to_file,
)
from .conftest import calculate_bool_vector_score


def test_mcp_edam_qa(
    model_name,
    test_data_mcp_edam_qa,
    mcp_conversation,
    multiple_testing,
):
    """Test MCP EDAM question answering capability.
    
    This test evaluates whether models can correctly answer questions
    using MCP tools from the EDAM ontology server.
    """
    yaml_data = test_data_mcp_edam_qa
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    
    skip_if_already_run(
        model_name=model_name,
        task=task,
        md5_hash=yaml_data["hash"],
    )
    
    # Skip if model doesn't support tool calling
    # Note: MCP requires models that support tool calling
    if "gpt-" not in model_name and model_name not in [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
    ]:
        pytest.skip(
            f"model {model_name} may not support MCP tool calling for {task} benchmark",
        )
    
    def run_test():
        mcp_conversation.reset()
        
        # Query with MCP tools available
        response, token_usage, _ = mcp_conversation.query(
            yaml_data["input"]["prompt"],
            mcp=True,
        )
        
        # Clean response for comparison
        response_clean = (
            response.lower()
            .replace(".", "")
            .replace("?", "")
            .replace("!", "")
            .replace(",", "")
            .strip()
        )
        expected_clean = (
            yaml_data["expected"]["answer"]
            .lower()
            .replace(".", "")
            .replace("?", "")
            .replace("!", "")
            .replace(",", "")
            .strip()
        )
        
        # Score: exact match
        score = [response_clean == expected_clean]
        
        return calculate_bool_vector_score(score)
    
    scores, max, n_iterations = multiple_testing(run_test)
    
    write_results_to_file(
        model_name,
        yaml_data["case"],
        f"{scores}/{max}",
        f"{n_iterations}",
        yaml_data["hash"],
        get_result_file_path(task),
    )

