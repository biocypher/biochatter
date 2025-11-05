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
        
        # Parse expected answer: semicolon-separated list of EDAM terms
        expected_terms = [
            term.strip() 
            for term in yaml_data["expected"]["answer"].split(";") 
            if term.strip()
        ]
        
        # Normalize response for searching (case-insensitive)
        response_normalized = response.lower()
        
        # Score: check if each expected EDAM term appears in the response
        # This follows the same pattern as test_api_calling.py
        score = []
        for expected_term in expected_terms:
            # Check if the EDAM term (or its ID) appears in the response
            # Handle both full URLs and just the operation ID
            term_normalized = expected_term.lower().strip()
            
            # Try exact match first, then check if operation ID is present
            if term_normalized in response_normalized:
                score.append(True)
            else:
                # Extract operation ID (e.g., "operation_3694" from URL)
                operation_id = term_normalized.split("/")[-1] if "/" in term_normalized else term_normalized
                # Check if the operation ID appears in the response
                if operation_id in response_normalized:
                    score.append(True)
                else:
                    score.append(False)
        
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

