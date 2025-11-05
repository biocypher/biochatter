"""Benchmark test for MCP-based question answering using EDAM ontology tools.

This test evaluates how well models can answer questions using MCP tools
from the EDAM ontology server.
"""

import inspect
import re
import warnings

import pytest

from biochatter.llm_connect.available_models import supports_tool_calling

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
    
    # MCP requires models that support native tool calling
    # Use the actual capability check to ensure we're testing compatible models
    if not supports_tool_calling(model_name):
        pytest.skip(
            f"model {model_name} does not support tool calling. "
            f"MCP benchmarks require models with native tool calling capability. "
            f"Use a model that supports tool calling (e.g., gpt-4o, gpt-4.1-mini, claude-3-7-sonnet-latest, etc.)",
        )
    
    # Additional validation: warn if we're using a model that might fall back to manual tool calling
    # This can happen if the model name matching fails unexpectedly
    if not hasattr(mcp_conversation, 'mcp') or not mcp_conversation.mcp:
        pytest.fail(
            f"MCP conversation was not properly initialized with mcp=True. "
            f"This indicates a configuration error in the benchmark setup."
        )
    
    def run_test():
        mcp_conversation.reset()
        
        # Query with MCP tools available
        response, token_usage, _ = mcp_conversation.query(
            yaml_data["input"]["prompt"],
            mcp=True,
        )
        
        # Critical validation: Check if the model fell back to manual tool calling
        # If tools_prompt is set, it means the model doesn't support native tool calling
        # and fell back to manual mode, which is invalid for MCP benchmarks
        if hasattr(mcp_conversation, 'tools_prompt') and mcp_conversation.tools_prompt is not None:
            pytest.fail(
                f"CRITICAL: Model {model_name} fell back to MANUAL tool calling instead of NATIVE tool calling. "
                f"This indicates that supports_tool_calling('{model_name}') returned False during execution. "
                f"MCP benchmarks require native tool calling capability. "
                f"Possible causes:\n"
                f"  1. Model name matching failed (check supports_tool_calling function)\n"
                f"  2. Model was incorrectly classified as supporting tool calling\n"
                f"  3. Configuration error in conversation setup\n"
                f"Please verify the model supports tool calling and update model capability detection if needed."
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

