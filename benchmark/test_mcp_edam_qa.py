"""Benchmark test for MCP-based question answering using EDAM ontology tools.

This test evaluates how well models can answer questions using MCP tools
from the EDAM ontology server.
"""

import inspect

import pytest

from biochatter.llm_connect.available_models import supports_tool_calling

from .benchmark_utils import (
    get_failure_mode_file_path,
    get_result_file_path,
    get_trace_file_path,
    skip_if_already_run,
    write_failure_modes_to_file,
    write_results_to_file,
    write_trace_to_file,
)
from .conftest import calculate_bool_vector_score

# System prompt to ensure consistent output format across MCP and baseline tests
EDAM_OUTPUT_FORMAT_PROMPT = (
    "You are answering questions about EDAM ontology concepts. "
    "Your response should contain only the EDAM ontology URIs (e.g., http://edamontology.org/data_2536) "
    "that match the question, separated by semicolons if multiple URIs are needed. "
    "Do not include any additional text, explanations, or formatting beyond the URIs."
)


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
    
    # Track failure mode information
    actual_answer = ""
    expected_answer = ""
    failure_mode = ""
    
    # Track tool call traces
    tool_schemas = {}
    tool_calls_trace = []
    tool_results_trace = []
    
    def run_test():
        nonlocal actual_answer
        nonlocal expected_answer
        nonlocal failure_mode
        nonlocal tool_schemas
        nonlocal tool_calls_trace
        nonlocal tool_results_trace
        
        mcp_conversation.reset()
        
        # Add system prompt to specify output format
        mcp_conversation.append_system_message(EDAM_OUTPUT_FORMAT_PROMPT)
        
        # Capture tool schemas (what the LLM sees)
        if hasattr(mcp_conversation, 'tools') and mcp_conversation.tools:
            for tool in mcp_conversation.tools:
                schema_info = {}
                if hasattr(tool, 'name'):
                    if hasattr(tool, 'args_schema'):
                        schema_info['args_schema'] = tool.args_schema
                    if hasattr(tool, 'input_schema'):
                        # Store schema type name for readability
                        schema_info['input_schema_type'] = str(type(tool.input_schema).__name__)
                    if hasattr(tool, 'description'):
                        schema_info['description'] = tool.description
                    tool_schemas[tool.name] = schema_info
        
        # Monkey-patch to capture tool calls and results
        original_process = mcp_conversation._process_tool_calls
        captured_calls = []
        captured_results = []
        
        def capture_tool_calls(tool_calls, *args, **kwargs):
            for tc in tool_calls:
                captured_calls.append({
                    "name": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })
            return original_process(tool_calls, *args, **kwargs)
        
        mcp_conversation._process_tool_calls = capture_tool_calls
        
        # Query with MCP tools available
        response, token_usage, _ = mcp_conversation.query(
            yaml_data["input"]["prompt"],
            mcp=True,
            track_tool_calls=True,  # Enable tool call tracking
        )
        
        # Capture tool results from messages after query completes
        from langchain_core.messages import ToolMessage
        if hasattr(mcp_conversation, 'messages'):
            for msg in mcp_conversation.messages:
                if isinstance(msg, ToolMessage):
                    # Extract result, truncate if too long
                    content = str(msg.content)
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    # Check if it's an error message
                    is_error = content.startswith("Error executing tool") or "Error" in content[:50]
                    captured_results.append({
                        "name": msg.name or "unknown",
                        "result": content if not is_error else None,
                        "error": content if is_error else None,
                    })
        
        # Store captured traces
        tool_calls_trace = captured_calls
        tool_results_trace = captured_results
        
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
        
        # Store expected answer for failure mode tracking
        expected_answer = yaml_data["expected"]["answer"]
        actual_answer = response
        
        # Normalize response for searching (case-insensitive)
        response_normalized = response.lower()
        
        # Score: check if each expected EDAM term appears in the response
        # Track which terms were found and which were missing
        score = []
        found_terms = []
        missing_terms = []
        
        for expected_term in expected_terms:
            # Check if the EDAM term (or its ID) appears in the response
            # Handle both full URLs and just the operation ID
            term_normalized = expected_term.lower().strip()
            
            # Try exact match first, then check if operation ID is present
            if term_normalized in response_normalized:
                score.append(True)
                found_terms.append(expected_term)
            else:
                # Extract operation ID (e.g., "operation_3694" from URL)
                operation_id = term_normalized.split("/")[-1] if "/" in term_normalized else term_normalized
                # Check if the operation ID appears in the response
                if operation_id in response_normalized:
                    score.append(True)
                    found_terms.append(expected_term)
                else:
                    score.append(False)
                    missing_terms.append(expected_term)
        
        # Determine failure mode if not all terms were found
        if len(missing_terms) > 0:
            total_terms = len(expected_terms)
            found_count = len(found_terms)
            missing_count = len(missing_terms)
            
            if found_count == 0:
                failure_mode = f"All Terms Missing ({missing_count}/{total_terms} missing)"
            elif missing_count == 1:
                failure_mode = f"Partial Match - Missing: {missing_terms[0]} ({found_count}/{total_terms} found)"
            else:
                # For multiple missing terms, list them concisely
                missing_short = [term.split("/")[-1] for term in missing_terms]
                failure_mode = f"Partial Match - Missing {missing_count} terms: {', '.join(missing_short[:3])}{'...' if len(missing_short) > 3 else ''} ({found_count}/{total_terms} found)"
        else:
            # All terms found - no failure
            failure_mode = ""
        
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
    
    # Write failure modes if there was a failure (not all terms found)
    if failure_mode and actual_answer:
        write_failure_modes_to_file(
            model_name,
            yaml_data["case"],
            actual_answer,
            expected_answer,
            failure_mode,
            yaml_data["hash"],
            get_failure_mode_file_path(task),
        )
    
    # Write trace information for all runs
    write_trace_to_file(
        model_name,
        yaml_data["case"],
        yaml_data["input"]["prompt"],
        actual_answer,
        tool_schemas,
        tool_calls_trace,
        tool_results_trace,
        yaml_data["hash"],
        get_trace_file_path(task),
    )


def test_mcp_edam_qa_baseline(
    model_name,
    test_data_mcp_edam_qa,
    baseline_conversation,
    multiple_testing,
):
    """Baseline test for MCP EDAM question answering (without MCP tools).
    
    This test evaluates the same questions as test_mcp_edam_qa but without
    MCP tools, providing a baseline for comparison. The model must answer
    questions about EDAM ontology concepts using only its training data.
    """
    yaml_data = test_data_mcp_edam_qa
    task = f"{inspect.currentframe().f_code.co_name.replace('test_', '')}"
    
    skip_if_already_run(
        model_name=model_name,
        task=task,
        md5_hash=yaml_data["hash"],
    )
    
    # Track failure mode information
    actual_answer = ""
    expected_answer = ""
    failure_mode = ""
    
    def run_test():
        nonlocal actual_answer
        nonlocal expected_answer
        nonlocal failure_mode
        
        baseline_conversation.reset()
        
        # Add system prompt to specify output format (same as MCP test for fairness)
        baseline_conversation.append_system_message(EDAM_OUTPUT_FORMAT_PROMPT)
        
        # Query without MCP tools (baseline)
        response, token_usage, _ = baseline_conversation.query(
            yaml_data["input"]["prompt"],
        )
        
        # Parse expected answer: semicolon-separated list of EDAM terms
        expected_terms = [
            term.strip() 
            for term in yaml_data["expected"]["answer"].split(";") 
            if term.strip()
        ]
        
        # Store expected answer for failure mode tracking
        expected_answer = yaml_data["expected"]["answer"]
        actual_answer = response
        
        # Normalize response for searching (case-insensitive)
        response_normalized = response.lower()
        
        # Score: check if each expected EDAM term appears in the response
        # Track which terms were found and which were missing
        score = []
        found_terms = []
        missing_terms = []
        
        for expected_term in expected_terms:
            # Check if the EDAM term (or its ID) appears in the response
            # Handle both full URLs and just the operation ID
            term_normalized = expected_term.lower().strip()
            
            # Try exact match first, then check if operation ID is present
            if term_normalized in response_normalized:
                score.append(True)
                found_terms.append(expected_term)
            else:
                # Extract operation ID (e.g., "operation_3694" from URL)
                operation_id = term_normalized.split("/")[-1] if "/" in term_normalized else term_normalized
                # Check if the operation ID appears in the response
                if operation_id in response_normalized:
                    score.append(True)
                    found_terms.append(expected_term)
                else:
                    score.append(False)
                    missing_terms.append(expected_term)
        
        # Determine failure mode if not all terms were found
        if len(missing_terms) > 0:
            total_terms = len(expected_terms)
            found_count = len(found_terms)
            missing_count = len(missing_terms)
            
            if found_count == 0:
                failure_mode = f"All Terms Missing ({missing_count}/{total_terms} missing)"
            elif missing_count == 1:
                failure_mode = f"Partial Match - Missing: {missing_terms[0]} ({found_count}/{total_terms} found)"
            else:
                # For multiple missing terms, list them concisely
                missing_short = [term.split("/")[-1] for term in missing_terms]
                failure_mode = f"Partial Match - Missing {missing_count} terms: {', '.join(missing_short[:3])}{'...' if len(missing_short) > 3 else ''} ({found_count}/{total_terms} found)"
        else:
            # All terms found - no failure
            failure_mode = ""
        
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
    
    # Write failure modes if there was a failure (not all terms found)
    if failure_mode and actual_answer:
        write_failure_modes_to_file(
            model_name,
            yaml_data["case"],
            actual_answer,
            expected_answer,
            failure_mode,
            yaml_data["hash"],
            get_failure_mode_file_path(task),
        )
    
    # Write trace information for baseline (no tools, so empty tool data)
    write_trace_to_file(
        model_name,
        yaml_data["case"],
        yaml_data["input"]["prompt"],
        actual_answer,
        {},  # No tool schemas for baseline
        [],  # No tool calls for baseline
        [],  # No tool results for baseline
        yaml_data["hash"],
        get_trace_file_path(task),
    )

