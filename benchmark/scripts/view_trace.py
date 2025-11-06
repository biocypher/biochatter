#!/usr/bin/env python3
"""View MCP tool call sequence from trace files.

Usage:
    python benchmark/scripts/view_trace.py benchmark/results/mcp_edam_qa_traces.jsonl
    python benchmark/scripts/view_trace.py benchmark/results/mcp_edam_qa_traces.jsonl --index 0
    python benchmark/scripts/view_trace.py benchmark/results/mcp_edam_qa_traces.jsonl --subtask "mcp_edam_dev:single_term_data_type_1"
"""

import json
import sys
from pathlib import Path


def view_trace(file_path: str, index: int | None = None, subtask: str | None = None):
    """View trace entries from a JSONL trace file.

    Args:
    ----
        file_path: Path to the JSONL trace file
        index: Optional index of the trace entry to view (0-based)
        subtask: Optional subtask name to filter by

    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    traces = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    if not traces:
        print(f"No traces found in {file_path}")
        sys.exit(1)

    # Filter by subtask if specified
    if subtask:
        traces = [t for t in traces if t.get("subtask") == subtask]
        if not traces:
            print(f"No traces found for subtask: {subtask}")
            sys.exit(1)

    # Select trace to display
    if index is not None:
        if index < 0 or index >= len(traces):
            print(f"Error: Index {index} out of range (0-{len(traces)-1})")
            sys.exit(1)
        traces_to_show = [traces[index]]
    else:
        traces_to_show = traces

    # Display each trace
    for i, trace in enumerate(traces_to_show):
        if len(traces_to_show) > 1:
            print(f"\n{'='*80}")
            print(f"Trace {i+1}/{len(traces_to_show)}")
            print(f"{'='*80}")

        print(f"\nModel: {trace.get('model_name', 'N/A')}")
        print(f"Subtask: {trace.get('subtask', 'N/A')}")
        print(f"Datetime: {trace.get('datetime', 'N/A')}")
        print(f"\nPrompt: {trace.get('prompt', 'N/A')}")

        # Show tool call sequence
        tool_calls = trace.get("tool_calls", [])
        tool_results = trace.get("tool_results", [])

        if not tool_calls:
            print("\nNo tool calls made.")
        else:
            print(f"\n{'='*80}")
            print("TOOL CALL SEQUENCE")
            print(f"{'='*80}")

            # Match calls with results by name (simple matching)
            results_by_name = {r["name"]: r for r in tool_results}

            for call_idx, call in enumerate(tool_calls, 1):
                tool_name = call.get("name", "unknown")
                call_id = call.get("id", "")
                args = call.get("args", {})

                print(f"\n{call_idx}. {tool_name}")
                if call_id:
                    print(f"   ID: {call_id}")
                print(f"   Args: {json.dumps(args, indent=6)}")

                # Show corresponding result
                if tool_name in results_by_name:
                    result = results_by_name[tool_name]
                    if result.get("error"):
                        error_msg = result["error"]
                        if len(error_msg) > 300:
                            error_msg = error_msg[:300] + "..."
                        print(f"   ❌ Error: {error_msg}")
                    else:
                        result_str = result.get("result", "")
                        is_truncated = "[truncated]" in result_str or "... [truncated]" in result_str

                        # Try to parse and pretty-print if it's JSON
                        try:
                            # Remove truncation marker if present for parsing
                            clean_result = result_str.replace("... [truncated]", "").replace("[truncated]", "")
                            result_json = json.loads(clean_result)
                            print(f"   ✓ Result{' (truncated)' if is_truncated else ''}:")
                            print(json.dumps(result_json, indent=8))
                        except:
                            if is_truncated:
                                print(f"   ✓ Result (truncated): {result_str}")
                            elif len(result_str) > 800:
                                print(f"   ✓ Result: {result_str[:800]}...")
                            else:
                                print(f"   ✓ Result: {result_str}")
                else:
                    print("   ⚠ No result found")

        print(f"\n{'='*80}")
        print("FINAL RESPONSE")
        print(f"{'='*80}")
        response = trace.get("response", "N/A")
        if len(response) > 500:
            print(f"{response[:500]}... [truncated]")
        else:
            print(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View MCP tool call sequences from trace files")
    parser.add_argument("file", help="Path to the JSONL trace file")
    parser.add_argument("--index", "-i", type=int, help="Index of the trace entry to view (0-based)")
    parser.add_argument("--subtask", "-s", help="Filter by subtask name")

    args = parser.parse_args()
    view_trace(args.file, args.index, args.subtask)
