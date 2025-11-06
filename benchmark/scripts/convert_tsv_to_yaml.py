"""Convert TSV question-answer files to BioChatter benchmark YAML format.

This script reads a TSV file with two columns (question, answer) and converts
it to the YAML format used by BioChatter benchmarks.
"""

import argparse
import csv
import re
from pathlib import Path

import yaml


def sanitize_case_name(text: str, index: int, case_prefix: str = "mcp_edam") -> str:
    """Generate a sanitized case name from text and index.

    Args:
    ----
        text: The question text to generate case name from
        index: The index of the question (1-based)
        case_prefix: Prefix for the case identifier

    Returns:
    -------
        A sanitized case identifier

    """
    # Extract first few meaningful words from question
    words = re.findall(r"\b\w+\b", text.lower())[:3]
    case_suffix = "_".join(words) if words else f"question_{index}"
    # Limit length and remove special chars
    case_suffix = re.sub(r"[^a-z0-9_]", "", case_suffix)[:50]
    return f"{case_prefix}:{case_suffix}_{index}"


def escape_yaml_string(text: str) -> str:
    """Escape special characters for YAML strings.

    Args:
    ----
        text: The text to escape

    Returns:
    -------
        Escaped YAML string

    """
    # Replace newlines with spaces for single-line prompts
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def convert_tsv_to_yaml(
    tsv_path: Path,
    output_path: Path,
    top_level_key: str = "mcp_edam_qa",
    case_prefix: str = "mcp_edam",
) -> None:
    """Convert TSV file to YAML benchmark format.

    Args:
    ----
        tsv_path: Path to input TSV file
        output_path: Path to output YAML file
        top_level_key: Top-level key in YAML (default: "mcp_edam_qa")
        case_prefix: Prefix for case identifiers

    """
    test_cases: list[dict] = []

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Expect columns: question, answer (or similar)
        # Handle different possible column names
        question_col = None
        answer_col = None

        for col in reader.fieldnames:
            col_lower = col.lower()
            if "question" in col_lower or "query" in col_lower:
                question_col = col
            elif "answer" in col_lower or "response" in col_lower:
                answer_col = col

        if not question_col or not answer_col:
            # Try to infer from first row or use first two columns
            if len(reader.fieldnames) >= 2:
                question_col = reader.fieldnames[0]
                answer_col = reader.fieldnames[1]
            else:
                raise ValueError(
                    f"Could not identify question and answer columns. " f"Found columns: {reader.fieldnames}"
                )

        for idx, row in enumerate(reader, start=1):
            question = row.get(question_col, "").strip()
            answer = row.get(answer_col, "").strip()

            if not question or not answer:
                print(f"Warning: Skipping row {idx} - empty question or answer")
                continue

            # Generate case name
            case_name = sanitize_case_name(question, idx, case_prefix)

            # Create test case structure
            test_case = {
                "case": case_name,
                "input": {
                    "prompt": escape_yaml_string(question),
                },
                "expected": {
                    "answer": escape_yaml_string(answer),
                },
            }

            test_cases.append(test_case)

    # Create YAML structure
    yaml_data = {top_level_key: test_cases}

    # Add header comment
    yaml_content = f"""# Top-level keys: benchmark modules
# Values: list of dictionaries, each containing a test case
#
# Test case keys:
# - input (for creating the test)
# - expected (for asserting outcomes and generating a score)
# - case (for categorizing the test case)
#
# This file was automatically generated from TSV format.
# Source: {tsv_path}

"""

    # Write YAML
    yaml_content += yaml.dump(
        yaml_data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=1000,  # Prevent line wrapping
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Converted {len(test_cases)} test cases from {tsv_path} to {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert TSV question-answer files to BioChatter benchmark YAML format"
    )
    parser.add_argument(
        "tsv_file",
        type=Path,
        help="Path to input TSV file (expected columns: question, answer)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output YAML file (default: benchmark/data/benchmark_mcp_edam_data.yaml)",
    )
    parser.add_argument(
        "--top-level-key",
        default="mcp_edam_qa",
        help="Top-level key in YAML output (default: mcp_edam_qa)",
    )
    parser.add_argument(
        "--case-prefix",
        default="mcp_edam",
        help="Prefix for case identifiers (default: mcp_edam)",
    )

    args = parser.parse_args()

    if not args.tsv_file.exists():
        parser.error(f"TSV file not found: {args.tsv_file}")

    if args.output is None:
        args.output = Path("benchmark/data/benchmark_mcp_edam_data.yaml")

    convert_tsv_to_yaml(
        args.tsv_file,
        args.output,
        args.top_level_key,
        args.case_prefix,
    )


if __name__ == "__main__":
    main()
