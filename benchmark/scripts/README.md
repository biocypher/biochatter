# Benchmark Scripts

This directory contains utility scripts for processing benchmark data.

## download_and_convert_edammcp_benchmarks.py

Automatically downloads and converts all `qa.tsv` files from subdirectories of the [edammcp benchmark repository](https://github.com/edamontology/edammcp/tree/main/benchmark).

### Usage

```bash
python benchmark/scripts/download_and_convert_edammcp_benchmarks.py [options]
```

### Options

- `--download-dir`: Directory to download TSV files (default: `benchmark/data/downloaded`)
- `--output-dir`: Directory to save converted YAML files (default: `benchmark/data`)
- `--repo`: Repository in format 'org/repo' (default: `edamontology/edammcp`)
- `--branch`: Branch name (default: `main`)
- `--subdirectories`: Specific subdirectories to process (default: auto-discover all)
- `--skip-download`: Skip download step, only convert existing TSV files
- `--skip-convert`: Skip conversion step, only download TSV files

### Examples

Download and convert all benchmarks automatically:

```bash
python benchmark/scripts/download_and_convert_edammcp_benchmarks.py
```

Process specific subdirectories:

```bash
python benchmark/scripts/download_and_convert_edammcp_benchmarks.py --subdirectories subdir1 subdir2
```

Only download (skip conversion):

```bash
python benchmark/scripts/download_and_convert_edammcp_benchmarks.py --skip-convert
```

Only convert existing files (skip download):

```bash
python benchmark/scripts/download_and_convert_edammcp_benchmarks.py --skip-download
```

### Notes

- The script uses GitHub API to discover subdirectories automatically
- Falls back to `urllib` if `requests` is not available
- Each subdirectory's `qa.tsv` is converted to a separate YAML file with appropriate naming

## convert_tsv_to_yaml.py

Converts TSV (tab-separated values) files with question-answer pairs into the YAML format used by BioChatter benchmarks.

### Usage

```bash
python benchmark/scripts/convert_tsv_to_yaml.py <tsv_file> [-o output_file] [options]
```

### Arguments

- `tsv_file`: Path to input TSV file (required)
  - Expected columns: question/query and answer/response
  - The script will automatically detect column names containing "question" or "query" for questions
  - The script will automatically detect column names containing "answer" or "response" for answers
- `-o, --output`: Path to output YAML file (optional)
  - Default: `benchmark/data/benchmark_mcp_edam_data.yaml`
- `--top-level-key`: Top-level key in YAML output (default: `mcp_edam_qa`)
- `--case-prefix`: Prefix for case identifiers (default: `mcp_edam`)

### Example

Convert a TSV file from the edammcp repository:

```bash
# Download the TSV file first
wget https://raw.githubusercontent.com/edamontology/edammcp/main/benchmark/qa.tsv

# Convert to YAML
python benchmark/scripts/convert_tsv_to_yaml.py qa.tsv -o benchmark/data/benchmark_mcp_edam_data.yaml
```

### Output Format

The script generates YAML files in the following format:

```yaml
mcp_edam_qa:
  - case: mcp_edam:question_1
    input:
      prompt: "What is the question?"
    expected:
      answer: "Expected answer"
  - case: mcp_edam:question_2
    ...
```

Each test case includes:
- `case`: A unique identifier for the test case
- `input.prompt`: The question to be asked
- `expected.answer`: The expected answer

