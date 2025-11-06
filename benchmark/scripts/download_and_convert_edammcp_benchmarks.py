"""Download and convert all qa.tsv files from edammcp benchmark subdirectories.

This script:
1. Discovers all qa.tsv files in subdirectories of the edammcp benchmark repository
2. Downloads each file
3. Converts them to YAML format using convert_tsv_to_yaml.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def get_benchmark_subdirectories(repo: str = "edamontology/edammcp", branch: str = "main") -> list[str]:
    """Get list of subdirectories in the benchmark folder.

    Uses GitHub API to discover subdirectories.

    Args:
    ----
        repo: Repository in format "org/repo"
        branch: Branch name

    Returns:
    -------
        List of subdirectory names

    """
    # Try GitHub API first (requires requests or urllib)
    api_url = f"https://api.github.com/repos/{repo}/contents/benchmark"

    try:
        if HAS_REQUESTS:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                contents = response.json()
                subdirs = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(subdirs)
        else:
            # Fallback to urllib
            req = Request(api_url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            with urlopen(req, timeout=10) as response:
                contents = json.loads(response.read())
                subdirs = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(subdirs)
    except (HTTPError, URLError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not fetch subdirectories from GitHub API: {e}")
        print("You may need to specify subdirectories manually with --subdirectories")

    return []


def download_qa_tsv(
    subdirectory: str, output_dir: Path, repo_base: str = "edamontology/edammcp", branch: str = "main"
) -> Path | None:
    """Download qa.tsv file from a subdirectory.

    Args:
    ----
        subdirectory: Name of the subdirectory
        output_dir: Directory to save the downloaded file
        repo_base: Repository in format "org/repo"
        branch: Branch name

    Returns:
    -------
        Path to downloaded file, or None if download failed

    """
    # Construct raw GitHub URL
    raw_url = f"https://raw.githubusercontent.com/{repo_base}/{branch}/benchmark/{subdirectory}/qa.tsv"

    output_file = output_dir / f"{subdirectory}_qa.tsv"

    try:
        print(f"Downloading {raw_url}...")

        if HAS_REQUESTS:
            response = requests.get(raw_url, timeout=30)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                print(f"  ✗ File not found: {raw_url}")
                return None
            else:
                print(f"  ✗ Failed to download: HTTP {response.status_code}")
                return None
        else:
            # Fallback to urllib
            try:
                with urlopen(raw_url, timeout=30) as response:
                    content = response.read().decode("utf-8")
            except HTTPError as e:
                if e.code == 404:
                    print(f"  ✗ File not found: {raw_url}")
                    return None
                else:
                    print(f"  ✗ Failed to download: HTTP {e.code}")
                    return None

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        print(f"  ✓ Downloaded to {output_file}")
        return output_file
    except Exception as e:
        print(f"  ✗ Error downloading {raw_url}: {e}")
        return None


def convert_tsv_to_yaml(tsv_file: Path, output_dir: Path, subdirectory: str) -> Path | None:
    """Convert a TSV file to YAML format.

    Args:
    ----
        tsv_file: Path to the TSV file
        output_dir: Directory to save the YAML file
        subdirectory: Name of the subdirectory (used for naming)

    Returns:
    -------
        Path to converted YAML file, or None if conversion failed

    """
    # Get the conversion script path
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_tsv_to_yaml.py"

    # Generate output filename with edam_mcp prefix to indicate topic
    output_file = output_dir / f"benchmark_edam_mcp_{subdirectory}_data.yaml"

    try:
        print(f"Converting {tsv_file.name}...")
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(convert_script),
                str(tsv_file),
                "-o",
                str(output_file),
                "--top-level-key",
                "mcp_edam_qa",
                "--case-prefix",
                f"mcp_edam_{subdirectory}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✓ Converted to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Conversion failed: {e.stderr}")
        return None
    except Exception as e:
        print(f"  ✗ Error during conversion: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and convert all qa.tsv files from edammcp benchmark subdirectories"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("benchmark/data/downloaded"),
        help="Directory to download TSV files (default: benchmark/data/downloaded)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/data"),
        help="Directory to save converted YAML files (default: benchmark/data)",
    )
    parser.add_argument(
        "--repo",
        default="edamontology/edammcp",
        help="Repository in format 'org/repo' (default: edamontology/edammcp)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name (default: main)",
    )
    parser.add_argument(
        "--subdirectories",
        nargs="+",
        default=None,
        help="Specific subdirectories to process (default: auto-discover all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, only convert existing TSV files",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion step, only download TSV files",
    )

    args = parser.parse_args()

    # Create output directories
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of subdirectories
    if args.subdirectories:
        subdirectories = args.subdirectories
        print(f"Processing specified subdirectories: {', '.join(subdirectories)}")
    else:
        print("Discovering subdirectories from GitHub...")
        subdirectories = get_benchmark_subdirectories(args.repo, args.branch)

        if not subdirectories:
            print("Warning: Could not auto-discover subdirectories.")
            print("You can specify them manually with --subdirectories")
            print("\nExample:")
            print("  python download_and_convert_edammcp_benchmarks.py --subdirectories subdir1 subdir2")
            return 1

        print(f"Found {len(subdirectories)} subdirectories: {', '.join(subdirectories)}")

    # Download and convert each file
    downloaded_files = []
    converted_files = []

    for subdir in subdirectories:
        print(f"\n{'='*60}")
        print(f"Processing subdirectory: {subdir}")
        print(f"{'='*60}")

        # Download
        if not args.skip_download:
            tsv_file = download_qa_tsv(subdir, args.download_dir, args.repo, args.branch)
            if tsv_file:
                downloaded_files.append(tsv_file)
            else:
                print(f"  Skipping conversion for {subdir} (download failed)")
                continue
        else:
            # Look for existing file
            tsv_file = args.download_dir / f"{subdir}_qa.tsv"
            if not tsv_file.exists():
                print(f"  File not found: {tsv_file}")
                continue

        # Convert
        if not args.skip_convert:
            yaml_file = convert_tsv_to_yaml(tsv_file, args.output_dir, subdir)
            if yaml_file:
                converted_files.append(yaml_file)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Downloaded: {len(downloaded_files)} files")
    print(f"Converted: {len(converted_files)} files")

    if converted_files:
        print("\nConverted YAML files:")
        for yaml_file in converted_files:
            print(f"  - {yaml_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
