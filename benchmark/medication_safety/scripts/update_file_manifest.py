from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "file_manifest.csv"
TEXT_SUFFIXES = {".csv", ".md", ".py", ".txt", ".yaml"}


def manifest_payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES or path.name == ".gitattributes":
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for path in files:
            payload = manifest_payload(path)
            writer.writerow(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    print(f"Updated {MANIFEST} with {len(files)} files.")


if __name__ == "__main__":
    main()
