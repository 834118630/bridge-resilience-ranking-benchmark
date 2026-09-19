"""Provenance helpers for final E1 outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import scipy


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        Path(path).as_posix(): sha256_file(Path(path))
        for path in paths
        if Path(path).exists()
    }


def write_run_metadata(
    output_dir: Path,
    *,
    command: str,
    parameters: Mapping[str, object] | None = None,
    input_paths: Iterable[Path] = (),
    status: str = "complete",
    failure_status: str = "none",
) -> Path:
    """Write a complete provenance record for one final output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "run_metadata.json"
    ]
    metadata = {
        "status": status,
        "failure_status": failure_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "parameters": dict(parameters or {}),
        "input_hashes": _hashes(input_paths),
        "output_hashes": {path.as_posix(): sha256_file(path) for path in sorted(output_paths)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
