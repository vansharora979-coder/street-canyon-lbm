"""Config loading and reproducible result/metadata I/O.

Every result written to disk gets a sidecar ``*.meta.json`` capturing the
config snapshot, git SHA, UTC timestamp, platform, and key library versions, so
any result can be traced back to the exact code and parameters that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def git_sha(short: bool = False) -> str:
    """Current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "matplotlib", "pandas", "yaml"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not installed"
    return versions


def run_metadata(config: dict | None = None, extra: dict | None = None) -> dict:
    """Assemble a reproducibility metadata record.

    Note: timestamp uses wall-clock time and is therefore the one
    non-deterministic field; all *scientific* outputs are seed/grid determined.
    """
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "config": config or {},
    }
    if extra:
        meta["extra"] = extra
    return meta


def save_result(
    path: str | Path,
    payload: Any,
    config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a JSON result plus a ``<stem>.meta.json`` sidecar; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with open(meta_path, "w") as fh:
        json.dump(run_metadata(config, extra), fh, indent=2)
    return path


def _json_default(obj):
    # Make numpy arrays/scalars JSON-serialisable.
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
