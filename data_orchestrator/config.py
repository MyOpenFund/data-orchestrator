"""Tiny ``.env`` loader for per-machine configuration.

Different machines store the synced corpora in different folders, so the data
source roots live in a ``.env`` file (git-ignored) instead of being hard-coded.
``.env.example`` documents the available keys.

This is a self-contained loader (no python-dotenv dependency). It only sets keys
that are *not already present* in the real environment, so an explicit
``set CB_CORPUS_ROOT=...`` (or a CLI ``--root``) always wins.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

_LOADED = False


def env_with_fallback(new: str, old: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``new`` from the environment, falling back to a deprecated ``old`` name.

    The ``RAGO_*`` prefix predates this repo's rename to data-orchestrator. The
    old names keep working for one release so an un-migrated ``.env`` does not
    fail *silently* (an unset key here is not an error — it just reverts to a
    built-in default, e.g. a different embedding model and hence a different
    collection name). Reading ``old`` therefore emits a ``DeprecationWarning``.

    If both names are set the new one wins, silently: warning at that point
    would fire on every run of a machine that has already migrated but kept the
    old line around, and train people to ignore the warning that matters.
    """
    value = os.environ.get(new)
    if value is not None:
        return value
    legacy = os.environ.get(old)
    if legacy is not None:
        warnings.warn(
            f"{old} is deprecated, use {new}",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return default


def find_dotenv() -> Optional[Path]:
    """Locate the nearest ``.env``.

    Search order: the current working directory and its parents, then the repo
    root that ships this package (one level above ``data_orchestrator/``).
    """
    candidates = []
    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])
    candidates.append(Path(__file__).resolve().parents[1])  # repo root
    seen: set[Path] = set()
    for folder in candidates:
        if folder in seen:
            continue
        seen.add(folder)
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Optional[Path] = None, *, override: bool = False) -> bool:
    """Parse a ``.env`` file and populate ``os.environ``.

    Returns ``True`` if a file was found and read. Idempotent: only runs once
    per process unless an explicit ``path`` is given.
    """
    global _LOADED
    if path is None:
        if _LOADED:
            return False
        path = find_dotenv()
        _LOADED = True
    if path is None or not Path(path).is_file():
        return False

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def get_path(key: str, default: Optional[Path] = None) -> Optional[Path]:
    """Read a filesystem path from the environment (loading ``.env`` first)."""
    load_dotenv()
    value = os.environ.get(key)
    if value:
        return Path(value).expanduser()
    return default
