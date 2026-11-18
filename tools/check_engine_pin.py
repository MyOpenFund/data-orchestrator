"""Check that the rag-engine installed is the commit pinned in pyproject.toml.

    python tools/check_engine_pin.py

Exit 0 when they are the same commit, 1 otherwise. Run by CI after the
install step. The installed commit is the one pip recorded when it installed
the package from the git URL (``direct_url.json`` in the distribution's
metadata, key ``vcs_info.commit_id``): pip writes it for a git dependency,
installed directly or as a dependency of this package. An install from a
local checkout (``pip install -e ../rag-engine``) or from an index records no
commit and fails this check: it says nothing about which commit is running.
"""
from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
DISTRIBUTION = "rag-engine"

# The dependency line: rag-engine[extras] @ git+<url>@<40 hex digits>
_PIN = re.compile(r'"rag-engine(?:\[[^\]]*\])?\s*@\s*git\+[^"@\s]+@([0-9a-f]{40})"')


def pinned_commit(pyproject_text: str) -> str:
    """The commit sha of the rag-engine dependency in ``pyproject_text``."""
    match = _PIN.search(pyproject_text)
    if match is None:
        raise ValueError("rag-engine is not pinned by a full commit sha in pyproject.toml")
    return match.group(1)


def installed_commit(distribution=None) -> str | None:
    """The git commit pip installed ``rag-engine`` from; ``None`` when it was
    not installed from a git URL (a local checkout, an index)."""
    if distribution is None:
        distribution = metadata.distribution(DISTRIBUTION)
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def check(pyproject_text: str, distribution=None) -> int:
    pinned = pinned_commit(pyproject_text)
    installed = installed_commit(distribution)
    if installed is None:
        print(f"error: rag-engine is not installed from a git commit (pinned: {pinned})",
              file=sys.stderr)
        return 1
    if installed != pinned:
        print(f"error: rag-engine installed at {installed}, pyproject.toml pins {pinned}",
              file=sys.stderr)
        return 1
    print(f"rag-engine installed at the pinned commit {pinned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(check(PYPROJECT.read_text(encoding="utf-8")))
