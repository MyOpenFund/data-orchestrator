"""Check that the git dependencies installed are the commits pinned in pyproject.toml.

    python tools/check_engine_pin.py

Two dependencies are pinned by commit: rag-engine (the engine) and
vault-contract (the vault's ``documents`` contract, directory ``contract/`` of
the vault repository). Exit 0 when each installed one is its pinned commit, 1
otherwise. Run by CI after the install step. The installed commit is the one
pip recorded when it installed the package from the git URL
(``direct_url.json`` in the distribution's metadata, key
``vcs_info.commit_id``): pip writes it for a git dependency, installed
directly or as a dependency of this package. An install from a local checkout
(``pip install -e ../rag-engine``) or from an index records no commit and
fails this check: it says nothing about which commit is running.
"""
from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
DISTRIBUTION = "rag-engine"
CONTRACT_DISTRIBUTION = "vault-contract"
#: The distributions pinned by commit, in the order they are checked.
DISTRIBUTIONS = (DISTRIBUTION, CONTRACT_DISTRIBUTION)


def _pin_pattern(distribution: str) -> re.Pattern:
    # The dependency line: <name>[extras] @ git+<url>@<40 hex digits>[#subdirectory=...]
    return re.compile(rf'"{re.escape(distribution)}(?:\[[^\]]*\])?\s*@\s*git\+[^"@\s]+@'
                      r'([0-9a-f]{40})(?:#[^"\s]*)?"')


def pinned_commit(pyproject_text: str, distribution: str = DISTRIBUTION) -> str:
    """The commit sha of the ``distribution`` dependency in ``pyproject_text``."""
    match = _pin_pattern(distribution).search(pyproject_text)
    if match is None:
        raise ValueError(f"{distribution} is not pinned by a full commit sha in pyproject.toml")
    return match.group(1)


def installed_commit(distribution=None, name: str = DISTRIBUTION) -> str | None:
    """The git commit pip installed distribution ``name`` from; ``None`` when
    it was not installed from a git URL (a local checkout, an index)."""
    if distribution is None:
        distribution = metadata.distribution(name)
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def check(pyproject_text: str, distribution=None, name: str = DISTRIBUTION) -> int:
    pinned = pinned_commit(pyproject_text, name)
    installed = installed_commit(distribution, name)
    if installed is None:
        print(f"error: {name} is not installed from a git commit (pinned: {pinned})",
              file=sys.stderr)
        return 1
    if installed != pinned:
        print(f"error: {name} installed at {installed}, pyproject.toml pins {pinned}",
              file=sys.stderr)
        return 1
    print(f"{name} installed at the pinned commit {pinned}")
    return 0


def check_all(pyproject_text: str) -> int:
    """:func:`check` for every pinned distribution; 1 if any fails."""
    return max(check(pyproject_text, name=name) for name in DISTRIBUTIONS)


if __name__ == "__main__":
    raise SystemExit(check_all(PYPROJECT.read_text(encoding="utf-8")))
