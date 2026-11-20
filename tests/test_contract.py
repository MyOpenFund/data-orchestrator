"""The documents contract this repo runs with is the vault's.

``vault-contract`` is a runtime dependency pinned by commit in
``pyproject.toml``; CI checks out the vault repository at a ref of its own
(``.github/workflows/tests.yml``). Both must name the same commit, and the
package installed must be the checkout's ``contract/`` directory, file for
file: then the split this repo builds payloads with is the one the vault
ingests with (``ingestion/ingest.py`` imports the same package).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import vault_contract

from tools import check_engine_pin as pin

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
VAULT_CHECKOUT_ENV = "DATA_ORCHESTRATOR_VAULT_CHECKOUT"


def _contract_pin() -> str:
    return pin.pinned_commit(pin.PYPROJECT.read_text(encoding="utf-8"), pin.CONTRACT_DISTRIBUTION)


def _workflow_vault_ref() -> str:
    """The ``ref:`` of the step checking out MyOpenFund/vault."""
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"repository:\s*MyOpenFund/vault\s*\n(?:\s+\w[\w-]*:.*\n)*?\s+ref:\s*(\S+)",
                      text)
    assert match, "the vault checkout of the workflow names no ref"
    return match.group(1)


def test_the_workflow_checks_out_the_pinned_vault_commit():
    assert _workflow_vault_ref() == _contract_pin()


def _checkout() -> Path | None:
    value = os.environ.get(VAULT_CHECKOUT_ENV)
    if value:
        return Path(value)
    if os.environ.get("CI"):
        pytest.fail(f"{VAULT_CHECKOUT_ENV} is not set: CI must run the contract test "
                    "against its vault checkout")
    pytest.skip(f"{VAULT_CHECKOUT_ENV} names no vault checkout")
    return None


def _package_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def test_the_installed_contract_is_the_checkout_s():
    checkout = _checkout()
    source = checkout / "contract" / "src" / "vault_contract"
    assert source.is_dir(), f"no contract package in the vault checkout {checkout}"
    installed = Path(vault_contract.__file__).parent
    assert _package_files(installed) == _package_files(source)


def test_the_vault_ingests_with_the_same_contract():
    """The vault's ingestion splits and checks rows with the package (no
    field list of its own any more)."""
    checkout = _checkout()
    ingest = (checkout / "ingestion" / "ingest.py").read_text(encoding="utf-8")
    assert re.search(r"^from vault_contract import .*\bcheck_row\b", ingest, re.M)
    assert not re.search(r"^KNOWN_FIELDS\s*=", ingest, re.M)


def test_the_contract_is_the_version_this_repo_was_written_for():
    assert vault_contract.CONTRACT_NAME == "documents"
    assert vault_contract.CONTRACT_VERSION.split(".")[0] == "1"


def test_the_vault_ddl_is_the_schema_the_integration_suite_copies():
    """tests/integration/conftest.py copies the vault's tables by hand: the
    pinned vault's DDL must still have the key and the columns it copies."""
    from tests.integration.conftest import SCHEMA_SQL

    ddl = (_checkout() / "ingestion" / "ingest.py").read_text(encoding="utf-8")
    flat = " ".join(ddl.split())
    copy = " ".join(SCHEMA_SQL.split())
    for clause in ("PRIMARY KEY (corpus, doc_id, collection)",
                   "FOREIGN KEY (corpus, doc_id) REFERENCES documents (corpus, doc_id)",
                   "UNIQUE (corpus, doc_id)",
                   "CREATE UNIQUE INDEX IF NOT EXISTS rag_ingestions_doc_collection_transition "
                   "ON rag_ingestions (doc_id, collection)"):
        assert clause in flat, clause
        assert clause in copy, clause
    for column in ("date_precision", "date_kind", "entity_scheme", "entity_value"):
        assert re.search(rf"^\s+{column} TEXT,?$", ddl, re.M), column
        assert f"{column} TEXT" in copy, column
