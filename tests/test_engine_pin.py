"""The check CI runs after installing: the rag-engine that is installed is the
commit pinned in ``pyproject.toml`` (``tools/check_engine_pin.py``)."""
import json
import re

import pytest

from tools import check_engine_pin as pin

SHA = "a" * 40
OTHER = "b" * 40


def _pyproject(sha=SHA):
    return ('[project]\nname = "x"\ndependencies = [\n    "psycopg2-binary>=2.9",\n'
            f'    "rag-engine[ingest,torch] @ git+https://github.com/MyOpenFund/rag-engine@{sha}",\n]\n')


class _Dist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _git(sha):
    return {"url": "https://github.com/MyOpenFund/rag-engine",
            "vcs_info": {"vcs": "git", "commit_id": sha, "requested_revision": sha}}


def test_the_pin_of_this_repository_is_a_full_commit_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", pin.pinned_commit(pin.PYPROJECT.read_text(encoding="utf-8")))


def test_pinned_commit_reads_the_sha_of_the_dependency_line():
    assert pin.pinned_commit(_pyproject()) == SHA


@pytest.mark.parametrize("text", [
    '[project]\ndependencies = ["rag-engine[ingest,torch]>=0.1"]\n',                        # not pinned
    '[project]\ndependencies = ["rag-engine @ git+https://github.com/MyOpenFund/rag-engine@main"]\n',  # a branch
    '[project]\ndependencies = ["rag-engine @ git+https://github.com/MyOpenFund/rag-engine@abc1234"]\n',  # short
])
def test_a_dependency_that_is_not_pinned_by_a_full_sha_is_an_error(text):
    with pytest.raises(ValueError, match="not pinned"):
        pin.pinned_commit(text)


def test_installed_commit_is_the_one_pip_recorded():
    assert pin.installed_commit(_Dist(_git(SHA))) == SHA


@pytest.mark.parametrize("direct_url", [
    None,                                                        # from an index: no direct_url.json
    {"url": "file:///src/rag-engine", "dir_info": {"editable": True}},  # a local checkout
])
def test_an_install_that_is_not_from_a_git_commit_has_no_commit(direct_url):
    assert pin.installed_commit(_Dist(direct_url)) is None


def test_check_passes_when_the_installed_commit_is_the_pinned_one(capsys):
    assert pin.check(_pyproject(), _Dist(_git(SHA))) == 0
    assert SHA in capsys.readouterr().out


def test_check_fails_on_another_commit(capsys):
    assert pin.check(_pyproject(), _Dist(_git(OTHER))) == 1
    err = capsys.readouterr().err
    assert SHA in err and OTHER in err


def test_check_fails_when_the_install_records_no_commit(capsys):
    assert pin.check(_pyproject(), _Dist({"url": "file:///src", "dir_info": {}})) == 1
    assert "not installed from a git commit" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# vault-contract, pinned the same way (a git subdirectory)
# ---------------------------------------------------------------------------
def _pyproject_with_contract(sha=SHA, contract=OTHER):
    return _pyproject(sha).replace(
        "]\n", f'    "vault-contract @ git+https://github.com/MyOpenFund/vault@{contract}'
               '#subdirectory=contract",\n]\n')


def test_the_contract_pin_of_this_repository_is_a_full_commit_sha():
    text = pin.PYPROJECT.read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{40}", pin.pinned_commit(text, pin.CONTRACT_DISTRIBUTION))


def test_each_distribution_has_its_own_pin():
    text = _pyproject_with_contract()
    assert pin.pinned_commit(text) == SHA
    assert pin.pinned_commit(text, "vault-contract") == OTHER


def test_check_reads_the_contract_pin_with_its_subdirectory(capsys):
    text = _pyproject_with_contract()
    assert pin.check(text, _Dist(_git(OTHER)), name="vault-contract") == 0
    assert pin.check(text, _Dist(_git(SHA)), name="vault-contract") == 1
    assert "vault-contract installed at" in capsys.readouterr().err


def test_check_all_checks_every_pinned_distribution():
    assert pin.DISTRIBUTIONS == ("rag-engine", "vault-contract")
