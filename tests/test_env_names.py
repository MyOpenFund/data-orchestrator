"""The ``RAGO_*`` -> ``DATA_ORCHESTRATOR_*`` env rename and its compat window.

The prefix came from the repo's old name. Renaming it outright would silently
break every machine whose ``.env`` still says ``RAGO_*`` — silently, because an
unset env key is not an error here, it just falls back to the built-in default
(a *different* embedding model, hence a different collection name, hence an
apparently empty index). So the old names stay readable for one release and say
so loudly via ``DeprecationWarning``; these tests pin both halves of that deal,
and the mixed case that decides which name wins.
"""
import importlib
import warnings

import pytest

from data_orchestrator import cli, config, routing


def test_helper_returns_the_new_names_value(monkeypatch):
    """The new name is the one the code is supposed to read: with only it set,
    the helper must return its value and never consult the legacy key.
    """
    monkeypatch.setenv("DATA_ORCHESTRATOR_TEST_KEY", "new-value")
    monkeypatch.delenv("RAGO_TEST_KEY", raising=False)
    assert config.env_with_fallback(
        "DATA_ORCHESTRATOR_TEST_KEY", "RAGO_TEST_KEY"
    ) == "new-value"


def test_helper_returns_the_default_when_neither_name_is_set(monkeypatch):
    """The compat shim must not change the "nothing configured" outcome: with
    neither name set the caller's default still comes back, so ``STATE_DIR``
    and the embedding model keep their built-in values on a clean machine.
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_TEST_KEY", raising=False)
    monkeypatch.delenv("RAGO_TEST_KEY", raising=False)
    assert config.env_with_fallback(
        "DATA_ORCHESTRATOR_TEST_KEY", "RAGO_TEST_KEY", "fallback-default"
    ) == "fallback-default"
    assert config.env_with_fallback("DATA_ORCHESTRATOR_TEST_KEY", "RAGO_TEST_KEY") is None


def test_helper_falls_back_to_the_legacy_name_and_warns(monkeypatch):
    """The whole point of the compat window: an un-migrated ``.env`` keeps
    working (value returned) *and* the developer is told to migrate it
    (DeprecationWarning naming both keys). A silent fallback would let the old
    name survive past the one release it is granted.
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_TEST_KEY", raising=False)
    monkeypatch.setenv("RAGO_TEST_KEY", "legacy-value")

    with pytest.warns(DeprecationWarning) as caught:
        value = config.env_with_fallback("DATA_ORCHESTRATOR_TEST_KEY", "RAGO_TEST_KEY")

    assert value == "legacy-value"
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "RAGO_TEST_KEY" in message and "DATA_ORCHESTRATOR_TEST_KEY" in message


def test_helper_prefers_the_new_name_silently_when_both_are_set(monkeypatch):
    """A developer mid-migration has both names in their ``.env``. The new name
    must win (so the migration actually takes effect) and must warn about
    nothing: warning here would train people to ignore the warning that matters.
    """
    monkeypatch.setenv("DATA_ORCHESTRATOR_TEST_KEY", "new-value")
    monkeypatch.setenv("RAGO_TEST_KEY", "legacy-value")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        value = config.env_with_fallback("DATA_ORCHESTRATOR_TEST_KEY", "RAGO_TEST_KEY")

    assert value == "new-value"


def test_routing_reads_the_new_embedding_model_name(monkeypatch):
    """Driven through the real ``embedding_model_name()``, not the helper: the
    constant, the call site and the helper have to line up for the rename to
    reach the collection name that every ingest writes into.
    """
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "sentence-transformers/new")
    monkeypatch.delenv("RAGO_EMBEDDING_MODEL", raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert routing.embedding_model_name() == "sentence-transformers/new"

    assert routing.EMBEDDING_MODEL_ENV == "DATA_ORCHESTRATOR_EMBEDDING_MODEL"
    assert routing.LEGACY_EMBEDDING_MODEL_ENV == "RAGO_EMBEDDING_MODEL"


def test_routing_falls_back_to_the_legacy_embedding_model_name(monkeypatch):
    """CI and every existing ``.env`` still say ``RAGO_EMBEDDING_MODEL``; until
    they are migrated the routed collection name must not silently flip back to
    the default e5 model (that would index into a differently-named collection).
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("RAGO_EMBEDDING_MODEL", "sentence-transformers/paraphrase-albert-small-v2")

    with pytest.warns(DeprecationWarning, match="RAGO_EMBEDDING_MODEL is deprecated"):
        model = routing.embedding_model_name()

    assert model == "sentence-transformers/paraphrase-albert-small-v2"


def test_state_dir_honours_the_new_env_name(monkeypatch, tmp_path):
    """``STATE_DIR`` is resolved at import time, so the rename only lands if the
    module-level expression itself goes through the helper. Reloading the module
    under the new key is the only way to exercise that expression.
    """
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "new-state"))
    monkeypatch.delenv("RAGO_STATE_DIR", raising=False)

    try:
        importlib.reload(cli)
        assert cli.STATE_DIR == tmp_path / "new-state"
    finally:
        monkeypatch.delenv("DATA_ORCHESTRATOR_STATE_DIR", raising=False)
        importlib.reload(cli)  # restore the import-time default for later tests


def test_state_dir_falls_back_to_the_legacy_env_name(monkeypatch, tmp_path):
    """Same compat deal for the ledger directory: a machine still exporting
    ``RAGO_STATE_DIR`` must keep writing its ledger where it always did, rather
    than starting a fresh one under the repo and re-ingesting everything.
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_STATE_DIR", raising=False)
    monkeypatch.setenv("RAGO_STATE_DIR", str(tmp_path / "legacy-state"))

    try:
        with pytest.warns(DeprecationWarning, match="RAGO_STATE_DIR is deprecated"):
            importlib.reload(cli)
        assert cli.STATE_DIR == tmp_path / "legacy-state"
    finally:
        monkeypatch.delenv("RAGO_STATE_DIR", raising=False)
        importlib.reload(cli)
