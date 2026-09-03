from pathlib import Path

import pytest

from data_orchestrator import routing


def test_default_embedding_model(monkeypatch):
    monkeypatch.delenv("RAGO_EMBEDDING_MODEL", raising=False)
    assert routing.embedding_model_name() == "intfloat/multilingual-e5-base"


def test_embedding_model_env_override(monkeypatch):
    monkeypatch.setenv("RAGO_EMBEDDING_MODEL", "sentence-transformers/tiny")
    assert routing.embedding_model_name() == "sentence-transformers/tiny"


def test_model_tag_known():
    assert routing.model_tag("intfloat/multilingual-e5-base") == "e5b"


def test_model_tag_unknown_is_sanitized_last_segment():
    assert routing.model_tag("org/Some_Model.v2") == "some-model-v2"


def test_collection_name_default_model(monkeypatch):
    monkeypatch.delenv("RAGO_EMBEDDING_MODEL", raising=False)
    assert routing.collection_name("central-bank") == "central-bank-e5b-v1"


def test_collection_name_explicit_version():
    name = routing.collection_name(
        "company", model_name="intfloat/multilingual-e5-base", version=3
    )
    assert name == "company-e5b-v3"


def test_routing_has_central_bank():
    route = routing.ROUTING["central-bank"]
    assert route.root_env_key == "CB_CORPUS_ROOT"


def test_corpus_root_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    assert routing.corpus_root("central-bank") == Path(tmp_path)


def test_corpus_root_unknown_corpus():
    with pytest.raises(KeyError):
        routing.corpus_root("no-such-corpus")


def test_corpus_root_unset_env(monkeypatch):
    monkeypatch.delenv("CB_CORPUS_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        routing.corpus_root("central-bank")


def test_resolve_local_path_strips_data_prefix_for_central_bank(monkeypatch, tmp_path):
    # CB_CORPUS_ROOT points at the data dir (contains raw/), but the vault's
    # local_path is repo-root-relative ("data/raw/..."); the resolver must
    # strip "data/" so the join lands on <root>/raw/... not <root>/data/raw/...
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    result = routing.resolve_local_path("central-bank", "data/raw/us/C1/2010/doc123.pdf")
    assert result == tmp_path / "raw" / "us" / "C1" / "2010" / "doc123.pdf"


def test_resolve_local_path_empty_strip_passes_through_unchanged(monkeypatch, tmp_path):
    monkeypatch.setitem(
        routing.ROUTING, "no-strip-corpus",
        routing.CorpusRoute(corpus="no-strip-corpus", root_env_key="NO_STRIP_ROOT"),
    )
    monkeypatch.setenv("NO_STRIP_ROOT", str(tmp_path))
    result = routing.resolve_local_path("no-strip-corpus", "data/raw/x.pdf")
    assert result == tmp_path / "data" / "raw" / "x.pdf"


def test_resolve_local_path_strip_only_matches_leading_prefix(monkeypatch, tmp_path):
    # "data/" appears mid-path here, not as a leading prefix — must be left alone.
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    result = routing.resolve_local_path("central-bank", "raw/us/data/2010/doc.pdf")
    assert result == tmp_path / "raw" / "us" / "data" / "2010" / "doc.pdf"
