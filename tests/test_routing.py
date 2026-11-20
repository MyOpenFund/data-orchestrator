from pathlib import Path

import pytest

from data_orchestrator import routing


def test_default_embedding_model(monkeypatch):
    monkeypatch.delenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", raising=False)
    assert routing.embedding_model_name() == "intfloat/multilingual-e5-base"


def test_embedding_model_env_override(monkeypatch):
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "sentence-transformers/tiny")
    assert routing.embedding_model_name() == "sentence-transformers/tiny"


def test_model_tag_known():
    assert routing.model_tag("intfloat/multilingual-e5-base") == "e5b"


def test_model_tag_unknown_is_sanitized_last_segment():
    assert routing.model_tag("org/Some_Model.v2") == "some-model-v2"


def test_collection_name_default_model(monkeypatch):
    monkeypatch.delenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", raising=False)
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


@pytest.mark.parametrize("corpus", ["central-bank", "company"])
def test_a_data_prefix_is_refused_not_stripped(monkeypatch, tmp_path, corpus):
    """The pre-contract central-bank form (``data/raw/...``, relative to the
    cb_corpus repository) is refused as the vault refuses it
    (``is_bad_local_path`` with ``transition=False``): never stripped, never
    read."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(tmp_path))
    with pytest.raises(routing.BadLocalPath, match="documents contract refuses"):
        routing.resolve_local_path(corpus, "data/raw/us/C1/2010/doc123.pdf")


def test_a_data_segment_inside_the_path_is_an_ordinary_segment(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    result = routing.resolve_local_path("central-bank", "raw/us/data/2010/doc.pdf")
    assert result == tmp_path / "raw" / "us" / "data" / "2010" / "doc.pdf"


# ---------------------------------------------------------------------------
# Supported types: the extensions are the engine's, the mime mapping is ours.
# ---------------------------------------------------------------------------
def test_mime_extensions_cover_exactly_what_the_engine_reads():
    """Bumping the rag-engine pin to a release that reads a new format (or
    drops one) must fail here until ``routing._EXTENSION_MIME`` follows: an
    extension without a mime type is never offered to the engine."""
    from rag_engine.chunking import SUPPORTED_EXTENSIONS

    offered = [ext for exts in routing.MIME_EXTENSIONS.values() for ext in exts]
    assert sorted(offered) == sorted(SUPPORTED_EXTENSIONS)
    assert len(offered) == len(set(offered))
    assert set(routing._EXTENSION_MIME) == set(SUPPORTED_EXTENSIONS)


def test_mime_extensions_keep_their_documented_mapping():
    assert routing.MIME_EXTENSIONS == {
        "application/pdf": (".pdf",),
        "text/markdown": (".md",),
        "text/plain": (".txt",),
        "text/csv": (".csv",),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (".docx",),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (".xlsx",),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": (".pptx",),
    }
    assert routing.DEFAULT_SUPPORTED_MIME_TYPES == tuple(routing.MIME_EXTENSIONS)


def test_an_extension_the_engine_stops_reading_is_no_longer_offered(monkeypatch):
    monkeypatch.setattr(routing, "SUPPORTED_EXTENSIONS", (".pdf", ".md"))
    assert routing._mime_extensions() == {"application/pdf": (".pdf",),
                                          "text/markdown": (".md",)}


def test_the_engine_model_variable_is_the_fallback_of_ours(monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_MODEL", "org/engine-model")
    assert routing.embedding_model_name() == "org/engine-model"
    assert routing.collection_name("central-bank") == "central-bank-engine-model-v1"
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "org/ours")
    assert routing.embedding_model_name() == "org/ours"


def test_a_blank_model_variable_reads_as_unset(monkeypatch):
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "  ")
    assert routing.embedding_model_name() == routing.DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# The documents contract: local_path relative to the corpus data root
# ---------------------------------------------------------------------------
def test_the_company_route_joins_its_root_and_the_local_path(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(tmp_path))
    path = routing.resolve_local_path("company", "raw/0000320193/A1/2024/cf0e49e6e176a851.pdf")
    assert path == tmp_path / "raw" / "0000320193" / "A1" / "2024" / "cf0e49e6e176a851.pdf"


def test_a_central_bank_path_on_the_contract_is_joined_as_it_is(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    assert routing.resolve_local_path("central-bank", "raw/ecb/A1/2022/a.pdf") == \
        tmp_path / "raw" / "ecb" / "A1" / "2022" / "a.pdf"


def test_the_data_prefix_fallback_is_gone():
    assert not hasattr(routing, "TRANSITION_LOCAL_PATH_STRIP")
    for route in routing.ROUTING.values():
        assert not hasattr(route, "local_path_strip")


@pytest.mark.parametrize("bad", [
    "raw/../../etc/passwd", "../raw/a.pdf", "/Users/x/raw/a.pdf", "raw//a.pdf", "./raw/a.pdf",
    "raw\\a.pdf", "C:/raw/a.pdf", "file:///raw/a.pdf", "~/raw/a.pdf", "raw/a.pdf/",
    "raw/\u200ba.pdf", "data/", "data/../x.pdf",
])
@pytest.mark.parametrize("corpus", ["central-bank", "company"])
def test_a_local_path_the_contract_calls_bad_is_refused_before_any_read(
        monkeypatch, tmp_path, corpus, bad):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(tmp_path))
    with pytest.raises(routing.BadLocalPath, match="documents contract refuses"):
        routing.resolve_local_path(corpus, bad)


def test_a_bad_local_path_is_refused_even_without_a_root(monkeypatch):
    monkeypatch.delenv("COMPANY_CORPUS_ROOT", raising=False)
    with pytest.raises(routing.BadLocalPath):
        routing.resolve_local_path("company", "../x.pdf")


def test_a_path_leaving_the_root_through_a_symbolic_link_is_refused(monkeypatch, tmp_path):
    root = tmp_path / "root"
    (root / "raw").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "raw" / "link").symlink_to(outside)
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(root))
    with pytest.raises(routing.BadLocalPath, match="symbolic link"):
        routing.resolve_local_path("company", "raw/link/a.pdf")


def test_a_symbolic_link_inside_the_root_and_a_linked_root_are_followed(monkeypatch, tmp_path):
    real = tmp_path / "real"
    (real / "raw" / "a").mkdir(parents=True)
    (real / "raw" / "b").symlink_to(real / "raw" / "a")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real)
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(linked_root))
    assert routing.resolve_local_path("company", "raw/b/x.pdf") == linked_root / "raw" / "b" / "x.pdf"
