"""Throwaway Postgres + Qdrant containers and adversarial corpus fixtures.

The Postgres schema replicates the vault's contract tables (documents +
rag_ingestions) at vault 12ffd11 (documents contract 1.0.0: identity
``(corpus, doc_id)``, ``rag_ingestions`` keyed ``(corpus, doc_id,
collection)``, the four contract columns) — the minimal columns this repo's
code touches. ``documents`` has ``document_url`` (formerly ``pdf_url``); a
test of a vault before that rename renames it back. Schema drift
against the real vault is guarded by the contract documented in the vault
README, not by importing vault code here.

Without Docker, the suite can run against services already started on this
machine (README, "Testing"): ``DATA_ORCHESTRATOR_IT_DATABASE_URL`` (an empty,
throwaway Postgres database: its tables are dropped and re-created),
``DATA_ORCHESTRATOR_IT_QDRANT`` (``host:port`` of a throwaway Qdrant without a
key: every collection on it is deleted) and ``DATA_ORCHESTRATOR_IT_QDRANT_KEYED``
(``host:port`` of a throwaway Qdrant started with ``IT_QDRANT_KEY`` as its API
key and ``IT_QDRANT_READ_ONLY_KEY`` as its read-only key). Never point them at
a Qdrant or a database that holds data.
"""
import os
import subprocess
import time
import uuid
from pathlib import Path

import psycopg2
import pytest

TINY_MODEL = "sentence-transformers/paraphrase-albert-small-v2"  # 768-d, small
TINY_MODEL_DIM = "768"  # any model but the default must declare its dimension

# The Qdrant the suite runs against: the version production runs, pinned so a
# new server release cannot change what these tests say (README, "Testing").
QDRANT_IMAGE = "qdrant/qdrant:v1.20.0"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    corpus TEXT NOT NULL,
    source_code TEXT,
    doc_type TEXT,
    title TEXT,
    document_url TEXT,
    source_url TEXT,
    date DATE,
    year INTEGER,
    language TEXT,
    provenance TEXT,
    mime_type TEXT,
    sha256 TEXT,
    local_path TEXT,
    date_precision TEXT,
    date_kind TEXT,
    entity_scheme TEXT,
    entity_value TEXT,
    last_seen_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    has_text_layer BOOLEAN,
    page_count INTEGER,
    extra JSONB,
    CONSTRAINT documents_corpus_doc_id_key UNIQUE (corpus, doc_id)
);
CREATE TABLE IF NOT EXISTS rag_ingestions (
    doc_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    corpus TEXT NOT NULL,
    source_code TEXT,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT,
    chunk_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT rag_ingestions_pkey PRIMARY KEY (corpus, doc_id, collection),
    CONSTRAINT rag_ingestions_corpus_doc_id_fkey FOREIGN KEY (corpus, doc_id)
        REFERENCES documents (corpus, doc_id)
);
-- The vault's transitional index (kept until the contract's cleanup), here so
-- that a run against this schema meets the same constraints as against the vault.
CREATE UNIQUE INDEX IF NOT EXISTS rag_ingestions_doc_collection_transition
    ON rag_ingestions (doc_id, collection);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    command TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    outcome TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    totals JSONB,
    sources JSONB,
    corpus TEXT,
    extra JSONB,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def docker_available():
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def _wait_port(probe, tries=60):
    for _ in range(tries):
        try:
            probe()
            return True
        except Exception:
            time.sleep(1)
    return False


@pytest.fixture(autouse=True)
def _no_repo_dotenv(tmp_path, monkeypatch):
    """Integration tests never read the repo ``.env`` (it may hold a real key).

    ``cli.main`` calls ``config.load_dotenv()`` once per process; pointing
    ``DATA_ORCHESTRATOR_DOTENV`` at an empty file and re-arming the loader makes
    that a no-op, and the dotenv-fed keys are cleared so nothing an earlier
    ``main`` leaked survives. Tests set their own env after this.
    """
    from data_orchestrator import config
    from tests.conftest import DOTENV_KEYS

    empty = tmp_path / "empty.env"
    empty.write_text("")
    for key in DOTENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_ORCHESTRATOR_DOTENV", str(empty))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", TINY_MODEL)
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_DIM", TINY_MODEL_DIM)
    monkeypatch.setattr(config, "_LOADED", False)
    monkeypatch.setattr(config, "_LOADED_PATH", None)
    yield


IT_DATABASE_URL_ENV = "DATA_ORCHESTRATOR_IT_DATABASE_URL"
IT_QDRANT_ENV = "DATA_ORCHESTRATOR_IT_QDRANT"
IT_QDRANT_KEYED_ENV = "DATA_ORCHESTRATOR_IT_QDRANT_KEYED"


def _external_addr(env):
    """``(host, port)`` of a service named by ``env`` (``host:port``), else ``None``."""
    value = (os.environ.get(env) or "").strip()
    if not value:
        return None
    host, _, port = value.rpartition(":")
    return host, int(port)


@pytest.fixture(scope="module")
def pg_url():
    external = (os.environ.get(IT_DATABASE_URL_ENV) or "").strip()
    if external:
        assert _wait_port(lambda: psycopg2.connect(external).close(), tries=5), "postgres not ready"
        yield external
        return
    if not docker_available():
        pytest.skip("docker unavailable")
    name = f"data-orchestrator-it-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-e", "POSTGRES_PASSWORD=it", "-e", "POSTGRES_USER=data_orchestrator",
         "-e", "POSTGRES_DB=documents", "-p", "127.0.0.1:0:5432", "postgres:16"],
        check=True, capture_output=True, timeout=300,
    )
    try:
        out = subprocess.run(["docker", "port", name, "5432/tcp"],
                             capture_output=True, text=True, check=True)
        port = out.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        url = f"postgresql://data_orchestrator:it@127.0.0.1:{port}/documents"
        assert _wait_port(lambda: psycopg2.connect(url).close()), "postgres not ready"
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(scope="module")
def qdrant_addr():
    external = _external_addr(IT_QDRANT_ENV)
    if external is not None:
        yield external
        return
    if not docker_available():
        pytest.skip("docker unavailable")
    from qdrant_client import QdrantClient
    name = f"data-orchestrator-it-qd-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-p", "127.0.0.1:0:6333", QDRANT_IMAGE],
        check=True, capture_output=True, timeout=300,
    )
    try:
        out = subprocess.run(["docker", "port", name, "6333/tcp"],
                             capture_output=True, text=True, check=True)
        port = int(out.stdout.strip().splitlines()[0].rsplit(":", 1)[1])
        assert _wait_port(
            lambda: QdrantClient(host="127.0.0.1", port=port, timeout=5).get_collections()
        ), "qdrant not ready"
        yield ("127.0.0.1", port)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


IT_QDRANT_KEY = "it-key-not-a-secret"
IT_QDRANT_READ_ONLY_KEY = "it-ro-key-not-a-secret"


@pytest.fixture(scope="module")
def qdrant_keyed_addr():
    """A Qdrant that requires ``IT_QDRANT_KEY`` and also accepts the read-only
    ``IT_QDRANT_READ_ONLY_KEY`` (the NAS setup in miniature)."""
    external = _external_addr(IT_QDRANT_KEYED_ENV)
    if external is not None:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=external[0], port=external[1], https=False,
                              api_key=IT_QDRANT_KEY)
        for coll in client.get_collections().collections:  # a fresh server per module, as with Docker
            client.delete_collection(coll.name)
        yield external
        return
    if not docker_available():
        pytest.skip("docker unavailable")
    from qdrant_client import QdrantClient
    name = f"data-orchestrator-it-qdk-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-e", f"QDRANT__SERVICE__API_KEY={IT_QDRANT_KEY}",
         "-e", f"QDRANT__SERVICE__READ_ONLY_API_KEY={IT_QDRANT_READ_ONLY_KEY}",
         "-p", "127.0.0.1:0:6333", QDRANT_IMAGE],
        check=True, capture_output=True, timeout=300,
    )
    try:
        out = subprocess.run(["docker", "port", name, "6333/tcp"],
                             capture_output=True, text=True, check=True)
        port = int(out.stdout.strip().splitlines()[0].rsplit(":", 1)[1])
        assert _wait_port(
            lambda: QdrantClient(host="127.0.0.1", port=port, timeout=5, https=False,
                                 api_key=IT_QDRANT_KEY).get_collections()
        ), "qdrant not ready"
        yield ("127.0.0.1", port)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture()
def clean_pg(pg_url):
    """Fresh vault tables (Postgres only: for tests that never reach Qdrant)."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS rag_ingestions; DROP TABLE IF EXISTS documents; "
                    "DROP TABLE IF EXISTS runs;")
        cur.execute(SCHEMA_SQL)
    conn.close()
    return pg_url


@pytest.fixture()
def clean_state(clean_pg, qdrant_addr):
    """Fresh vault tables and an empty Qdrant between tests."""
    from qdrant_client import QdrantClient
    pg_url = clean_pg
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    for coll in client.get_collections().collections:
        client.delete_collection(coll.name)
    return pg_url


def insert_documents(pg_url, rows):
    """rows: dicts with doc_id (required) and any of the documents columns."""
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        for r in rows:
            cols = list(r.keys())
            cur.execute(
                f"INSERT INTO documents ({', '.join(cols)}) "
                f"VALUES ({', '.join(['%s'] * len(cols))})",
                [r[c] for c in cols],
            )
    conn.commit()
    conn.close()


def write_text_pdf(path):
    """A 2-page PDF with a real text layer (has_text_layer=True, page_count=2)."""
    import fitz  # pymupdf, a dependency of chunknorris (rag-engine[ingest])

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in (1, 2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Monetary policy page {i}. " * 30)
    doc.save(path)
    doc.close()


def write_empty_pdf(path):
    """A 1-page PDF with no text layer (has_text_layer=False, page_count=1)."""
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()


def write_note_md(path):
    """A markdown note with enough prose to survive chunking as non-empty.

    MarkdownChunker enforces a minimum chunk word count (15).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Note\n\n"
        "A markdown paragraph about inflation. Central banks track consumer "
        "prices closely and adjust policy rates when inflation drifts away "
        "from target. This note discusses how inflation expectations feed "
        "back into wage negotiations and long-run price stability.\n"
    )


@pytest.fixture()
def corpus_dir(tmp_path):
    """Adversarial mini-corpus: 2-page text PDF, empty PDF, markdown note.

    Files sit flat at the corpus root — this fixture backs the disk-source tests,
    which build SourceItem paths directly and never go through the vault's
    local_path/CB_CORPUS_ROOT resolver. Vault-source and probe end-to-end tests
    override this fixture locally with a realistic nested "raw/..." layout
    (see their own conftest-shadowing corpus_dir) to exercise that resolver.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    write_text_pdf(d / "text.pdf")
    write_empty_pdf(d / "empty.pdf")
    write_note_md(d / "note.md")
    return d
