"""Throwaway Postgres + Qdrant containers and adversarial corpus fixtures.

The Postgres schema replicates the vault's contract tables (documents +
rag_ingestions) — the minimal columns this repo's code touches. Schema drift
against the real vault is guarded by the contract documented in the vault
README, not by importing vault code here.
"""
import subprocess
import time
import uuid
from pathlib import Path

import psycopg2
import pytest

TINY_MODEL = "sentence-transformers/paraphrase-albert-small-v2"  # 768-d, small

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    corpus TEXT NOT NULL,
    source_code TEXT,
    doc_type TEXT,
    title TEXT,
    date DATE,
    year INTEGER,
    language TEXT,
    provenance TEXT,
    mime_type TEXT,
    sha256 TEXT,
    local_path TEXT,
    last_seen_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    has_text_layer BOOLEAN,
    page_count INTEGER,
    extra JSONB
);
CREATE TABLE IF NOT EXISTS rag_ingestions (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    collection TEXT NOT NULL,
    corpus TEXT NOT NULL,
    source_code TEXT,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT,
    chunk_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, collection)
);
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


@pytest.fixture(scope="session", autouse=True)
def tiny_model_env():
    import os
    os.environ.setdefault("RAGO_EMBEDDING_MODEL", TINY_MODEL)
    yield


@pytest.fixture(scope="module")
def pg_url():
    if not docker_available():
        pytest.skip("docker unavailable")
    name = f"rago-it-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-e", "POSTGRES_PASSWORD=it", "-e", "POSTGRES_USER=rago",
         "-e", "POSTGRES_DB=documents", "-p", "127.0.0.1:0:5432", "postgres:16"],
        check=True, capture_output=True, timeout=300,
    )
    try:
        out = subprocess.run(["docker", "port", name, "5432/tcp"],
                             capture_output=True, text=True, check=True)
        port = out.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        url = f"postgresql://rago:it@127.0.0.1:{port}/documents"
        assert _wait_port(lambda: psycopg2.connect(url).close()), "postgres not ready"
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(scope="module")
def qdrant_addr():
    if not docker_available():
        pytest.skip("docker unavailable")
    from qdrant_client import QdrantClient
    name = f"rago-it-qd-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-p", "127.0.0.1:0:6333", "qdrant/qdrant"],
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


@pytest.fixture()
def clean_state(pg_url, qdrant_addr):
    """Fresh vault tables and an empty Qdrant between tests."""
    from qdrant_client import QdrantClient
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS rag_ingestions; DROP TABLE IF EXISTS documents; "
                    "DROP TABLE IF EXISTS runs;")
        cur.execute(SCHEMA_SQL)
    conn.close()
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
    import fitz  # pymupdf, via the eigenmind dependency tree

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

    Files sit flat at the corpus root — this fixture backs wave-1 tests,
    which build SourceItem paths directly and never go through the vault's
    local_path/CB_CORPUS_ROOT resolver. Wave-2 and probe end-to-end tests
    override this fixture locally with a realistic nested "raw/..." layout
    (see their own conftest-shadowing corpus_dir) to exercise that resolver.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    write_text_pdf(d / "text.pdf")
    write_empty_pdf(d / "empty.pdf")
    write_note_md(d / "note.md")
    return d
