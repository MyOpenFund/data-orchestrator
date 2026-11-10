"""The supported-type predicate of the vault selection, against a real Postgres.

The unit tests only see SQL text; this checks what Postgres actually does with
NULL mime/path, mime parameters, blank and tab-padded mimes, upper-case
extensions, a supported mime on a foreign extension and an unsupported mime
on a supported extension — and that the
selection and the unsupported count partition the candidates exactly.
"""
import psycopg2
import pytest

from data_orchestrator.sources.vault import build_selection_sql, count_unsupported

from .conftest import SCHEMA_SQL, insert_documents

pytestmark = pytest.mark.integration

CORPUS, COLLECTION = "central-bank", "central-bank-e5b-v1"

# (doc_id, source_code, mime_type, local_path, expected selected)
MATRIX = [
    ("null-mime-upper-pdf", "us", None, "data/raw/us/C1/2015/UPPER.PDF", True),
    ("null-mime-html", "us", None, "data/raw/us/C1/2015/page.html", False),
    ("plain-charset-txt", "ecb", "text/plain; charset=utf-8", "data/raw/ecb/a.txt", True),
    ("blank-mime-md", "ecb", "", "data/raw/ecb/note.md", True),
    ("tab-pdf-mime", "fr", "\tapplication/pdf", "data/raw/fr/x.pdf", True),
    ("pdf-mime-html-ext", "fr", "application/pdf", "data/raw/fr/x.html", False),
    ("pdf-mime-no-ext", "fr", "application/pdf", "data/raw/fr/noext", False),
    ("null-mime-null-path", "us", None, None, False),
    ("html-html", "ecb", "text/html", "data/raw/ecb/p.html", False),
    ("html-mime-pdf-ext", "ecb", "text/html", "data/raw/ecb/q.pdf", False),
    ("pdf-trailing-ws", "us", "APPLICATION/PDF ", "data/raw/us/y.pdf\n", True),
    ("no-source-unsupported", None, "text/html", "data/raw/x/p.html", False),
]


@pytest.fixture()
def seeded(pg_url):
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS rag_ingestions; DROP TABLE IF EXISTS documents; "
                    "DROP TABLE IF EXISTS runs;")
        cur.execute(SCHEMA_SQL)
    conn.close()
    rows = [{"doc_id": d, "corpus": CORPUS, "source_code": s, "mime_type": m,
             "local_path": p} for d, s, m, p, _ in MATRIX]
    # Not candidates: already ingested, soft-deleted, other corpus.
    rows += [
        {"doc_id": "ingested", "corpus": CORPUS, "source_code": "us",
         "mime_type": "application/pdf", "local_path": "a/ingested.pdf"},
        {"doc_id": "deleted", "corpus": CORPUS, "source_code": "us",
         "mime_type": "text/html", "local_path": "a/deleted.html",
         "deleted_at": "2026-01-01T00:00:00Z"},
        {"doc_id": "other-corpus", "corpus": "company", "source_code": "us",
         "mime_type": "text/html", "local_path": "a/o.html"},
    ]
    insert_documents(pg_url, rows)
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rag_ingestions (doc_id, collection, corpus, embedding_model, "
            "chunk_count) VALUES ('ingested', %s, %s, 'm', 1)", (COLLECTION, CORPUS))
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_predicate_matrix_partitions_the_candidates(seeded):
    sql, params = build_selection_sql(CORPUS, COLLECTION)
    with seeded.cursor() as cur:
        cur.execute(sql, params)
        selected = [row[0] for row in cur.fetchall()]
    assert selected == sorted(d for d, *_, ok in MATRIX if ok)

    unsupported = count_unsupported(seeded, CORPUS, COLLECTION)
    assert unsupported == {"ecb": 2, "fr": 2, "unknown": 1, "us": 2}

    assert len(selected) + sum(unsupported.values()) == len(MATRIX)


def test_narrowed_mimes_move_rows_to_unsupported(seeded):
    sql, params = build_selection_sql(CORPUS, COLLECTION, supported_mimes=("text/markdown",))
    with seeded.cursor() as cur:
        cur.execute(sql, params)
        selected = [row[0] for row in cur.fetchall()]
    assert selected == ["blank-mime-md"]
    unsupported = count_unsupported(seeded, CORPUS, COLLECTION,
                                    supported_mimes=("text/markdown",))
    # Every candidate but the markdown one, per source (NULL source -> unknown).
    assert unsupported == {"ecb": 3, "fr": 3, "unknown": 1, "us": 4}
    assert len(selected) + sum(unsupported.values()) == len(MATRIX)
