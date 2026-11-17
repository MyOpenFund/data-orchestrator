"""Issue #8, production side: one loader, run first, at the CLI entry point."""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import DOTENV_KEYS

ROOT = Path(__file__).resolve().parents[1]


def _run(cwd, code, *, path=ROOT, extra_env=None):
    """Run ``code`` in a fresh interpreter, ``path`` PREPENDED to PYTHONPATH
    (an existing PYTHONPATH, e.g. CI's, is kept after it)."""
    env = {k: v for k, v in os.environ.items() if k not in DOTENV_KEYS}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(path), os.environ.get("PYTHONPATH", "")) if p)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=120)


def _minimal_cb_corpus(tmp_path):
    root = tmp_path / "cbc"
    (root / "raw").mkdir(parents=True)
    (root / "manifest").mkdir()
    (root / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    return root


def _cb_count_code(root):
    return f"""
        import sys
        from data_orchestrator import cli
        print("module:", cli.__file__)
        sys.exit(cli.main(["cb_corpus", "--no-vault", "--count-only", "--root", {str(root)!r}]))
    """


def _cb_run_code(root):
    """A real (non-count) cb_corpus --no-vault run on an empty corpus: the
    engine's REAL QdrantStore is constructed from the settings read after the
    ``.env`` was loaded; only the embedder and the collection check are faked
    (nothing listens on the port) — so the ``Qdrant:`` line is what the engine
    resolved."""
    return f"""
        import sys
        import warnings
        warnings.simplefilter("ignore")  # qdrant-client: no server to ask its version
        from rag_engine.embedding import FakeEmbedder, ModelSpec
        from rag_engine.store import QdrantStore
        from data_orchestrator import cli, core
        print("module:", cli.__file__)
        core.build_embedder = lambda settings: FakeEmbedder(
            ModelSpec("fake", 8, "passage: ", "query: "))
        QdrantStore.ensure_collection = lambda self, name, dim: None
        sys.exit(cli.main(["cb_corpus", "--no-vault", "--root", {str(root)!r}]))
    """


def _qdrant_lines(out):
    return [line for line in out.stdout.splitlines() if line.startswith("Qdrant:")]


def _ocr_lines(out):
    return [line for line in out.stdout.splitlines() if line.startswith("OCR:")]


HEAVY = ("torch", "sentence_transformers", "chunknorris", "fitz", "pymupdf", "markitdown")


def test_importing_the_cli_loads_no_backend_and_no_parser(tmp_path):
    """The engine imports its embedding backend and its parsers on first use;
    importing the CLI (``--count-only``, ``--help``) must stay light."""
    out = _run(tmp_path, f"""
        import sys
        import data_orchestrator.cli
        print(sorted(m for m in {HEAVY!r} if m in sys.modules))
    """)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_importing_the_package_reads_no_dotenv_and_no_engine_setting(tmp_path):
    """Nothing is read at import, by this repo or by the engine: a ``.env`` in
    the cwd and an invalid engine variable are both inert until ``main()``."""
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(tmp_path, """
        import os
        import data_orchestrator.cli, data_orchestrator.core
        import rag_engine.config, rag_engine.ingest, rag_engine.store
        print("QDRANT_PORT" in os.environ, rag_engine.config.Settings.from_env({}).qdrant_port)
    """, extra_env={"RAG_ENGINE_BATCH_SIZE": "not-a-number"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "6333"]


def test_main_loads_the_dotenv_first_and_without_interpolation(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / ".env").write_text(
        f"CB_CORPUS_ROOT={tmp_path / 'corpus'}\n"
        "QDRANT_PORT=6433\n"
        "DATA_ORCHESTRATOR_PROBE=a$HOME\n", encoding="utf-8")
    out = _run(tmp_path, f"""
        import os, sys
        from data_orchestrator import cli
        def fake(args, collection):
            print(os.environ["QDRANT_PORT"], os.environ["DATA_ORCHESTRATOR_PROBE"],
                  sorted(m for m in {HEAVY!r} if m in sys.modules))
            return 0
        cli._count_vault_source = fake
        sys.exit(cli.main(["vault", "--count-only"]))
    """)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == f"Config: .env = {tmp_path / '.env'}"
    assert lines[-1].split() == ["6433", "a$HOME", "[]"]


def test_the_engine_settings_and_ocr_switches_come_from_the_dotenv(tmp_path):
    """The engine reads its settings, and ENABLE_OCR / TESSDATA_PREFIX at each
    file, from the process environment: the loader of ``main()`` must have put
    the ``.env`` values there before the run. A silent "auto -> never" (the
    OCR switches left out of the environment) bit this project once."""
    (tmp_path / ".env").write_text(
        "QDRANT_PORT=6433\n"
        "ENABLE_OCR=true\n"
        f"TESSDATA_PREFIX={tmp_path}\n"
        "RAG_ENGINE_OCR_LANGUAGES=eng+fra\n", encoding="utf-8")
    code = _cb_run_code(_minimal_cb_corpus(tmp_path)).replace(
        "sys.exit(", """from rag_engine.chunking import ocr_available
        real_run = cli.run_ingest
        def run(items, **kwargs):
            print("engine sees:", ocr_available(), kwargs["settings"].ocr_languages)
            return real_run(items, **kwargs)
        cli.run_ingest = run
        sys.exit(""", 1)
    out = _run(tmp_path, code,
               extra_env={"DATA_ORCHESTRATOR_STATE_DIR": str(tmp_path / "state")})
    assert out.returncode == 0, out.stderr
    assert _ocr_lines(out) == ["OCR: available (languages: eng+fra)"]
    assert "engine sees: True eng+fra" in out.stdout.splitlines()
    assert _qdrant_lines(out) == ["Qdrant: localhost:6433 (api key: unset)"]


def test_without_the_ocr_switches_the_run_says_ocr_is_off(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(tmp_path, _cb_run_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_STATE_DIR": str(tmp_path / "state")})
    assert out.returncode == 0, out.stderr
    assert _ocr_lines(out) == ["OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)"]


def test_the_real_environment_wins_over_the_dotenv_for_engine_settings(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\nQDRANT_HOST=from-dotenv\n",
                                   encoding="utf-8")
    out = _run(tmp_path, _cb_run_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"QDRANT_PORT": "7000",
                          "DATA_ORCHESTRATOR_STATE_DIR": str(tmp_path / "state")})
    assert out.returncode == 0, out.stderr
    assert _qdrant_lines(out) == ["Qdrant: from-dotenv:7000 (api key: unset)"]


def test_the_repo_root_fallback_is_used_from_an_unrelated_cwd(tmp_path):
    """The literal #8 scenario: a copy of the package with its repo-root
    ``.env``, the CLI launched from an unrelated folder (no
    DATA_ORCHESTRATOR_DOTENV): the repo-root file is the one read and the run
    — through the engine's real store — targets QDRANT_PORT=6433."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    above = [d / ".env" for d in (elsewhere, *elsewhere.parents) if (d / ".env").is_file()]
    if above:
        pytest.skip(f"an ancestor of the temp dir has a .env ({above[0]}): the cwd "
                    "search would find it before the repo-root fallback")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "data_orchestrator", repo / "data_orchestrator",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (repo / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(elsewhere, _cb_run_code(_minimal_cb_corpus(tmp_path)), path=repo,
               extra_env={"DATA_ORCHESTRATOR_STATE_DIR": str(tmp_path / "state")})
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == f"module: {repo / 'data_orchestrator' / 'cli.py'}"
    assert lines[1] == f"Config: .env = {repo / '.env'}"
    assert _qdrant_lines(out) == ["Qdrant: localhost:6433 (api key: unset)"]
    assert "warning:" not in out.stderr


def test_a_count_only_run_prints_no_qdrant_line(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(tmp_path, _cb_count_code(_minimal_cb_corpus(tmp_path)))
    assert out.returncode == 0, out.stderr
    assert _qdrant_lines(out) == []


def test_the_engine_loads_no_dotenv_of_its_own(tmp_path):
    """The one loader is ``main()``'s. The fork this repo used to drive ran
    python-dotenv at import; the engine does not: after a count-only ``main``
    (our loader ran), the engine's settings are exactly the process
    environment, and a second ``.env``-like file elsewhere is never read."""
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".env").write_text("QDRANT_PORT=6333\nQDRANT_HOST=other\n", encoding="utf-8")
    root = _minimal_cb_corpus(tmp_path)
    out = _run(tmp_path, f"""
        import os
        from data_orchestrator import cli, core
        assert cli.main(["cb_corpus", "--no-vault", "--count-only", "--root", {str(root)!r}]) == 0
        os.chdir({str(sub)!r})
        settings = core.engine_settings()
        print("target:", settings.qdrant_host, settings.qdrant_port)
    """)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "target: localhost 6433"


def test_an_explicit_dotenv_wins_over_the_cwd_one(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    explicit = tmp_path / "conf" / "prod.env"
    explicit.parent.mkdir()
    explicit.write_text("QDRANT_HOST=qdrant.lan\nQDRANT_PORT=7000\n", encoding="utf-8")
    out = _run(tmp_path, _cb_run_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_DOTENV": str(explicit),
                          "DATA_ORCHESTRATOR_STATE_DIR": str(tmp_path / "state")})
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[1] == f"Config: .env = {explicit}"
    assert _qdrant_lines(out) == ["Qdrant: qdrant.lan:7000 (api key: unset)"]


def test_a_relative_explicit_dotenv_is_printed_absolute(tmp_path):
    explicit = tmp_path / "conf" / "prod.env"
    explicit.parent.mkdir()
    explicit.write_text("QDRANT_PORT=7000\n", encoding="utf-8")
    out = _run(tmp_path, _cb_count_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_DOTENV": "conf/prod.env"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[1] == f"Config: .env = {explicit.resolve()}"


def test_an_explicit_dotenv_that_does_not_exist_exits_2_before_any_work(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(tmp_path, _cb_count_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_DOTENV": str(tmp_path / "missing.env")})
    assert out.returncode == 2
    assert out.stderr.startswith("error:") and "DATA_ORCHESTRATOR_DOTENV" in out.stderr
    assert "Config:" not in out.stdout and "Qdrant:" not in out.stdout
