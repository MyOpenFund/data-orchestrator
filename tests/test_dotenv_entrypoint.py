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


def test_importing_the_cli_does_not_import_eigenmind(tmp_path):
    out = _run(tmp_path, """
        import sys
        import data_orchestrator.cli
        print("eigenmind" in sys.modules)
    """)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_main_loads_the_dotenv_before_eigenmind_and_without_interpolation(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / ".env").write_text(
        f"CB_CORPUS_ROOT={tmp_path / 'corpus'}\n"
        "QDRANT_PORT=6433\n"
        "DATA_ORCHESTRATOR_PROBE=a$HOME\n", encoding="utf-8")
    out = _run(tmp_path, """
        import os, sys
        from data_orchestrator import cli
        def fake(args, collection):
            print(os.environ["QDRANT_PORT"], os.environ["DATA_ORCHESTRATOR_PROBE"],
                  "eigenmind" in sys.modules)
            return 0
        cli._count_vault_source = fake
        sys.exit(cli.main(["vault", "--count-only"]))
    """)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == f"Config: .env = {tmp_path / '.env'}"
    assert lines[-1].split() == ["6433", "a$HOME", "False"]


def test_the_repo_root_fallback_is_used_from_an_unrelated_cwd(tmp_path):
    """The literal #8 scenario: the CLI is launched from a folder with no
    ``.env`` anywhere above it; the ``.env`` next to the package's repo root
    is the one read (the editable-install layout)."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "data_orchestrator", repo / "data_orchestrator",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (repo / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = _run(elsewhere, _cb_count_code(_minimal_cb_corpus(tmp_path)), path=repo)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == f"module: {repo / 'data_orchestrator' / 'cli.py'}"
    assert lines[1] == f"Config: .env = {repo / '.env'}"
    assert lines[2] == "Qdrant: localhost:6433"
    assert "warning:" not in out.stderr


def test_ours_wins_over_eigenmind_s_loader(tmp_path):
    """Ours runs first with override=False, so a later python-dotenv load from
    the eigenmind side (here: an explicitly planted second file, since
    python-dotenv's own search from eigenmind's location finds nothing in CI)
    can only add keys: QDRANT_PORT from our file survives into eigenmind."""
    pytest.importorskip("dotenv")
    pytest.importorskip("eigenmind.config")
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    eigen_side = tmp_path / "eigenmind-side.env"
    eigen_side.write_text("QDRANT_PORT=6333\nEIGENMIND_SIDE_ONLY=added\n", encoding="utf-8")
    root = _minimal_cb_corpus(tmp_path)
    out = _run(tmp_path, f"""
        import os
        from data_orchestrator import cli
        assert cli.main(["cb_corpus", "--no-vault", "--count-only", "--root", {str(root)!r}]) == 0
        import dotenv
        dotenv.load_dotenv({str(eigen_side)!r}, override=False)
        import eigenmind.config
        print("port:", eigenmind.config.qdrant_port(), os.environ["EIGENMIND_SIDE_ONLY"])
    """)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "port: 6433 added"


def test_an_explicit_dotenv_wins_over_the_cwd_one(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    explicit = tmp_path / "conf" / "prod.env"
    explicit.parent.mkdir()
    explicit.write_text("QDRANT_HOST=qdrant.lan\nQDRANT_PORT=7000\n", encoding="utf-8")
    out = _run(tmp_path, _cb_count_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_DOTENV": str(explicit)})
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[1] == f"Config: .env = {explicit}"
    assert lines[2] == "Qdrant: qdrant.lan:7000"


def test_an_explicit_dotenv_that_does_not_exist_exits_2_before_any_work(tmp_path):
    (tmp_path / ".env").write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    out = _run(tmp_path, _cb_count_code(_minimal_cb_corpus(tmp_path)),
               extra_env={"DATA_ORCHESTRATOR_DOTENV": str(tmp_path / "missing.env")})
    assert out.returncode == 2
    assert out.stderr.startswith("error:") and "DATA_ORCHESTRATOR_DOTENV" in out.stderr
    assert "Config:" not in out.stdout and "Qdrant:" not in out.stdout
