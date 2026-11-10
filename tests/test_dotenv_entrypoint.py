"""Issue #8, production side: one loader, run first, at the CLI entry point."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests.conftest import DOTENV_KEYS

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, code):
    env = {k: v for k, v in os.environ.items() if k not in DOTENV_KEYS}
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=120)


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
    assert out.stdout.split() == ["6433", "a$HOME", "False"]
