"""``python -m data_orchestrator.dashboard`` -> ``streamlit run app.py``."""
from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.stderr.write(
            "streamlit is not installed. Install the dashboard extra:\n"
            "    pip install -e '.[dashboard]'\n"
        )
        return 1
    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
