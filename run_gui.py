"""Точка входа веб-интерфейса NovelForge.

Запуск::

    python run_gui.py
    python run_gui.py --port 8760 --no-browser

Сервер слушает только петлевой интерфейс: наружу интерфейс не выставляется.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import uvicorn

    from novel import config
    from novel.console import setup_console

    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Веб-интерфейс NovelForge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"NovelForge: {url}")
    print(f"каталог:    {ROOT}")
    print(f"база:       {config.DB_PATH}")

    if not args.no_browser:
        def open_later() -> None:
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=open_later, daemon=True).start()

    uvicorn.run("gui.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
