"""Запуск NovelForge с ярлыка: без окна консоли и с понятной ошибкой.

Обычный ``run_gui.py`` печатает всё в консоль. Ярлыку консоль не нужна, но
тогда непонятно, почему ничего не произошло: ``pythonw`` молча глотает ошибки.
Этот запускатель пишет вывод в журнал, а при падении показывает окно.

Если интерфейс уже запущен, второй раз его поднимать не нужно: проверяется
ответ на ``/api/status``, и в этом случае просто открывается браузер.

Запуск::

    pythonw launcher.py
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PORT = 8760
LOG_PATH = PROJECT / "logs" / "gui_start.log"
URL = f"http://127.0.0.1:{PORT}/"
TITLE = "NovelForge"


def already_running() -> bool:
    """Отвечает ли интерфейс на своём порту.

    @returns: ``True``, если интерфейс уже поднят.
    """
    try:
        with urllib.request.urlopen(f"{URL}api/status", timeout=3) as response:
            json.loads(response.read().decode("utf-8"))
        return True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def show_message(text: str, title: str = TITLE) -> None:
    """Показывает окно с сообщением: pythonw иначе молчит при падении.

    @param text: текст сообщения.
    @param title: заголовок окна.
    """
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except (AttributeError, OSError):
        pass


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("a", encoding="utf-8")
    log.write(f"\n=== запуск {time.strftime('%Y-%m-%d %H:%M:%S')} (основной) ===\n")
    log.flush()
    sys.stdout = log
    sys.stderr = log

    if already_running():
        log.write("интерфейс уже запущен, открываю браузер\n")
        webbrowser.open(URL)
        return 0

    try:
        sys.path.insert(0, str(PROJECT))
        # run_gui разбирает sys.argv сам, поэтому лишние ключи запускателя нужно
        # убрать: иначе argparse откажется работать.
        sys.argv = ["run_gui.py", "--port", str(PORT)]
        from run_gui import main as gui_main

        return gui_main()
    except Exception:  # noqa: BLE001 — запускатель обязан показать любую ошибку
        traceback.print_exc()
        log.flush()
        show_message(
            f"Не удалось запустить {TITLE}.\n\n"
            f"Подробности: {LOG_PATH}\n\n"
            f"Частая причина — занят порт {PORT} другим процессом."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
