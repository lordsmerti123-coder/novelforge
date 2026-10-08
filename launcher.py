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
        with urllib.request.urlopen(f"{URL}api/status", timeout=12) as response:
            json.loads(response.read().decode("utf-8"))
        return True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def port_busy() -> bool:
    """Занят ли порт, даже если приложение не отвечает.

    Ответа на ``/api/status`` для этого мало: экземпляр может быть занят
    загрузкой модели и не ответить за отведённые секунды, оставаясь при этом
    хозяином порта. Подключение показывает занятость независимо от ответа.

    @returns: ``True``, если порт кем-то занят.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2.0)
        return probe.connect_ex(("127.0.0.1", PORT)) == 0


def ask_take_over() -> bool:
    """Спрашивает, закрывать ли прежний экземпляр.

    @returns: ``True``, если пользователь согласен на замену.
    """
    try:
        answer = ctypes.windll.user32.MessageBoxW(
            None,
            f"{TITLE} уже запущен, но не отвечает.\n\n"
            f"Закрыть прежний экземпляр и запустить заново?\n\n"
            f"Если ответить «Нет», откроется окно уже работающего приложения.",
            TITLE, 0x04 | 0x30,  # да/нет, значок вопроса
        )
        return answer == 6  # IDYES
    except (AttributeError, OSError):
        return False


def close_previous(log) -> int:
    """Закрывает прежние экземпляры приложения.

    Гасятся только процессы этого самого запускателя: чужие трогать не за что.

    @param log: открытый журнал.
    @returns: сколько процессов закрыто.
    """
    import subprocess

    script = (
        "Get-CimInstance Win32_Process -Filter \"name='pythonw.exe' OR name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*novelforge*launcher.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        log.write(f"закрыть прежние экземпляры не удалось: {exc}\n")
        return 0
    time.sleep(2)
    return 1


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

    if port_busy():
        # Порт занят, а приложение не отвечает: скорее всего, остался прежний
        # экземпляр. Молча занимать порт нельзя — отказ уйдёт в журнал, и
        # снаружи это выглядит как «ничего не произошло».
        log.write("порт занят, приложение не отвечает — спрашиваю пользователя\n")
        if ask_take_over():
            close_previous(log)
            for _ in range(20):
                if not port_busy():
                    break
                time.sleep(1)
        else:
            log.write("пользователь отказался, открываю браузер\n")
            webbrowser.open(URL)
            return 0
        if port_busy():
            log.write("порт всё ещё занят\n")
            show_message(
                f"Порт {PORT} занят другим процессом.\n\n"
                f"Закройте его или перезагрузите компьютер.\n"
                f"Подробности: {LOG_PATH}"
            )
            return 1

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
