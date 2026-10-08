"""Управление сервером ComfyUI: проверка, остановка, прямой запуск.

Прямой запуск повторяет команду Comfy Desktop, но без оболочки — нажатие Start
в окне приложения не требуется. Это и нужно оркестратору.

Запуск::

    python bench\\comfy_control.py status
    python bench\\comfy_control.py stop
    python bench\\comfy_control.py start
    python bench\\comfy_control.py restart
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.comfy import ComfyClient, ComfyError  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.procutil import _comfy_pids, _terminate_tree  # noqa: E402


def status(client: ComfyClient) -> int:
    """Печатает состояние сервера и моделей."""
    if not client.is_alive():
        pid = metrics.pid_on_port(8188)
        print(f"сервер не отвечает; порт 8188 {'занят pid ' + str(pid) if pid else 'свободен'}")
        return 1
    stats = client.system_stats()
    device = (stats.get("devices") or [{}])[0]
    print(f"сервер отвечает: ComfyUI {(stats.get('system') or {}).get('comfyui_version')}")
    print(f"устройство: {device.get('name')}")
    print(f"VRAM свободно: {round((device.get('vram_free') or 0) / (1024 * 1024))} MB")
    check = client.check_models()
    if check["ok"]:
        print("все три модели пайплайна видны загрузчикам")
    else:
        print(f"НЕ найдены модели: {check['missing']}")
    return 0


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Управление сервером ComfyUI")
    parser.add_argument("action", choices=["status", "stop", "start", "restart"])
    parser.add_argument("--timeout", type=float, default=300.0, help="предел ожидания старта")
    parser.add_argument("--mode", default="direct", choices=["direct", "desktop"])
    args = parser.parse_args()

    client = ComfyClient()

    if args.action in ("stop", "restart"):
        pids = _comfy_pids()
        if not pids:
            print("сервер ComfyUI не запущен")
        else:
            print(f"останавливаю сервер: pid {pids}")
            started = time.time()
            _terminate_tree(pids, timeout_s=25.0)
            print(f"остановлен за {time.time() - started:.1f} c")
        if args.action == "stop":
            return 0

    if args.action in ("start", "restart"):
        if client.is_alive():
            print("сервер уже отвечает")
            return status(client)
        print(f"запускаю сервер напрямую (режим {args.mode}) ...")
        print("команда: " + " ".join(config.comfy_server_argv()))
        print(f"каталог: {config.COMFY_SERVER_CWD}")
        started = time.time()
        try:
            alive, waited = client.ensure_running(timeout_s=args.timeout, mode=args.mode)
        except ComfyError as exc:
            print(f"не удалось запустить: {exc}")
            return 1
        if not alive:
            print(f"сервер не поднялся за {waited:.0f} c; смотри {config.LOGS_DIR / 'comfyui_server.log'}")
            return 1
        print(f"сервер поднялся за {waited:.1f} c")
        return status(client)

    return status(client)


if __name__ == "__main__":
    raise SystemExit(main())
