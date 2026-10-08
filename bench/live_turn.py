"""Живой ход через веб-интерфейс: сквозная проверка всей связки.

Скрипт стучится в уже запущенный интерфейс на 8760, отправляет одну реплику и
ждёт, пока автомат пройдёт все состояния. Печатает журнал и тайминги.

Проверяется ровно то, ради чего всё собиралось: движок поднимается, модель
отвечает по протоколу с тегами, сцена ставится в очередь, VRAM переключается,
картинка рисуется, движок возвращается.

Запуск::

    python bench\\live_turn.py
    python bench\\live_turn.py --text "Я рисую меч на стене таверны."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

BASE = "http://127.0.0.1:8760"


def call(path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    """Запрос к интерфейсу.

    @param path: путь, начиная со слэша.
    @param body: тело POST-запроса; ``None`` — обычный GET.
    @returns: разобранный JSON.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} на {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"интерфейс недоступен на {BASE}: {exc.reason}") from exc


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Один ход через веб-интерфейс")
    parser.add_argument(
        "--text",
        default="Я вхожу в таверну и осматриваюсь по сторонам.",
        help="реплика игрока",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="предел ожидания хода, с")
    args = parser.parse_args()

    print("\n=== Живой ход через интерфейс ===\n")
    status = call("/api/status")
    print(f"до хода: состояние {status['state']}, "
          f"VRAM свободно {status['gpu']['free_mb']} MB, "
          f"движок {'активен' if status['engine']['healthy'] else 'остановлен'}, "
          f"ComfyUI {'активен' if status.get('comfyui', {}).get('alive') else 'нет'}")

    if not status.get("comfyui", {}).get("alive"):
        print("\nComfyUI не отвечает. Открой Comfy Desktop и нажми Start — иначе картинка не сгенерируется.")

    started = time.time()
    call("/api/chat", {"text": args.text})
    print(f"\nреплика отправлена: {args.text!r}\n")

    seen_lines = 0
    deadline = started + args.timeout
    while time.time() < deadline:
        time.sleep(2.0)
        doc = call("/api/status")
        lines = doc.get("log") or []
        for line in lines[seen_lines:]:
            print("  " + line)
        seen_lines = len(lines)
        task = doc.get("task") or {}
        if not task.get("running"):
            break
    else:
        print("\n!! ход не завершился за отведённое время")

    print("\n=== Результат ===")
    turn = (call("/api/status") or {}).get("last_turn") or {}
    if turn.get("errors"):
        print("ошибки:", turn["errors"])
    print("тайминги:", json.dumps(turn.get("timings") or {}, ensure_ascii=False))
    print(f"итого: {turn.get('total_s')} c")
    for image in turn.get("images") or []:
        if image.get("error"):
            print(f"  кадр сцены #{image.get('scene_id')}: ОШИБКА {image['error'][:160]}")
        else:
            print(f"  кадр сцены #{image.get('scene_id')}: {image.get('path')} "
                  f"({image.get('elapsed_s')} c, пик VRAM {image.get('peak_vram_mb')} MB)")

    history = call("/api/history")
    last_assistant = [t for t in history["turns"] if t["role"] == "assistant"]
    if last_assistant:
        print("\n--- ответ мастера ---")
        print(last_assistant[-1]["content"])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
