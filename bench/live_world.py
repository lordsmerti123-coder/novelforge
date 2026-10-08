"""Сквозная проверка нового пути: мир из заготовки, разбор, ход, отладка.

Прогон проверяет ровно то, что добавилось на этом этапе: создание мира из
пресета, структурирование описания силами модели, слоистую сборку контекста с
подсчётом токенов и запись хода в партию.

Картинки по умолчанию не рисуются — политика выставляется в ``manual``, иначе
прогон занимает лишние полторы минуты. Флаг ``--with-image`` возвращает обычное
поведение.

Запуск::

    python bench\\live_world.py
    python bench\\live_world.py --with-image --preset cyberpunk
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


def call(path: str, body: dict | None = None, timeout: float = 900.0) -> object:
    """Запрос к интерфейсу."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} на {path}: {exc.read().decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"интерфейс недоступен: {exc.reason}") from exc


def wait_idle(deadline_s: float, label: str) -> dict:
    """Ждёт завершения фоновой операции, печатая новые строки журнала."""
    seen = 0
    started = time.time()
    while time.time() - started < deadline_s:
        time.sleep(2.0)
        status = call("/api/status", timeout=60.0)
        lines = status.get("log") or []
        for line in lines[seen:]:
            print("   " + line)
        seen = len(lines)
        if not (status.get("task") or {}).get("running"):
            return status
    print(f"   !! {label}: не завершилось за {deadline_s} c")
    return call("/api/status", timeout=60.0)


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Сквозная проверка мира и хода")
    parser.add_argument("--preset", default="tavern", help="ключ заготовки мира")
    parser.add_argument("--with-image", action="store_true", help="разрешить генерацию картинки")
    parser.add_argument("--policy", default=None,
                        help="политика картинок на время проверки: minimal, master, manual и т. д.")
    parser.add_argument("--no-structure", action="store_true", help="не звать модель для разбора")
    parser.add_argument("--session", type=int, default=None, help="продолжить существующую партию")
    parser.add_argument(
        "--text", default="Я толкаю дверь и захожу внутрь.", help="реплика игрока"
    )
    args = parser.parse_args()

    print("\n=== Сквозная проверка ===")

    settings = call("/api/settings")
    policy = args.policy or ("on_scene_change" if args.with_image else "manual")
    call(
        "/api/settings",
        {"changes": {"image_policy": policy,
                     "engine_autostart": True, "debug_mode": True}},
    )
    print(f"\n1. Настройки: политика картинок -> {policy}")

    if args.session is not None:
        session_id = args.session
        session = call(f"/api/sessions/{session_id}/history")["session"]
        world_id = session["world_id"]
        doc = call(f"/api/worlds/{world_id}")
        print(f"2. Продолжаем мир #{world_id} «{doc['world']['name']}», партия #{session_id}")
        print(f"   правил {len(doc['rules'])}, персонажей {len(doc['characters'])}, "
              f"сообщений в партии {len(call(f'/api/sessions/{session_id}/history')['messages'])}")
    else:
        created = call("/api/worlds", {"name": "", "preset_key": args.preset})
        world_id, session_id = created["world_id"], created["session_id"]
        doc = call(f"/api/worlds/{world_id}")
        print(f"2. Мир #{world_id} из заготовки «{args.preset}»: правил {len(doc['rules'])}, "
              f"персонажей {len(doc['characters'])}, партия #{session_id}")

    if not args.no_structure and args.session is None:
        call(f"/api/worlds/{world_id}/structure/apply", {}, timeout=900.0)
        doc = call(f"/api/worlds/{world_id}")
        print(f"3. Разбор моделью: правил {len(doc['rules'])}, персонажей {len(doc['characters'])}")
        print(f"   жанр={doc['world']['genre']!r} тон={doc['world']['tone']!r}")
        print(f"   стиль={doc['world']['style']!r}")
    else:
        print("3. Разбор моделью пропущен")

    print(f"4. Ход: {args.text!r}")
    call("/api/chat", {"session_id": session_id, "text": args.text})
    status = wait_idle(600.0, "ход")

    print("\n5. Результат")
    history = call(f"/api/sessions/{session_id}/history")
    print(f"   сообщений: {len(history['messages'])}, сцен: {len(history['scenes'])}")
    assistant = [m for m in history["messages"] if m["role"] == "assistant"]
    if assistant:
        print("\n   --- ответ ведущего ---")
        print("   " + assistant[-1]["content"][:700].replace("\n", "\n   "))

    debug = call("/api/debug")
    prompt = debug.get("prompt") or {}
    if prompt:
        print("\n6. Отладка сборки контекста")
        print(f"   токенов {prompt['total_tokens']} "
              f"({'точно' if prompt['exact_tokens'] else 'оценка'}) "
              f"из бюджета {prompt['budget_tokens']}, "
              f"стабильных {prompt['stable_tokens']}, "
              f"сообщений в окне {prompt['included_messages']}, "
              f"выброшено {prompt['dropped_messages']}")
        for layer in prompt["layers"]:
            print(f"     {layer['title']:<28} {layer['tokens']:>6} ток. "
                  f"{'стабильный' if layer['stable'] else 'меняется'}")
        for warning in prompt.get("warnings") or []:
            print(f"     предупреждение: {warning}")
        usage = debug.get("usage") or {}
        print(f"   расход: вход {usage.get('prompt_tokens')} ток., "
              f"выход {usage.get('completion_tokens')} ток., "
              f"ttft {usage.get('ttft_s')} c, {usage.get('decode_tps')} ток/с")

    validation = call(f"/api/sessions/{session_id}/validate")
    print("\n7. Проверки")
    for issue in validation.get("issues", []):
        print(f"   [{issue['level']}] {issue['text']}")
        if issue.get("fix"):
            print(f"        -> {issue['fix']}")

    call("/api/settings", {"changes": {"image_policy": settings["image_policy"]}})
    print(f"\nНастройки возвращены: политика картинок {settings['image_policy']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
