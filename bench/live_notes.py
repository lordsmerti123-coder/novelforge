"""Проверка заметок «задумано на потом» сквозным ходом.

Заводит заметку с проверяемым условием, делает ход и смотрит:

* попала ли заметка в промпт;
* прислал ли ведущий блок ``<notes>`` с её номером;
* закрылась ли она после этого.

Работает на одноразовом мире, настоящие партии не трогает.

Запуск::

    python bench\\live_notes.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402

BASE = "http://127.0.0.1:8760"
PASSED = 0
FAILED = 0

#: Условие нарочно простое: заметка должна сбыться в первом же ответе.
NOTE = "пусть в таверну войдёт стражник и потребует у игрока назвать имя"
TURN = "Я сажусь за стол у стены и жду, что будет."


def check(name: str, condition: bool, detail: str = "") -> None:
    """Печатает результат одной проверки."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  ФЕЙЛ {name}{'' if not detail else ' — ' + detail}")


def call(path: str, body: dict | None = None, timeout: float = 900.0):
    """Запрос к интерфейсу."""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else None


def main() -> int:
    setup_console()
    print("\n=== Заметки «задумано на потом» ===\n")

    settings = call("/api/settings")
    policy = settings["image_policy"]
    if policy != "manual":
        call("/api/settings", {"changes": {"image_policy": "manual"}})
        print(f"  политика картинок на время проверки: manual (была {policy})")
    keep = call("/api/settings").get("debug_keep_payloads")
    if not keep:
        call("/api/settings", {"changes": {"debug_keep_payloads": True}})

    world = call("/api/worlds", {"name": "", "preset_key": "tavern", "format": "story"})
    world_id = world["world_id"]
    # Создание мира отдаёт только его номер: партии берём отдельным запросом.
    session_id = call(f"/api/worlds/{world_id}")["sessions"][0]["id"]
    print(f"  полигон: мир #{world_id}, партия #{session_id}\n")

    notes = call(f"/api/sessions/{session_id}/notes", {"text": NOTE})
    note_id = notes["note_id"]
    check("заметка заведена", note_id > 0, str(note_id))

    db = NovelDB()
    active = db.notes(session_id)
    db.close()
    check("заметка активна", len(active) == 1 and active[0].text == NOTE,
          str([n.text for n in active]))

    # Слой заметок обязан быть в промпте.
    db = NovelDB()
    from novel.context import ContextBuilder  # noqa: E402
    from novel.settings import Settings  # noqa: E402

    prompt = ContextBuilder(db, object(), lambda: Settings()).build(
        session_id, TURN, count_exactly=False
    )
    db.close()
    layers = {layer.key: layer.text for layer in prompt.layers}
    check("слой заметок попал в промпт", "notes" in layers, str(list(layers)))
    check("текст заметки виден ведущему",
          NOTE[:40] in layers.get("notes", ""), layers.get("notes", "")[:120])
    check("инструкция про блок notes на месте",
          "<notes>" in layers.get("notes", ""))
    print(f"    слой заметок: {len(layers.get('notes', ''))} символов")

    # Живой ход.
    call("/api/chat", {"session_id": session_id, "text": TURN})
    started = time.time()
    while time.time() - started < 900:
        time.sleep(3)
        if not (call("/api/status").get("task") or {}).get("running"):
            break

    debug = call("/api/debug")
    raw = debug.get("raw_reply") or ""
    print()
    print("    --- хвост сырого ответа ---")
    for line in raw.strip().splitlines()[-6:]:
        print(f"    {line[:110]}")

    check("ведущий прислал блок notes", "<notes>" in raw)
    parsed = debug.get("parsed") or {}
    check("номер заметки разобран",
          note_id in (parsed.get("fulfilled_notes") or []),
          str(parsed.get("fulfilled_notes")))

    db = NovelDB()
    after = db.note(note_id)
    db.close()
    check("заметка закрыта ведущим", after is not None and after.status != "active",
          after.status if after else "заметки нет")

    # Уборка: полигон больше не нужен.
    request = urllib.request.Request(f"{BASE}/api/worlds/{world_id}", method="DELETE")
    with urllib.request.urlopen(request, timeout=60):
        pass
    call("/api/settings", {"changes": {"image_policy": policy}})
    print(f"\n  полигон #{world_id} удалён, политика возвращена: {policy}")
    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
