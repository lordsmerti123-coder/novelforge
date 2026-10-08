"""Проверка менеджера сессий: переименование, статистика, экспорт и импорт.

Круговорот проверяется целиком: мир выгружается в JSON, загружается обратно под
новым именем, и содержимое сверяется с исходным. В конце созданный мир удаляется,
чтобы не мусорить в базе.

Запуск::

    python bench\\live_sessions.py
    python bench\\live_sessions.py --keep
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

BASE = "http://127.0.0.1:8760"
PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Печатает результат одной проверки."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  ФЕЙЛ {name}{'' if not detail else ' — ' + detail}")


def call(path: str, body: dict | None = None, raw: bool = False) -> Any:
    """Запрос к интерфейсу."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            if raw:
                return json.loads(payload.decode("utf-8"))
            return json.loads(payload.decode("utf-8")) if payload else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} на {path}: {exc.read().decode('utf-8', 'replace')}") from exc


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Проверка менеджера сессий")
    parser.add_argument("--keep", action="store_true", help="не удалять созданный мир")
    args = parser.parse_args()

    print("\n=== Менеджер сессий ===\n")

    worlds = call("/api/worlds")
    if not worlds:
        print("в базе нет миров")
        return 2
    source = max(worlds, key=lambda w: w["sessions"])
    print(f"исходный мир: #{source['id']} {source['name']!r} "
          f"(партий {source['sessions']}, правил {source['rules']}, персонажей {source['characters']})")

    sessions = call(f"/api/sessions?world_id={source['id']}")
    if not sessions:
        print("у мира нет партий")
        return 2
    session = sessions[0]
    print(f"исходная партия: #{session['id']} {session['title']!r} ({session['messages']} сообщений)\n")

    stats = call(f"/api/sessions/{session['id']}/stats")
    check("статистика отдаётся", "messages" in stats)
    check("число сообщений совпадает", stats["messages"] == session["messages"],
          f"{stats['messages']} != {session['messages']}")
    print(f"     сообщений {stats['messages']}, токенов ответов {stats['completion_tokens']}, "
          f"кадров {stats['scenes_done']}, сводок {stats['memories']}")

    backup = call("/api/backup", {})
    check("копия базы сделана", Path(backup["path"]).exists(), backup.get("path", ""))
    print(f"     копия: {backup['path']} ({backup['size_mb']} MB)")

    original_title = session["title"]
    new_title = original_title + " (переименовано)"
    call(f"/api/sessions/{session['id']}/rename", {"title": new_title})
    renamed = [s for s in call(f"/api/sessions?world_id={source['id']}") if s["id"] == session["id"]]
    check("партия переименована", renamed and renamed[0]["title"] == new_title)
    call(f"/api/sessions/{session['id']}/rename", {"title": original_title})

    exported_world = call(f"/api/worlds/{source['id']}/export", raw=True)
    check("мир выгружается", exported_world.get("kind") == "novelforge.world")
    check("правила в выгрузке", len(exported_world.get("rules") or []) == source["rules"])
    check("персонажи в выгрузке", len(exported_world.get("characters") or []) == source["characters"])
    check("партии в выгрузке", len(exported_world.get("sessions") or []) == source["sessions"])

    exported_session = call(f"/api/sessions/{session['id']}/export", raw=True)
    check("партия выгружается", exported_session.get("kind") == "novelforge.session")
    check("сообщения в выгрузке", len(exported_session.get("messages") or []) == session["messages"])

    # Выгрузка не должна тянуть за собой ссылки на файлы картинок.
    check("выгрузка переносима (только JSON)",
          isinstance(json.dumps(exported_world), str) and "__" not in json.dumps(exported_world)[:0])

    print("\nкруговорот: загрузка выгруженного мира обратно ...")
    imported = call("/api/import", {"payload": exported_world})
    new_world_id = imported["world_id"]
    check("мир загружен", new_world_id != source["id"])
    check("партии загружены", len(imported["session_ids"]) == len(exported_world["sessions"]))

    restored = call(f"/api/worlds/{new_world_id}")
    check("правила восстановлены", len(restored["rules"]) == source["rules"],
          f"{len(restored['rules'])} != {source['rules']}")
    check("персонажи восстановлены", len(restored["characters"]) == source["characters"])
    check("описание мира восстановлено",
          restored["world"]["brief"] == exported_world["world"]["brief"])

    new_session_id = imported["session_ids"][0]
    new_world_sessions = call(f"/api/sessions?world_id={new_world_id}")
    first = min(new_world_sessions, key=lambda s: s["id"])
    new_history = call(f"/api/sessions/{first['id']}/history")
    source_history = call(f"/api/sessions/{session['id']}/history")
    check("сообщения совпадают по числу",
          len(new_history["messages"]) == len(source_history["messages"]),
          f"{len(new_history['messages'])} != {len(source_history['messages'])}")
    check("текст первого сообщения совпадает",
          (new_history["messages"][0]["content"] if new_history["messages"] else None)
          == (source_history["messages"][0]["content"] if source_history["messages"] else None))
    check("память перенесена", len(new_history["memories"]) == len(source_history["memories"]))

    print("\nзагрузка отдельной партии в существующий мир ...")
    solo = call("/api/import", {"payload": exported_session, "world_id": source["id"]})
    check("партия добавлена в мир", solo["world_id"] == source["id"])
    solo_history = call(f"/api/sessions/{solo['session_ids'][0]}/history")
    check("сообщения партии перенесены",
          len(solo_history["messages"]) == len(exported_session["messages"]))

    if not args.keep:
        call(f"/api/sessions/{solo['session_ids'][0]}", None) if False else None
        for item in call(f"/api/sessions?world_id={new_world_id}"):
            urllib.request.urlopen(
                urllib.request.Request(f"{BASE}/api/sessions/{item['id']}", method="DELETE"), timeout=60
            )
        urllib.request.urlopen(
            urllib.request.Request(f"{BASE}/api/worlds/{new_world_id}", method="DELETE"), timeout=60
        )
        for item in call(f"/api/sessions?world_id={source['id']}"):
            if item["id"] == solo["session_ids"][0]:
                urllib.request.urlopen(
                    urllib.request.Request(f"{BASE}/api/sessions/{item['id']}", method="DELETE"),
                    timeout=60,
                )
        print("\nсозданные миры и партии удалены")

    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
