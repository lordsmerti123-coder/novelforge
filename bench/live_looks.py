"""Проверка ходов, кадров и смены внешности на живой машине.

Делает несколько ходов подряд, снимает доспехи и смотрит:

* рисуется ли кадр на каждом ходу;
* меняется ли зерно, то есть будут ли кадры разными;
* приходит ли от ведущего блок ``looks`` и попадает ли он в состояние партии;
* виден ли этот блок в следующем запросе к модели.

Работает на копии мира, чтобы не портить настоящую партию.

Запуск::

    python bench\\live_looks.py
    python bench\\live_looks.py --world 8 --keep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402

BASE = "http://127.0.0.1:8760"
PASSED = 0
FAILED = 0

TURNS = [
    "Я выхожу из таверны на улицу и оглядываюсь по сторонам.",
    "Я снимаю с себя доспехи и остаюсь в простой рубахе, перевязываю руку тряпкой.",
    "Я иду к колодцу в конце улицы и заглядываю в него.",
]


def check(name: str, condition: bool, detail: str = "") -> None:
    """Печатает результат одной проверки."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  ФЕЙЛ {name}{'' if not detail else ' — ' + detail}")


def call(path: str, body: dict | None = None, timeout: float = 900.0) -> Any:
    """Запрос к интерфейсу."""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(
        f"{BASE}{path}", data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} на {path}: {exc.read().decode('utf-8', 'replace')}") from exc


def wait_turn(limit_s: float = 900.0) -> float:
    """Ждёт окончания хода и возвращает, сколько он занял."""
    started = time.time()
    while time.time() - started < limit_s:
        time.sleep(4.0)
        status = call("/api/status", timeout=60.0)
        if not (status.get("task") or {}).get("running"):
            return time.time() - started
    return time.time() - started


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Ходы, кадры и смена внешности")
    parser.add_argument("--world", type=int, default=8, help="мир, с которого снять копию")
    parser.add_argument("--keep", action="store_true", help="не удалять копию после проверки")
    args = parser.parse_args()

    print("\n=== Ходы, кадры и смена внешности ===\n")
    settings = call("/api/settings")
    print(f"политика картинок: {settings['image_policy']}")
    if settings["image_policy"] != "every_turn":
        call("/api/settings", {"changes": {"image_policy": "every_turn"}})
        print("поставил на время проверки «каждый ход»")

    copy = call(f"/api/worlds/{args.world}/duplicate", {"with_sessions": False})
    world_id = copy["world_id"]
    session_id = copy["session_ids"][0]
    print(f"полигон: мир #{world_id} «{call(f'/api/worlds/{world_id}')['world']['name']}», "
          f"партия #{session_id}\n")

    seeds: list[int] = []
    for index, text in enumerate(TURNS, start=1):
        print(f"--- ход {index}: {text[:64]}")
        call("/api/chat", {"session_id": session_id, "text": text})
        spent = wait_turn()
        db = NovelDB()
        scenes = db.scenes(session_id)
        looks = db.get_state(session_id, "looks", {}) or {}
        db.close()
        if scenes:
            last = scenes[-1]
            seeds.append(last.seed or 0)
            has = "есть" if last.path else "НЕТ"
            print(f"    кадр #{last.id}: {last.status}, картинка {has}, "
                  f"seed={last.seed}, {round(last.elapsed_s or 0)} c, ход {round(spent)} c")
            print(f"    описание: {last.prompt[:90]}")
        else:
            print("    сцен нет")
        print(f"    внешность сейчас: {looks or 'пусто'}")

    db = NovelDB()
    scenes = db.scenes(session_id)
    looks = db.get_state(session_id, "looks", {}) or {}
    place = db.get_state(session_id, "last_image_location", "")
    db.close()

    print()
    check("на каждом ходу появилась сцена", len(scenes) >= len(TURNS), str(len(scenes)))
    drawn = [s for s in scenes if s.path]
    check("все кадры нарисованы", len(drawn) == len(scenes), f"{len(drawn)} из {len(scenes)}")
    check("зерно у кадров разное", len(set(seeds)) == len(seeds), str(seeds))

    files = {s.path for s in drawn}
    check("файлы у кадров разные", len(files) == len(drawn), f"{len(files)} из {len(drawn)}")

    check("ведущий сообщил о смене внешности", bool(looks), str(looks))
    if looks:
        text = " ".join(looks.values()).lower()
        check("в описании есть переодевание",
              any(word in text for word in ("рубах", "доспех", "бинт", "перевяз")), text[:120])
        check("внешность привязана к имени", all(name.strip() for name in looks))

    check("место последнего кадра запомнено", bool(place), repr(place))

    # Слой внешности обязан быть в следующем запросе.
    layers = call("/api/debug").get("prompt", {}).get("layers") or []
    by_key = {layer["key"]: layer["text"] for layer in layers}
    check("слой внешности есть в запросе", "looks" in by_key, str(list(by_key)))
    if "looks" in by_key:
        print(f"\n    слой «Внешность сейчас»:\n    " +
              by_key["looks"].replace("\n", "\n    ")[:400])

    if not args.keep:
        call(f"/api/worlds/{world_id}", None) if False else None
        request = urllib.request.Request(f"{BASE}/api/worlds/{world_id}", method="DELETE")
        with urllib.request.urlopen(request, timeout=60):
            pass
        print(f"\nполигон #{world_id} удалён")

    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
