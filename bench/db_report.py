"""Отчёт по базе: какие таблицы есть, что в них лежит.

Полезно после миграций схемы и при разборе «куда пропал мир».

Запуск::

    python bench\\db_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Отчёт по базе NovelForge")
    parser.add_argument("--session", type=int, default=None, help="показать сообщения этой партии")
    parser.add_argument("--tail", type=int, default=8, help="сколько последних сообщений показать")
    parser.add_argument("--world", type=int, default=None, help="то же для последней партии мира")
    args = parser.parse_args()

    db = NovelDB()

    if args.session is not None or args.world is not None:
        session_id = args.session
        if session_id is None:
            sessions = db.sessions(args.world)
            if not sessions:
                print(f"у мира {args.world} нет партий")
                return 1
            session_id = sessions[0].id
        session = db.session(session_id)
        print(f"\n=== Партия #{session_id} {session.title!r} ===\n")
        messages = db.messages(session_id)
        for message in messages[-args.tail :]:
            mark = f" [вложений {len(message.attachment_paths)}]" if message.attachment_paths else ""
            print(f"--- {message.role}{mark} #{message.id} ---")
            print(message.content[:500])
            print()
        db.close()
        return 0

    print(f"\n=== База {db.path} ===\n")

    tables = sorted(
        row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    )
    print("таблицы:", ", ".join(tables))

    version = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    print("версия схемы:", version["value"] if version else "не записана")
    print("счётчики:", db.stats())

    print("\n--- Миры ---")
    for world in db.worlds():
        rules = db.rules(world.id)
        characters = db.characters(world.id)
        sessions = db.sessions(world.id)
        print(
            f"  #{world.id} {world.name!r} формат={world.format} "
            f"правил={len(rules)} персонажей={len(characters)} партий={len(sessions)}"
        )
        print(f"      жанр={world.genre!r} тон={world.tone!r}")
        print(f"      стиль={world.style!r}")
        print(f"      описание: {world.brief[:90]!r}")

    print("\n--- Партии ---")
    for session in db.sessions():
        messages = db.messages(session.id)
        scenes = db.scenes(session.id)
        memories = db.memories(session.id)
        print(
            f"  #{session.id} мир={session.world_id} {session.title!r} "
            f"сообщений={len(messages)} сцен={len(scenes)} память={len(memories)}"
        )
        for scene in scenes:
            mark = "есть" if scene.path and Path(scene.path).exists() else "нет файла"
            print(f"      сцена #{scene.id} [{scene.status}] файл: {mark} — {scene.prompt[:60]!r}")

    print("\n--- Прочее ---")
    for table in tables:
        if table.startswith(("turns", "scenes_", "speculative_", "world_state_")):
            count = db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"  остаток старой схемы: {table} ({count} строк)")

    db.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
