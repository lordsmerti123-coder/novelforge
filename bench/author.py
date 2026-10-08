"""Сочинение миров, правил, персонажей и завязок силами локальной модели.

Инструмент для писательской работы: придумать мир с нуля, дописать к нему
правила и персонажей, предложить завязки, переписать неудачный текст, разобрать
готовый мир на слабые места.

Модель должна быть запущена: инструмент сам поднимает движок, если он стоит.

Примеры::

    python bench\\author.py world "затонувший город, где память хранят в бутылках"
    python bench\\author.py rules --world 3 --count 3
    python bench\\author.py scenarios --world 3 --count 5
    python bench\\author.py critique --world 3
    python bench\\author.py rewrite --text "мрачный мир" --instruction "сделай конкретнее"
    python bench\\author.py rules --world 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402
from novel.authoring import Author, AuthoringError, save_world  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.engine import EngineConfig, EngineController  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402
from novel.settings import SettingsStore  # noqa: E402


def ensure_engine() -> None:
    """Поднимает движок, если он не отвечает.

    @raises SystemExit: если поднять не удалось.
    """
    client = FreeTokenClient(timeout_s=10.0)
    try:
        if client.health().get("status") == "ok":
            print("движок уже работает")
            return
    except FreeTokenError:
        pass

    controller = EngineController(port=1919)
    if controller.port_pid() is not None:
        print("порт 1919 занят, но сервер не отвечает — жду остановки")
        if not controller.wait_port_free(timeout_s=30.0):
            print("порт не освободился")
            raise SystemExit(2)

    store = SettingsStore()
    store.load()
    settings = store.settings
    print(f"поднимаю движок: {Path(settings.model_path).name} ...")
    report = controller.cold_start(
        EngineConfig(
            name="author",
            memory_ratio=settings.engine_memory_ratio,
            moe_strategy=settings.engine_moe_strategy,
            model_path=Path(settings.model_path),
        ),
        timeout_s=900.0,
    )
    if not report.get("ready"):
        errors = (report.get("timeline") or {}).get("errors") or []
        print(f"движок не поднялся: {errors[-1][:300] if errors else 'причина неизвестна'}")
        raise SystemExit(1)
    print(f"готов за {report['timeline']['spawn_to_ready_s']} c")


def print_rules(rules: list[dict[str, str]]) -> None:
    """Печатает правила списком."""
    for index, rule in enumerate(rules, start=1):
        title = f"{rule['title']}: " if rule["title"] else ""
        print(f"  {index}. {title}{rule['body']}")


def print_characters(characters: list[dict[str, str]]) -> None:
    """Печатает карточки персонажей."""
    for character in characters:
        print(f"  {character['name']} — {character['role']}")
        if character["description"]:
            print(f"     характер: {character['description']}")
        if character["appearance"]:
            print(f"     внешность: {character['appearance']}")


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Сочинение силами локальной модели")
    parser.add_argument("action", choices=["world", "rules", "characters", "scenarios", "critique", "rewrite"])
    parser.add_argument("text", nargs="?", default="", help="идея мира для действия world, текст для rewrite")
    parser.add_argument("--world", type=int, default=None, help="мир, с которым работаем")
    parser.add_argument("--count", type=int, default=3, help="сколько придумать")
    parser.add_argument("--instruction", default="", help="указание для rewrite")
    parser.add_argument("--dry-run", action="store_true", help="не записывать в базу")
    parser.add_argument("--json", action="store_true", help="печатать сырой JSON")
    args = parser.parse_args()

    print(f"\n=== Сочинение: {args.action} ===\n")

    if args.action == "rewrite":
        text = args.text
        if not text:
            print("нужен --text")
            return 2
        if not args.instruction:
            print("нужно --instruction")
            return 2
    elif args.action == "world":
        if not args.text:
            print("нужна идея мира первым аргументом")
            return 2
    elif args.world is None:
        print("нужен --world")
        return 2

    ensure_engine()
    client = FreeTokenClient(timeout_s=900.0)
    author = Author(client)
    db = NovelDB()

    try:
        if args.action == "world":
            print(f"идея: {args.text}\n")
            drafted = author.draft_world(args.text)
            print(f"мир: {drafted.name}")
            print(f"жанр: {drafted.genre} | тон: {drafted.tone} | повествование: {drafted.narrator}")
            print(f"стиль: {drafted.style}\n")
            print(f"{drafted.brief}\n")
            print(f"правила ({len(drafted.rules)}):")
            print_rules(drafted.rules)
            print(f"\nперсонажи ({len(drafted.characters)}):")
            print_characters(drafted.characters)
            if args.json:
                print("\n" + json.dumps(drafted.as_dict(), ensure_ascii=False, indent=2))
            if args.dry_run:
                print("\nпробный прогон: в базу не записано")
            else:
                world_id, session_id = save_world(db, drafted)
                print(f"\nзаписан мир #{world_id}, партия #{session_id}")

        elif args.action == "rules":
            world = db.world(args.world)
            if world is None:
                print(f"мир {args.world} не найден")
                return 2
            rules = author.more_rules(world, db.rules(args.world), args.count)
            print(f"новые правила ({len(rules)}):")
            print_rules(rules)
            if not args.dry_run:
                for rule in rules:
                    db.add_rule(args.world, rule["body"], title=rule["title"])
                print(f"\nдобавлено в мир #{args.world}")

        elif args.action == "characters":
            world = db.world(args.world)
            if world is None:
                print(f"мир {args.world} не найден")
                return 2
            characters = author.more_characters(world, db.characters(args.world), args.count)
            print(f"новые персонажи ({len(characters)}):")
            print_characters(characters)
            if not args.dry_run:
                for character in characters:
                    db.add_character(
                        args.world, character["name"], role=character["role"],
                        description=character["description"], appearance=character["appearance"],
                        speech=character["speech"],
                    )
                print(f"\nдобавлено в мир #{args.world}")

        elif args.action == "scenarios":
            world = db.world(args.world)
            if world is None:
                print(f"мир {args.world} не найден")
                return 2
            scenarios = author.scenarios(world, db.rules(args.world), args.count)
            for index, item in enumerate(scenarios, start=1):
                print(f"--- {index}. {item['title']} ---")
                print(f"{item['opening']}")
                if item["hook"]:
                    print(f"конфликт: {item['hook']}")
                if item["twist"]:
                    print(f"поворот: {item['twist']}")
                print()
            if args.json:
                print(json.dumps(scenarios, ensure_ascii=False, indent=2))

        elif args.action == "critique":
            world = db.world(args.world)
            if world is None:
                print(f"мир {args.world} не найден")
                return 2
            report = author.critique(world, db.rules(args.world), db.characters(args.world))
            print(f"вердикт: {report.get('verdict')}\n")
            for title, key in (("проблемы", "problems"), ("что делать", "fixes"), ("чего не хватает", "missing")):
                items = report.get(key) or []
                if items:
                    print(f"{title}:")
                    for item in items:
                        print(f"  - {item}")
                    print()

        elif args.action == "rewrite":
            print(f"указание: {args.instruction}\n")
            result = author.rewrite(args.text, args.instruction)
            print("--- было ---")
            print(args.text)
            print("\n--- стало ---")
            print(result)

    except AuthoringError as exc:
        print(f"не получилось: {exc}")
        return 1
    finally:
        db.close()

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
