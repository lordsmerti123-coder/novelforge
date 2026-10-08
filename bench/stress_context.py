"""Стресс-тест контекста: 20, 50 и 100 сообщений истории.

Проверяется не скорость модели, а поведение сборки контекста: сколько токенов
уходит на каждый слой, сколько сообщений попадает в окно, что выбрасывается и
как справляется суммаризация.

История наполняется напрямую в хранилище: сто настоящих ходов заняли бы
полчаса, а измеряется здесь работа с объёмом, а не генерация. Настоящий запрос
к модели делается один — в конце каждого уровня, чтобы убедиться, что модель
работает с полной памятью и не теряет формат.

Запуск::

    python bench\\stress_context.py
    python bench\\stress_context.py --levels 20,50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.context import ContextBuilder  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402
from novel.presets import get_preset  # noqa: E402
from novel.protocol import parse_reply  # noqa: E402
from novel.settings import SettingsStore  # noqa: E402

PLAYER_LINES = [
    "Я осматриваюсь по сторонам и прислушиваюсь к разговорам.",
    "Подхожу к стойке и спрашиваю, есть ли работа.",
    "Проверяю, не следит ли кто-нибудь за мной.",
    "Достаю карту и сверяюсь с ней.",
    "Заказываю еду и сажусь в углу.",
    "Спрашиваю у трактирщика про пропавший караван.",
    "Оставляю монету на стойке и выхожу на улицу.",
]

MASTER_OPENERS = [
    "Ты толкаешь тяжёлую дверь, и в лицо ударяет запах жареного мяса и старого эля.",
    "Трактирщик отрывается от кружки и смотрит на тебя оценивающе, не говоря ни слова.",
    "Разговоры за столами стихают на пару секунд, потом возобновляются.",
    "Дождь барабанит по ставням, и в зале становится заметно темнее.",
    "За дальним столом двое наёмников пересчитывают монеты и о чём-то спорят.",
    "Сурр подаётся вперёд и понижает голос так, что его слышно только тебе.",
]

MASTER_BODIES = [
    "Марла вытирает руки о передник и кивает в сторону лестницы. На втором этаже "
    "свободна комната, но за неё просят вперёд и не деньгами.",
    "Ты замечаешь, что за столом у окна сидит человек в плаще с низко опущенным "
    "капюшоном. Он не пьёт и не ест, только смотрит на дверь.",
    "Хозяин отвечает уклончиво: караван вышел три недели назад и не вернулся. "
    "Стража объявила розыск, но дальше бумаги дело не пошло.",
    "На полу под столом блестит оброненная монета нездешней чеканки. На ней "
    "изображён профиль, которого ты не узнаёшь.",
    "Сурр шепчет, что за информацию надо платить дважды: сначала ему, потом тем, "
    "у кого он её берёт. Иначе не доживёшь до второго раза.",
    "С улицы доносится стук копыт. Кто-то подъехал к заднему входу и не спешит "
    "входить внутрь.",
]

MEMORY_NOTE = "Ты помнишь, что Марла говорила о долге, и это всё ещё не решено."


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def synthesize(db: NovelDB, session_id: int, count: int, rng: random.Random) -> None:
    """Наполняет партию правдоподобной историей без обращения к модели.

    @param db: хранилище.
    @param session_id: партия.
    @param count: сколько сообщений добавить.
    @param rng: источник случайности с фиксированным зерном, чтобы прогоны были
        сопоставимы между собой.
    """
    for index in range(count):
        if index % 2 == 0:
            text = rng.choice(PLAYER_LINES)
        else:
            head = rng.choice(MASTER_OPENERS)
            body = rng.choice(MASTER_BODIES)
            tail = MEMORY_NOTE if rng.random() < 0.25 else ""
            text = " ".join(part for part in (head, body, tail) if part)
        role = "user" if index % 2 == 0 else "assistant"
        db.add_message(session_id, role, text, tokens=len(text) // 3)


def describe(builder: ContextBuilder, session_id: int, user_input: str) -> dict[str, Any]:
    """Собирает контекст и возвращает послойный разбор."""
    started = time.time()
    prompt = builder.build(session_id, user_input)
    elapsed = time.time() - started
    return {
        "build_s": round(elapsed, 2),
        "total_tokens": prompt.total_tokens,
        "budget_tokens": prompt.budget_tokens,
        "stable_tokens": sum(layer.tokens for layer in prompt.layers if layer.stable),
        "window_tokens": sum(layer.tokens for layer in prompt.layers if layer.key == "window"),
        "included_messages": len(prompt.included_message_ids),
        "dropped_messages": len(prompt.dropped_message_ids),
        "warnings": prompt.warnings,
        "layers": [
            {"key": layer.key, "title": layer.title, "tokens": layer.tokens}
            for layer in prompt.layers
        ],
        "exact": prompt.exact_tokens,
    }


def real_turn(client: FreeTokenClient, builder: ContextBuilder, session_id: int, note: str) -> dict[str, Any]:
    """Настоящий запрос к модели на текущем контексте."""
    user_input = "Я подвожу итог и решаю, что делать дальше."
    prompt = builder.build(session_id, user_input)
    started = time.time()
    try:
        result = client.chat_stream(
            prompt.chat_messages(), max_tokens=300, temperature=0.8, timeout_s=600.0
        )
    except FreeTokenError as exc:
        return {"ok": False, "error": str(exc)}
    parsed = parse_reply(result.text)
    return {
        "ok": True,
        "note": note,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "ttft_s": None if result.ttft_s is None else round(result.ttft_s, 2),
        "elapsed_s": round(result.elapsed_s, 2),
        "decode_tps": round(result.decode_tokens_per_second, 2),
        "wall_s": round(time.time() - started, 2),
        "format_ok": bool(parsed.prose.strip()),
        "prose_head": parsed.prose[:160].replace("\n", " "),
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Стресс-тест контекста")
    parser.add_argument("--levels", default="20,50,100", help="объёмы истории через запятую")
    parser.add_argument("--seed", type=int, default=7, help="зерно генератора истории")
    parser.add_argument("--keep", action="store_true", help="не удалять созданные миры")
    args = parser.parse_args()

    levels = [int(item) for item in args.levels.split(",") if item.strip()]
    db = NovelDB()
    store = SettingsStore()
    store.load()
    client = FreeTokenClient()
    builder = ContextBuilder(db, client, lambda: store.settings)
    rng = random.Random(args.seed)

    report: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "levels": [],
        "settings": {
            "context_budget_tokens": store.settings.context_budget_tokens,
            "context_turns": store.settings.context_turns,
            "max_tokens": store.settings.max_tokens,
            "auto_summarize": store.settings.auto_summarize,
            "summarize_after_dropped": store.settings.summarize_after_dropped,
        },
    }

    print("\n=== Стресс-тест контекста ===\n")
    print(f"Бюджет {store.settings.context_budget_tokens} токенов, "
          f"окно {store.settings.context_turns} сообщений, "
          f"резерв под ответ {store.settings.max_tokens}, "
          f"суммаризация после {store.settings.summarize_after_dropped} выброшенных\n")

    if client.health().get("status") != "ok":
        print("Движок не отвечает — подними его и повтори.")
        return 2

    preset = get_preset("tavern")
    world_id = db.create_world(
        name=f"Стресс-тест {datetime.now():%H:%M}",
        format="story",
        brief=preset["brief"],
        genre=preset["genre"],
        tone=preset["tone"],
        style=preset["style"],
        narrator=preset["narrator"],
    )
    for rule in preset["rules"]:
        db.add_rule(world_id, rule["body"], title=rule["title"])
    for character in preset["characters"]:
        db.add_character(
            world_id,
            character["name"],
            role=character["role"],
            description=character["description"],
            appearance=character["appearance"],
            speech=character["speech"],
        )
    print(f"Мир #{world_id} создан: правил {len(preset['rules'])}, "
          f"персонажей {len(preset['characters'])}\n")

    for level in levels:
        session_id = db.create_session(world_id, title=f"История {level} сообщений")
        synthesize(db, session_id, level, rng)
        print(f"--- {level} сообщений ---")

        before = describe(builder, session_id, "Что происходит вокруг?")
        print(f"    сборка за {before['build_s']} c: всего {before['total_tokens']} токенов "
              f"(стабильных {before['stable_tokens']}, окно {before['window_tokens']}), "
              f"в окне {before['included_messages']}, выброшено {before['dropped_messages']}")

        summary: dict[str, Any] | None = None
        if builder.needs_summary(session_id, before["dropped_messages"]):
            print("    сворачиваю историю ...")
            started = time.time()
            try:
                summary = builder.summarize(session_id)
            except FreeTokenError as exc:
                summary = {"error": str(exc)}
            if summary:
                summary["wall_s"] = round(time.time() - started, 2)
                if summary.get("error"):
                    print(f"    ОШИБКА суммаризации: {summary['error']}")
                else:
                    print(f"    свёрнуто {summary['summarized_messages']} сообщений за "
                          f"{summary['wall_s']} c, память {summary['tokens']} токенов")
        else:
            print("    суммаризация не требуется")

        after = describe(builder, session_id, "Что происходит вокруг?")
        if summary:
            print(f"    после свёртки: всего {after['total_tokens']} токенов, "
                  f"окно {after['window_tokens']}, выброшено {after['dropped_messages']}")

        turn = real_turn(client, builder, session_id, f"{level}")
        if turn.get("ok"):
            print(f"    настоящий запрос: {turn['completion_tokens']} токенов за "
                  f"{turn['elapsed_s']} c (ttft {turn['ttft_s']} c, {turn['decode_tps']} ток/с), "
                  f"формат {'ок' if turn['format_ok'] else 'НАРУШЕН'}")
        else:
            print(f"    ОШИБКА запроса: {turn.get('error')}")

        memories = db.memories(session_id)
        report["levels"].append(
            {
                "messages": level,
                "session_id": session_id,
                "before": before,
                "summary": summary,
                "after": after,
                "turn": turn,
                "memory_chars": sum(len(m.summary) for m in memories),
                "memory_count": len(memories),
            }
        )
        print()

    if not args.keep:
        db.delete_world(world_id)
        print(f"Мир #{world_id} удалён (--keep оставляет его в базе)\n")
    db.close()

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"stress_context_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Итог ===")
    print(f"  {'сообщений':>10} {'до свёртки':>12} {'окно':>7} {'выброшено':>10} "
          f"{'после':>8} {'память':>8} {'запрос':>8}")
    for entry in report["levels"]:
        before, after, turn = entry["before"], entry["after"], entry["turn"]
        print(
            f"  {entry['messages']:>10} {before['total_tokens']:>12} "
            f"{before['window_tokens']:>7} {before['dropped_messages']:>10} "
            f"{after['total_tokens']:>8} {entry['memory_chars']:>8} "
            f"{(turn.get('elapsed_s') if turn.get('ok') else '—')!s:>8}"
        )
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
