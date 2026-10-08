"""Живая проверка агента-редактора через интерфейс.

Проверяется весь путь: указание словами, разбор на действия, выполнение,
предпросмотр без изменений и запрет на удаление без явного разрешения.

Запуск::

    python bench\\live_agent.py
    python bench\\live_agent.py --keep
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


def wait_agent(limit_s: float = 900.0) -> dict:
    """Ждёт завершения работы агента и возвращает отчёт."""
    started = time.time()
    while time.time() - started < limit_s:
        time.sleep(2.0)
        status = call("/api/status", timeout=60.0)
        if not (status.get("task") or {}).get("running"):
            return call("/api/agent", timeout=60.0).get("last_run") or {}
    return {}


def run_agent(instruction: str, **options: Any) -> dict:
    """Запускает агента и ждёт отчёт."""
    print(f"\n--- указание: {instruction}")
    call("/api/agent", {"instruction": instruction, **options})
    report = wait_agent()
    for step in report.get("steps") or []:
        if step.get("thought"):
            print(f"    мысль: {step['thought'][:140]}")
        for index, action in enumerate(step.get("actions") or []):
            result = (step.get("results") or [{}])[index] if index < len(
                step.get("results") or []
            ) else {}
            if result.get("preview"):
                mark = "будет сделано"
            elif result.get("ok"):
                mark = "готово"
            else:
                mark = f"ошибка: {result.get('error')}"
            print(f"    {action.get('op')} -> {mark}")
    if report.get("error"):
        print(f"    остановка: {report['error']}")
    print(f"    изменений {report.get('changed')}, шагов {len(report.get('steps') or [])}, "
          f"{report.get('seconds')} c")
    return report


def test_interrupt(world_id: int) -> None:
    """Выгрузка движка посреди работы агента.

    Движок останавливают штатно — перед генерацией картинки, при смене модели и
    кнопкой «Стоп всё». Агент обязан либо пережить это, подняв движок заново,
    либо остановиться на границе шага и честно показать, что успел применить.
    """
    global PASSED, FAILED
    print("\n--- выгрузка движка во время работы агента")
    instruction = (
        f"в мире #{world_id} добавь правило про туман в порту, "
        f"затем добавь правило про контрабанду, затем добавь персонажа-лоцмана"
    )
    call("/api/agent", {"instruction": instruction})
    time.sleep(3.5)

    status = call("/api/status", timeout=60.0)
    running = (status.get("task") or {}).get("running")
    if not running:
        print("    агент успел закончить до выгрузки — проверка не показательная")
        wait_agent()
        return

    print("    гашу движок прямо посреди работы")
    call("/api/kill", {"stop_comfy": False}, timeout=120.0)
    report = wait_agent()

    check("отчёт получен после выгрузки", bool(report.get("instruction")))
    check("агент не завис и завершил прогон",
          report.get("finished") or report.get("interrupted") or bool(report.get("error")),
          str({k: report.get(k) for k in ("finished", "interrupted", "error")}))
    print(f"    прерван: {report.get('interrupted')}, движок поднимался заново: "
          f"{report.get('revived')}, изменений {report.get('changed')}")
    if report.get("applied"):
        print(f"    успело примениться: {', '.join(report['applied'])}")
    if report.get("error"):
        print(f"    сообщение: {report['error'][:120]}")

    db = NovelDB()
    integrity = db.check_integrity()
    db.close()
    check("база осталась целой после выгрузки", integrity == [], str(integrity))

    print("    возвращаю движок")
    call("/api/engine/start", {})
    wait_agent(600.0)


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Живая проверка агента")
    parser.add_argument("--keep", action="store_true", help="не удалять созданный мир")
    parser.add_argument("--skip-interrupt", action="store_true", help="не проверять выгрузку")
    args = parser.parse_args()

    print("\n=== Агент-редактор ===\n")

    db = NovelDB()
    world_id = db.create_world(
        name="Полигон агента", format="story",
        brief="Портовый город, где гильдии делят доки.", genre="тёмное фэнтези",
    )
    rule_id = db.add_rule(world_id, "В городе запрещено носить оружие открыто.", title="Оружие")
    db.add_character(world_id, "Старый Боцман", role="хозяин кабака", appearance="old sailor")
    db.close()
    print(f"полигон: мир #{world_id}, правило #{rule_id}")

    overview = call("/api/agent", timeout=60.0)
    check("агент отдаёт список действий", len(overview.get("operations") or []) >= 10,
          str(len(overview.get("operations") or [])))
    check("действия с удалением помечены",
          any(op["destructive"] for op in overview.get("operations") or []))

    preview = run_agent(
        f"в мире #{world_id} добавь правило о том, что любая сделка с гильдией имеет цену",
        preview=True,
    )
    check("предпросмотр вернул план", bool(preview.get("steps")))
    check("предпросмотр ничего не изменил", preview.get("changed") == 0,
          str(preview.get("changed")))
    db = NovelDB()
    before = len(db.rules(world_id))
    db.close()
    check("правил в базе не прибавилось", before == 1, str(before))

    applied = run_agent(
        f"в мире #{world_id} добавь правило: любая сделка с гильдией имеет свою цену"
    )
    check("агент выполнил указание", applied.get("changed", 0) >= 1,
          f"изменений {applied.get('changed')}")
    db = NovelDB()
    rules = db.rules(world_id)
    added = [rule for rule in rules if rule.id != rule_id]
    check("правило появилось в базе", len(added) >= 1, str([r.title for r in rules]))
    if added:
        print(f"    новое правило: {added[-1].title} — {added[-1].body[:90]}")
    db.close()

    denied = run_agent(f"удали правило #{rule_id} в мире #{world_id}")
    errors = [
        result.get("error", "")
        for step in denied.get("steps") or []
        for result in step.get("results") or []
        if not result.get("ok")
    ]
    db = NovelDB()
    still_there = db.conn.execute(
        "SELECT 1 FROM world_rules WHERE id = ?", (rule_id,)
    ).fetchone() is not None
    db.close()
    check("удаление без разрешения отклонено", still_there is not False)
    check("в отчёте есть причина отказа", bool(errors) or denied.get("changed") == 0,
          str(errors))

    rewritten = run_agent(
        f"перепиши правило #{rule_id} в мире #{world_id}: сделай его мрачнее и конкретнее"
    )
    db = NovelDB()
    row = db.conn.execute("SELECT body FROM world_rules WHERE id = ?", (rule_id,)).fetchone()
    body = row["body"] if row else ""
    db.close()
    check("правило переписано", body != "В городе запрещено носить оружие открыто.",
          body[:80])
    check("текст правила не пустой", len(body) > 10, body[:60])
    if body:
        print(f"    стало: {body[:120]}")

    if not args.skip_interrupt:
        test_interrupt(world_id)

    if not args.keep:
        db = NovelDB()
        db.delete_world(world_id)
        db.close()
        print(f"\nполигон #{world_id} удалён")

    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
