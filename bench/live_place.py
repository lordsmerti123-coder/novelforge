"""Проверка постоянства места: один и тот же замок должен выглядеть одинаково.

Рисуются три кадра:

* первый кадр места — он же становится образцом;
* второй кадр того же места с другим описанием — должен использовать образец;
* контрольный кадр без места — рисуется свободно.

Сравниваются попарно: кадры одного места обязаны быть ближе друг к другу, чем к
контрольному. Сравнение грубое — средняя разница по уменьшенным копиям, — но его
достаточно, чтобы отличить «то же самое» от «совсем другое».

Запуск::

    python bench\\live_place.py
    python bench\\live_place.py --steps 20 --keep
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

PLACE_NAME = "Замок Вороньего Клыка"
FIRST_PROMPT = (
    "a tall dark stone castle on a cliff above the sea, three towers, "
    "narrow windows, storm clouds, oil painting, dark fantasy"
)
SECOND_PROMPT = (
    "the same dark stone castle seen from the courtyard gate at dusk, "
    "torches along the wall, oil painting, dark fantasy"
)
CONTROL_PROMPT = (
    "a bright wooden village house with a red roof in a green valley, "
    "sunny day, oil painting"
)


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


def wait_task(limit_s: float, label: str) -> dict:
    """Ждёт завершения фоновой операции."""
    started = time.time()
    while time.time() - started < limit_s:
        time.sleep(2.0)
        status = call("/api/status", timeout=60.0)
        if not (status.get("task") or {}).get("running"):
            return status
    print(f"    !! {label}: не завершилось за {limit_s} c")
    return call("/api/status", timeout=60.0)


def difference(first: Path, second: Path) -> float:
    """Средняя разница яркости двух изображений по уменьшенным копиям.

    @returns: число от 0 (одинаковые) до 255 (предельно разные).
    """
    from PIL import Image, ImageChops

    size = (64, 64)
    with Image.open(first) as a, Image.open(second) as b:
        left = a.convert("RGB").resize(size)
        right = b.convert("RGB").resize(size)
        diff = ImageChops.difference(left, right)
        pixels = list(diff.getdata())
    total = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
    return round(total, 2)


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Проверка постоянства места")
    parser.add_argument("--steps", type=int, default=15, help="шагов генерации")
    parser.add_argument("--keep", action="store_true", help="не удалять созданный мир")
    args = parser.parse_args()

    print("\n=== Постоянство места ===\n")

    settings = call("/api/settings")
    call("/api/settings", {"changes": {"image_policy": "manual", "image_steps": args.steps}})
    print(f"генерация: {args.steps} шагов\n")

    db = NovelDB()
    world_id = db.create_world(
        name="Полигон места", format="story",
        brief="Скалистый берег, замок над морем.", style="oil painting, dark fantasy",
    )
    session_id = db.create_session(world_id, "Проверка")
    location_id = db.add_location(
        world_id, PLACE_NAME, prompt=FIRST_PROMPT, style="oil painting, dark fantasy", seed=4242
    )
    first_scene = db.add_scene(session_id, FIRST_PROMPT, seed=4242, location_id=location_id)
    db.close()
    print(f"мир #{world_id}, место #{location_id} «{PLACE_NAME}»")
    print(f"первый кадр места: сцена #{first_scene}\n")

    if not call("/api/status")["comfyui"]["alive"]:
        print("поднимаю ComfyUI ...")
        call("/api/comfy/start", {})
        wait_task(400.0, "запуск ComfyUI")

    print("рисую первый кадр места ...")
    call("/api/generate", {"session_id": session_id, "scene_ids": [first_scene]})
    wait_task(900.0, "первый кадр")

    # Вторая сцена создаётся только после того, как у места появился образец.
    db = NovelDB()
    second_scene = db.add_scene(session_id, SECOND_PROMPT, seed=777, location_id=location_id)
    db.close()
    print("рисую второй кадр того же места ...")
    call("/api/generate", {"session_id": session_id, "scene_ids": [second_scene]})
    wait_task(900.0, "второй кадр")

    db = NovelDB()
    control_scene = db.add_scene(session_id, CONTROL_PROMPT, seed=99)
    db.close()
    print("рисую контрольный кадр без места ...")
    call("/api/generate", {"session_id": session_id, "scene_ids": [control_scene]})
    wait_task(900.0, "контрольный кадр")

    db = NovelDB()
    try:
        first = db.scene(first_scene)
        second = db.scene(second_scene)
        control = db.scene(control_scene)
        place = db.location(location_id)
        print()
        check("первый кадр нарисован", bool(first and first.path), str(first and first.status))
        check("второй кадр нарисован", bool(second and second.path), str(second and second.status))
        check("контрольный кадр нарисован", bool(control and control.path))
        check("у места появился образец", bool(place and place.reference_path),
              str(place and place.reference_path))
        if place and first:
            check("образцом стал именно первый кадр",
                  place.reference_path == first.path, str((place.reference_path, first.path)))

        # Второй кадр того же места обязан рисоваться по образцу.
        check("второй кадр использовал образец",
              bool(second and second.used_reference),
              "признак «рисовался по образцу» не выставлен")
        check("контрольный кадр рисовался без образца",
              bool(control and not control.used_reference))

        if first and first.path and second and second.path and control and control.path:
            same_place = difference(Path(first.path), Path(second.path))
            different = difference(Path(first.path), Path(control.path))
            print(f"\n    разница «то же место»:      {same_place}")
            print(f"    разница «другое место»:     {different}")
            check("кадры одного места ближе друг к другу, чем к чужому",
                  same_place < different,
                  f"{same_place} >= {different} — образец не удержал облик")
            check("разница между кадрами места небольшая", same_place < 60,
                  f"{same_place} — кадры слишком разные")
    finally:
        db.close()

    call("/api/settings", {"changes": {"image_policy": settings["image_policy"],
                                       "image_steps": settings["image_steps"]}})
    if not args.keep:
        db = NovelDB()
        db.delete_world(world_id)
        db.close()
        print(f"\nполигон #{world_id} удалён")

    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
