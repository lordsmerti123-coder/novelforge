"""Прогон непроверенных путей: автозапуск ComfyUI, форматы чата, фото, смена модели.

Скрипт дополняет :mod:`bench.live_world`: там проверялся основной путь «история с
описаниями», а здесь — всё остальное, что до сих пор не гонялось живьём.

Порядок проверок:

1. ComfyUI убит — поднимает ли его оркестратор сам;
2. формат «просто чат»: ответ без блока сцены, реплики строками;
3. фотография в реплике: видит ли модель вложение;
4. смена модели через интерфейс: перезапускается ли движок;
5. автоматическая суммаризация во время хода;
6. полный ход с иллюстрацией на поднятом напрямую ComfyUI;
7. аварийная остановка и возврат движка.

Запуск::

    python bench\\live_matrix.py
    python bench\\live_matrix.py --skip-image --skip-model-switch
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402

MODELS_DIR = config.DEFAULT_MODELS_DIR
from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402

BASE = "http://127.0.0.1:8760"
PASSED = 0
FAILED = 0
GEMMA = MODELS_DIR / "Gemma-4-26B-A4B-NVFP4"
GPT_OSS = MODELS_DIR / "gpt-oss-20b"
SAMPLE_IMAGE = config.COMFY_OUTPUT_DIR / "novel_00001_00001_.png"


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
    """Ждёт завершения фоновой операции, печатая новые строки журнала.

    Журнал отдаётся последними восемьюдесятью строками, поэтому следить за
    номером позиции нельзя: при переполнении срез сдвигается и печатается
    заново то, что уже было. Ориентиром служит последняя увиденная строка.
    """
    last_line: str | None = None
    started = time.time()
    while time.time() - started < limit_s:
        time.sleep(2.0)
        status = call("/api/status", timeout=60.0)
        lines = status.get("log") or []
        fresh: list[str]
        if last_line is None:
            fresh = lines
        elif last_line in lines:
            index = len(lines) - 1 - lines[::-1].index(last_line)
            fresh = lines[index + 1 :]
        else:
            fresh = lines
        for line in fresh:
            print("       " + line)
        if lines:
            last_line = lines[-1]
        if not (status.get("task") or {}).get("running"):
            return status
    print(f"       !! {label}: не завершилось за {limit_s} c")
    return call("/api/status", timeout=60.0)


def turn(session_id: int, text: str, images: list[str] | None = None, limit: float = 600.0) -> dict:
    """Один ход через интерфейс."""
    call("/api/chat", {"session_id": session_id, "text": text, "images": images or []})
    wait_task(limit, "ход")
    history = call(f"/api/sessions/{session_id}/history")
    return history


def test_comfy_autostart(skip: bool) -> None:
    """ComfyUI убит — поднимает ли его оркестратор сам."""
    print("\n1. Автозапуск ComfyUI")
    if skip:
        print("  пропущено")
        return
    was_alive = call("/api/status")["comfyui"]["alive"]
    print(f"     до: ComfyUI {'отвечает' if was_alive else 'не отвечает'}")
    if was_alive:
        print("     сервер уже поднят — проверка автозапуска пропущена")
        return

    started = time.time()
    call("/api/comfy/start", {})
    wait_task(400.0, "запуск ComfyUI")
    elapsed = time.time() - started
    status = call("/api/status")
    check("оркестратор поднял ComfyUI сам", status["comfyui"]["alive"], f"за {elapsed:.0f} c")
    print(f"     поднялся за {elapsed:.1f} c")
    if status["comfyui"]["alive"]:
        stats = call("/api/status")
        print(f"     движок при этом: {stats['engine'].get('model')}")


def test_chat_format() -> tuple[int, int]:
    """Формат «просто чат»: без сцены, репликами."""
    print("\n2. Формат «просто чат»")
    created = call("/api/worlds", {"name": "Проверка чата", "format": "chat"})
    world_id, session_id = created["world_id"], created["session_id"]
    call(f"/api/worlds/{world_id}", {"changes": {
        "brief": "Переписка двух знакомых из небольшого города.",
        "tone": "тёплый, бытовой",
    }})
    call(f"/api/worlds/{world_id}/rules", {"body": "Пиши только сообщения, без описаний действий."})

    history = turn(session_id, "Привет! Ты уже дома?")
    messages = history["messages"]
    assistant = [m for m in messages if m["role"] == "assistant"]
    check("ответ получен", bool(assistant))
    if assistant:
        text = assistant[-1]["content"]
        check("ответ не пустой", bool(text.strip()))
        check("теги не просочились в текст", "<prose>" not in text and "<scene>" not in text,
              text[:80])
        print(f"     ответ: {text.strip()[:160]}")
    check("сцен не создано (формат без картинок)", not history["scenes"],
          f"сцен {len(history['scenes'])}")
    check("формат определён как chat", history["format"]["key"] == "chat")
    return world_id, session_id


def test_photo(session_id: int) -> None:
    """Фотография в реплике: видит ли модель вложение."""
    print("\n3. Фотография в реплике")
    if not SAMPLE_IMAGE.exists():
        print(f"     нет файла {SAMPLE_IMAGE} — пропущено")
        return
    raw = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{raw}"
    print(f"     прикладываю {SAMPLE_IMAGE.name} ({SAMPLE_IMAGE.stat().st_size // 1024} KB)")

    history = turn(session_id, "Что на этой фотографии? Опиши, что видно.", [data_url])
    messages = history["messages"]
    user_with_photo = [m for m in messages if m["role"] == "user" and m.get("attachments")]
    check("вложение записано в сообщение", bool(user_with_photo))
    if user_with_photo:
        print(f"     вложений: {user_with_photo[-1]['attachments']}")
    assistant = [m for m in messages if m["role"] == "assistant"]
    if assistant:
        text = assistant[-1]["content"].strip()
        print(f"     ответ: {text[:220]}")
        check("ответ не пустой", bool(text))
        # Точность описания зависит от формулировки и оценивается человеком.
        # Здесь проверяется то, что обязано быть всегда: модель не делает вид,
        # что вложения не было.
        denial = ("не вижу", "не могу увидеть", "прикрепи", "загрузите изображение",
                  "нет картинки", "не вижу картинку")
        check("модель не отрицает, что видит изображение",
              not any(phrase in text.lower() for phrase in denial), text[:120])
        markers = ("свеч", "таверн", "трактир", "стол", "мужчин", "стен", "свет", "дерев", "уютн")
        if any(word in text.lower() for word in markers):
            print("     в ответе есть детали с картинки")
        else:
            print("     деталей не названо — оцени вручную")


def test_model_switch(skip: bool) -> None:
    """Смена модели через интерфейс: перезапускается ли движок."""
    print("\n4. Смена модели через интерфейс")
    if skip:
        print("  пропущено")
        return
    before = call("/api/status")["engine"].get("model")
    print(f"     сейчас: {before}")

    call("/api/models/select", {"path": GPT_OSS, "apply_preset": True})
    started = time.time()
    wait_task(400.0, "перезапуск движка")
    status = call("/api/status")
    print(f"     после переключения: {status['engine'].get('model')} "
          f"({time.time() - started:.0f} c)")
    check("движок переключился на gpt-oss",
          "gpt-oss" in str(status["engine"].get("model")), str(status["engine"].get("model")))
    check("движок отвечает", status["engine"].get("healthy") is True)

    settings = call("/api/settings")
    check("выбор сохранён в настройках", settings["model_path"] == GPT_OSS, settings["model_path"])

    switched = call("/api/models")
    gemma = [m for m in switched["models"] if m["path"] == GEMMA]
    check("прежняя модель есть в реестре", bool(gemma))

    call("/api/models/select", {"path": GEMMA, "apply_preset": True})
    wait_task(400.0, "возврат движка")
    back = call("/api/status")
    check("движок вернулся на Gemma", "Gemma" in str(back["engine"].get("model")),
          str(back["engine"].get("model")))


def test_auto_summary() -> None:
    """Автоматическая суммаризация во время настоящего хода."""
    print("\n5. Автоматическая суммаризация во время хода")
    db = NovelDB()
    try:
        world_id = db.create_world(name="Проверка суммаризации", format="story",
                                   brief="Небольшой тракт между двумя городами.")
        session_id = db.create_session(world_id, "Длинная дорога")
        # Истории должно быть заведомо больше окна: при окне в 12 сообщений и
        # пороге в 6 выброшенных сводка включается начиная с 19-го сообщения.
        for index in range(24):
            role = "user" if index % 2 == 0 else "assistant"
            db.add_message(session_id, role, f"шаг {index}: " + "событие на дороге " * 12)
        before = db.memories(session_id)
        print(f"     сообщений до хода: {db.count_messages(session_id)}, сводок: {len(before)}")
    finally:
        db.close()

    turn(session_id, "Я продолжаю путь и смотрю по сторонам.")

    db = NovelDB()
    try:
        memories = db.memories(session_id)
        check("сводка появилась автоматически", len(memories) > len(before),
              f"было {len(before)}, стало {len(memories)}")
        if memories:
            print(f"     сводка ({memories[-1].tokens} токенов): {memories[-1].summary[:160]}")
        stats = db.session_stats(session_id)
        check("сводка покрывает старые сообщения", stats["memories"] >= 1)
        db.delete_world(world_id)
    finally:
        db.close()


def test_illustrated_turn() -> None:
    """Полный ход с иллюстрацией на поднятом напрямую ComfyUI."""
    print("\n6. Ход с иллюстрацией")
    settings = call("/api/settings")
    call("/api/settings", {"changes": {"image_policy": "on_scene_change", "image_steps": 12}})
    try:
        created = call("/api/worlds", {"name": "Проверка иллюстрации", "format": "story"})
        world_id, session_id = created["world_id"], created["session_id"]
        call(f"/api/worlds/{world_id}", {"changes": {
            "brief": "Подземный город под горой, освещённый кристаллами.",
            "style": "oil painting, dark fantasy, underground city",
        }})
        print("     отправляю ход со сценой (генерация около полутора минут) ...")
        history = turn(session_id, "Я выхожу на площадь подземного города.", limit=900.0)
        scenes = history["scenes"]
        check("сцена создана", bool(scenes), f"сцен {len(scenes)}")
        done = [s for s in scenes if s["status"] == "done"]
        check("кадр сгенерирован", bool(done), str([s["status"] for s in scenes]))
        if done:
            image = Path(done[-1]["path"] or "")
            check("файл кадра на диске", image.exists(), str(image))
            check("время генерации записано", bool(done[-1]["elapsed_s"]))
            print(f"     кадр: {image} ({done[-1]['elapsed_s']:.0f} c)")

        status = call("/api/status")
        check("движок вернулся после генерации", status["engine"].get("healthy") is True,
              str(status["engine"].get("model")))
        db = NovelDB()
        try:
            db.delete_world(world_id)
        finally:
            db.close()
    finally:
        call("/api/settings", {"changes": {
            "image_policy": settings["image_policy"],
            "image_steps": settings["image_steps"],
        }})


def test_kill_switch() -> None:
    """Аварийная остановка и возврат движка."""
    print("\n7. Аварийная остановка")
    before = call("/api/status")
    print(f"     до: движок {before['engine'].get('model')}")
    report = call("/api/kill", {"stop_comfy": False}, timeout=120.0)
    print(f"     остановлено pid {report['engine_pids']}, освобождено {report['freed_mb']} MB")
    check("движок погашен", not call("/api/status")["engine"]["healthy"])
    check("ComfyUI не тронут", call("/api/status")["comfyui"]["alive"],
          "stop_comfy=False, но сервер погас")

    call("/api/engine/start", {})
    wait_task(400.0, "возврат движка")
    after = call("/api/status")
    check("движок вернулся", after["engine"].get("healthy") is True)


def main() -> int:
    setup_console()
    config.ensure_dirs()
    parser = argparse.ArgumentParser(description="Прогон непроверенных путей")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-model-switch", action="store_true")
    parser.add_argument("--skip-comfy", action="store_true")
    args = parser.parse_args()

    print("\n=== Прогон непроверенных путей ===\n")
    started = time.time()

    test_comfy_autostart(args.skip_comfy)
    _world_id, session_id = test_chat_format()
    test_photo(session_id)
    test_model_switch(args.skip_model_switch)
    test_auto_summary()
    if not args.skip_image:
        test_illustrated_turn()
    else:
        print("\n6. Ход с иллюстрацией — пропущен")
    test_kill_switch()

    print(f"\n=== Итог: пройдено {PASSED}, провалено {FAILED}, "
          f"заняло {time.time() - started:.0f} c ===\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
