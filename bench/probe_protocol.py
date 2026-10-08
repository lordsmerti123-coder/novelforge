r"""Проба: годится ли модель в ведущего — протокол и промпт для картинки.

Малая модель может писать складный текст и при этом не уметь того, на чём
держится приложение: блока ``<scene>`` с английским описанием кадра. Без него
не будет картинок, а без корректного JSON разборщик отбросит блок целиком.

Проба берёт **настоящий промпт приложения** — тот же сборщик контекста, что и в
игре, — и разбирает ответ **настоящим разборщиком**. Поэтому результат говорит
ровно о пригодности к проекту, а не о красоте текста вообще.

Мир и реплики в пробе нейтральные: проверяется формат ответа, а не содержание.

Запуск::

    python bench\probe_protocol.py
    python bench\probe_protocol.py --model <путь>.gguf --turns 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.context import ContextBuilder  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.external import ExternalConfig, find_server  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402
from novel.protocol import parse_reply  # noqa: E402
from novel.settings import Settings, SettingsStore  # noqa: E402

#: Малая модель: проверяется, тянет ли она роль ведущего дешёвым чекпоинтом.
#: Путь задаётся переменной окружения или ключом ``--model``.
DEFAULT_MODEL = Path(os.environ.get("NOVELFORGE_GGUF_MODEL",
                                    Path.home() / "models" / "model.gguf"))
PORT = 1924

#: Нейтральные реплики: проверяется форма ответа, а не содержание.
TURNS = [
    "Я вхожу в таверну и оглядываюсь.",
    "Подхожу к стойке и спрашиваю у хозяйки, что слышно в городе.",
]


def wait_ready(process: subprocess.Popen, timeout_s: int = 240) -> bool:
    """Ждёт готовности сервера.

    @param process: запущенный сервер.
    @param timeout_s: сколько ждать.
    @returns: ``True``, если сервер поднялся.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=4) as response:
                if b"ok" in response.read():
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def make_world(db: NovelDB, fmt: str = "story") -> tuple[int, int]:
    """Заводит нейтральный мир и партию для пробы.

    @param db: база пробы.
    @param fmt: формат диалога; в форматах переписки кадр рисуется портретом
        в полный рост, в истории — обычной иллюстрацией.
    @returns: идентификаторы мира и партии.
    """
    from novel.presets import get_preset

    preset = get_preset("tavern") or {}
    world_id = db.create_world(
        name="Проба протокола", format=fmt,
        brief=preset.get("brief", ""), genre=preset.get("genre", ""),
        tone=preset.get("tone", ""), style=preset.get("style", ""),
    )
    for rule in preset.get("rules", []):
        db.add_rule(world_id, rule["body"], title=rule["title"], kind="rule")
    for character in preset.get("characters", []):
        db.add_character(world_id, character["name"], role=character["role"],
                         description=character["description"],
                         appearance=character["appearance"], speech=character["speech"])
    session_id = db.create_session(world_id, "Проба")
    return world_id, session_id


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Проба протокола модели")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--turns", type=int, default=len(TURNS))
    parser.add_argument("--format", default="story",
                        help="формат мира: story или chat_photo")
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    model = Path(args.model)
    print("Проба протокола и промпта для картинок")
    print(f"модель: {model.name} ({model.stat().st_size / 1024**3:.2f} ГБ)" if model.exists()
          else f"модель: {model} — НЕ НАЙДЕНА")
    if not model.exists():
        return 1

    server = find_server()
    if server is None:
        print("рабочая сборка llama.cpp не найдена")
        return 1
    print(f"сборка: {server.parent.name}\n")

    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cfg = ExternalConfig(model_path=model, port=PORT, context=8192)
    process = subprocess.Popen(cfg.argv(server), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=creation,
                               cwd=str(server.parent))
    try:
        started = time.time()
        if not wait_ready(process):
            print("сервер не поднялся")
            return 1
        print(f"модель загружена за {time.time() - started:.1f} c\n")

        client = FreeTokenClient(f"http://127.0.0.1:{PORT}", timeout_s=600)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = NovelDB(Path(tmp) / "probe.db")
            try:
                world_id, session_id = make_world(db, args.format)
                store = SettingsStore(Path(tmp) / "settings.json")
                builder = ContextBuilder(db, client, lambda: store.settings, lambda *a: [])
                from novel.formats import get_format

                # Требование «в полный рост» зависит от формата, поэтому
                # спрашивается у самого сборщика контекста, а не угадывается.
                portrait = builder._portrait_mode(get_format(args.format))
                print(f"формат: {args.format} | портрет в полный рост: "
                      f"{'требуется' if portrait else 'не требуется'}\n")
                for index, text in enumerate(TURNS[:args.turns], 1):
                    prompt = builder.build(session_id, text, count_exactly=False)
                    messages = prompt.chat_messages()
                    body = {
                        "model": "probe", "messages": messages,
                        "max_tokens": store.settings.max_tokens,
                        "temperature": 0.8, "stream": False,
                    }
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                        headers={"Content-Type": "application/json; charset=utf-8"},
                        method="POST")
                    turn_started = time.time()
                    with urllib.request.urlopen(request, timeout=900) as response:
                        doc = json.loads(response.read().decode("utf-8"))
                    elapsed = time.time() - turn_started
                    raw = ((doc.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                    usage = doc.get("usage") or {}
                    parsed = parse_reply(raw)

                    print(f"--- ход {index}: {text[:56]}")
                    print(f"    время {elapsed:5.1f} c | токенов {usage.get('completion_tokens')} "
                          f"| промпт {prompt.total_tokens}")
                    print(f"    текст для игрока: {len(parsed.prose)} символов")
                    print(f"    блок scene: {'ЕСТЬ' if parsed.scene else 'НЕТ'}")
                    if parsed.scene is not None:
                        image = parsed.scene.image_prompt or ""
                        english = image and all(ord(c) < 128 for c in image)
                        print(f"      место: {parsed.scene.location or '—'}")
                        print(f"      кадр:  {image[:110]}")
                        print(f"      по-английски: {'да' if english else 'НЕТ'}")
                        if portrait:
                            print(f"      в полный рост: "
                                  f"{'да' if 'full body' in image.lower() else 'НЕТ'}")
                    if parsed.errors:
                        print(f"    ошибки разбора: {parsed.errors[:2]}")
                    db.add_message(session_id, "user", text)
                    if parsed.prose:
                        db.add_message(session_id, "assistant", parsed.prose)
            finally:
                db.close()
    finally:
        if not args.keep_running:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
