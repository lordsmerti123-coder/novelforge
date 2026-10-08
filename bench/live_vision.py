"""Проверка зрения: модель получает изображение и описывает его.

Заодно измеряется цена картинки в токенах: у модели есть бюджет на изображение,
и от него зависит, сколько кадров можно приложить к одному запросу.

Запуск::

    python bench\\live_vision.py
    python bench\\live_vision.py --image <каталог вывода ComfyUI>\\novel_00001_00001_.png
    python bench\\live_vision.py --question "Что не так с этой сценой?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402

MODELS_DIR = config.DEFAULT_MODELS_DIR
from novel.console import setup_console  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402
from novel.vision import VisionError, prepare_image, user_message  # noqa: E402

DEFAULT_IMAGE = config.COMFY_OUTPUT_DIR / "novel_00001_00001_.png"


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Проверка зрения модели")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="путь к изображению")
    parser.add_argument("--question", default="Что ты видишь на этой картинке? Опиши коротко и по делу.")
    parser.add_argument("--text-only", action="store_true", help="тот же вопрос без картинки, для сравнения")
    args = parser.parse_args()

    client = FreeTokenClient(timeout_s=300.0)
    print("\n=== Проверка зрения ===\n")

    try:
        status = client.health().get("status")
    except FreeTokenError as exc:
        print(f"движок недоступен: {exc}")
        return 2
    if status != "ok":
        print("движок не готов")
        return 2

    model = (client.stats().get("model") or {})
    modalities = model.get("input_modalities") or []
    print(f"модель: {model.get('id')}")
    print(f"принимает: {', '.join(modalities) if modalities else 'не сообщено'}")
    if "image" not in modalities:
        print("\nэта модель не принимает изображения — выбери Gemma-4 или Qwen3.6")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"нет файла {image_path}")
        return 1

    # Сначала тот же вопрос без картинки: разница в токенах и есть цена изображения.
    print("\n--- без картинки ---")
    started = time.time()
    blind = client.chat([{"role": "user", "content": args.question}], max_tokens=200, temperature=0.6)
    print(f"токенов входа {blind.prompt_tokens}, ответ {blind.completion_tokens} за "
          f"{time.time() - started:.1f} c")
    print(f"ответ: {blind.text.strip()[:200]}")

    if args.text_only:
        return 0

    print("\n--- с картинкой ---")
    try:
        data_url, width, height, original = prepare_image(image_path)
    except VisionError as exc:
        print(f"не удалось подготовить изображение: {exc}")
        return 1
    size_kb = len(data_url) * 3 / 4 / 1024
    print(f"файл: {image_path.name}")
    print(f"размер: {original[0]}x{original[1]} -> {width}x{height}, ~{size_kb:.0f} KB")

    started = time.time()
    try:
        result = client.chat_stream(
            [user_message(args.question, [data_url])], max_tokens=300, temperature=0.6
        )
    except FreeTokenError as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    elapsed = time.time() - started
    print(f"токенов входа {result.prompt_tokens} (+{result.prompt_tokens - blind.prompt_tokens} к тексту), "
          f"ответ {result.completion_tokens} за {elapsed:.1f} c")
    print(f"\nответ модели:\n{result.text.strip()}")

    print("\n=== Итог ===")
    print(f"  изображение стоит примерно {result.prompt_tokens - blind.prompt_tokens} токенов входа")
    print(f"  время ответа с картинкой: {elapsed:.1f} c\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
