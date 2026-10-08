"""Что каждая модель умеет на входе: текст, картинки, видео.

Проверяется статически по конфигам чекпоинта: наличие башни зрения, препроцессора
изображений и медиа-полей в конфиге. Живой ответ движка (``input_modalities`` в
``/v1/stats``) зависит от того, какая модель сейчас загружена, поэтому статическая
проверка удобнее для обзора всех моделей сразу.

Запуск::

    python bench\\modality_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.models import discover_models  # noqa: E402

VISION_KEYS = (
    "vision_config", "vision_tower", "mm_vision_tower", "image_token_index",
    "vision_start_token_id", "mm_projector_type", "image_seq_length",
)
VIDEO_KEYS = ("video_config", "video_token_index", "video_seq_length")


def read_json(path: Path) -> dict[str, Any]:
    """Читает JSON-файл, возвращая пустой словарь при любой проблеме."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe(folder: Path) -> dict[str, Any]:
    """Определяет входные модальности чекпоинта по его файлам."""
    config = read_json(folder / "config.json")
    processors = sorted(p.name for p in folder.glob("*preprocessor_config.json"))
    keys = set(config)
    vision = bool(keys & set(VISION_KEYS)) or any("preprocessor_config" in name for name in processors)
    video = bool(keys & set(VIDEO_KEYS)) or any("video_preprocessor" in name for name in processors)
    return {
        "vision": vision,
        "video": video,
        "processors": processors,
        "vision_keys": sorted(keys & set(VISION_KEYS)),
        "has_mmproj": any(folder.glob("mmproj*")),
    }


def main() -> int:
    setup_console()
    print("\n=== Входные модальности моделей ===\n")

    checked = 0
    for model in discover_models():
        if model.kind == "gguf":
            continue
        folder = Path(model.path)
        info = describe(folder)
        modalities = ["текст"]
        if info["vision"]:
            modalities.append("картинки")
        if info["video"]:
            modalities.append("видео")
        print(f"  {model.name}")
        print(f"     архитектура: {model.model_type}")
        print(f"     принимает: {', '.join(modalities)}")
        if info["vision_keys"]:
            print(f"     признаки зрения в конфиге: {', '.join(info['vision_keys'])}")
        if info["processors"]:
            print(f"     препроцессоры: {', '.join(info['processors'])}")
        print()
        checked += 1

    print(f"проверено моделей: {checked}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
