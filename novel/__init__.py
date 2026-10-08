"""NovelForge — генератор историй с ИИ-мастером и иллюстрациями.

Пакет разделён на независимые слои:

* :mod:`novel.metrics` — замеры VRAM, RAM и времени;
* :mod:`novel.freetoken` — клиент текстового движка FreeToken;
* :mod:`novel.comfy` — клиент генерации изображений ComfyUI;
* :mod:`novel.db` — хранилище ходов, сцен и состояния мира;
* :mod:`novel.protocol` — разбор ответа LLM на теги ``<prose>``/``<scene>``;
* :mod:`novel.machine` — конечный автомат, связывающий всё вместе.
"""

__all__ = ["config"]
