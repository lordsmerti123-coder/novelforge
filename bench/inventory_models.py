"""Инвентаризация локальных моделей: что вообще можно подать движку FreeToken.

Список берётся из общего реестра :mod:`novel.models`, а не собирается заново:
иначе инвентаризация и интерфейс начинают расходиться в том, какие модели
считаются пригодными.

Запуск::

    python bench\\inventory_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.models import SEARCH_ROOTS, discover_models, preset_for  # noqa: E402

KIND_TITLES = {"hf": "safetensors", "ftw": "родной формат движка", "gguf": "GGUF"}


def main() -> int:
    setup_console()
    print("\n=== Локальные модели ===\n")
    print("каталоги поиска: " + ", ".join(str(root) for root in SEARCH_ROOTS) + "\n")

    models = discover_models()
    if not models:
        print("ничего не найдено")
        return 0

    supported = [item for item in models if item.supported]
    print(f"всего найдено: {len(models)}, пригодных к запуску: {len(supported)}\n")

    print("--- Пригодны к запуску ---")
    for model in supported:
        moe = ""
        if model.experts:
            moe = f", MoE {model.experts} экспертов по {model.experts_per_tok} на токен"
        preset = preset_for(model.model_type)
        print(f"  {model.name}")
        print(f"     {model.model_type}{moe}")
        print(f"     формат: {KIND_TITLES.get(model.kind, model.kind)}, "
              f"квант: {model.quant or 'bf16/fp16'}, "
              f"контекст: {model.max_ctx or 'из модели'}, размер: {model.size_gb} GB")
        print(f"     пресет: temp {preset['temperature']}, top_p {preset['top_p']}, "
              f"top_k {preset['top_k']}, ответ {preset['max_tokens']} токенов")
        print(f"     {model.path}")
        if model.note:
            print(f"     примечание: {model.note}")

    unsupported = [item for item in models if not item.supported]
    if unsupported:
        print(f"\n--- Не поддерживаются движком ({len(unsupported)}) ---")
        for model in unsupported[:10]:
            print(f"  {model.name} ({model.model_type}) — {model.note}")
        if len(unsupported) > 10:
            print(f"  ... и ещё {len(unsupported) - 10}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
