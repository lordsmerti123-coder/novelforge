"""Шаг 3b: сколько стоит повторная генерация, пока модели остаются в VRAM.

``POST /free`` мгновенно возвращает около 8 GB, но вместе с ними уходят и
загруженные веса. Если генерировать несколько кадров подряд, не выгружая модели
между ними, второй и последующие кадры не платят за загрузку.

Прогон отвечает на вопрос, стоит ли копить сцены и генерировать их пачкой:

* кадр A — на пустой VRAM (веса грузятся с диска);
* кадр B — сразу за A, модели уже в памяти;
* ``POST /free``;
* кадр C — снова на пустой VRAM, проверка, что разница именно в загрузке.

Запуск::

    python bench\\phase3b_warm.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.comfy import ComfyClient, ComfyError, build_t2i_graph  # noqa: E402
from novel.console import setup_console  # noqa: E402

PROMPTS = [
    "dark elf barkeep behind a candlelit tavern counter, oil painting, fantasy",
    "ruined mine entrance overgrown with moss, oil painting, fantasy",
    "hooded traveller reading a map by lantern light, oil painting, fantasy",
]


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_one(comfy: ComfyClient, index: int, label: str) -> dict[str, Any]:
    """Генерирует один кадр, замеряя память и время.

    @returns: отчёт по кадру.
    """
    graph = build_t2i_graph(
        PROMPTS[index],
        width=1024,
        height=1024,
        seed=1000 + index,
        steps=25,
        filename_prefix=f"warm_{label}",
    )
    print(f"  -> кадр {label} ...", flush=True)
    sampler = metrics.Sampler()
    sampler.start()
    try:
        result = comfy.generate(graph, timeout_s=1200.0)
    except ComfyError as exc:
        sampler.stop()
        print(f"     ошибка: {exc}")
        return {"label": label, "error": str(exc)}
    sampler.stop()
    summary = sampler.summary().as_dict()
    print(
        f"     {result.elapsed_s:.1f} c, пик VRAM {summary['gpu_used_mb']['max']:.0f} MB, "
        f"минимум свободной {summary['gpu_free_mb']['min']:.0f} MB",
        flush=True,
    )
    return {
        "label": label,
        "elapsed_s": round(result.elapsed_s, 2),
        "image": None if result.first_image is None else str(result.first_image),
        "memory": summary,
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()
    comfy = ComfyClient()

    print("\n=== Шаг 3b: холодная и тёплая генерация ===\n")
    if not comfy.is_alive():
        print("ComfyUI не отвечает на 127.0.0.1:8188 — запусти Comfy Desktop и нажми Start.")
        return 2

    report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds")}

    print("Выгрузка моделей перед началом:")
    comfy.free()
    time.sleep(3.0)
    gpu = metrics.gpu_stats()
    report["after_initial_free"] = gpu
    print(f"  свободно VRAM {gpu['free_mb']} MB\n")

    report["cold_a"] = run_one(comfy, 0, "cold_a")
    report["warm_b"] = run_one(comfy, 1, "warm_b")

    print("\nВыгрузка моделей:")
    gpu_before = metrics.gpu_stats()
    comfy.free()
    time.sleep(3.0)
    gpu_after = metrics.gpu_stats()
    report["free_after_warm"] = {
        "gpu_before_mb": gpu_before,
        "gpu_after_mb": gpu_after,
        "freed_mb": gpu_after["free_mb"] - gpu_before["free_mb"],
    }
    print(f"  свободно VRAM {gpu_before['free_mb']} -> {gpu_after['free_mb']} MB\n")

    report["cold_c"] = run_one(comfy, 2, "cold_c")

    cold_a = report["cold_a"].get("elapsed_s")
    warm_b = report["warm_b"].get("elapsed_s")
    cold_c = report["cold_c"].get("elapsed_s")
    report["conclusion"] = {
        "cold_a_s": cold_a,
        "warm_b_s": warm_b,
        "cold_c_s": cold_c,
        "saving_per_extra_image_s": None if not (cold_a and warm_b) else round(cold_a - warm_b, 1),
    }

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"phase3b_warm_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Итог ===")
    print(f"  первый кадр (веса с диска):   {cold_a} c")
    print(f"  второй кадр (веса в VRAM):    {warm_b} c")
    print(f"  третий кадр (после /free):    {cold_c} c")
    if cold_a and warm_b:
        print(f"  экономия на каждом следующем кадре: {cold_a - warm_b:.1f} c")
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
