"""Шаг 3: реальная цена генерации картинки на Qwen-Image-2.1.

Проверяет три вещи, от которых зависит бюджет игрового хода:

1. сколько времени и VRAM стоит полноразмерная генерация 1024²;
2. сколько стоит черновой режим 512² с малым числом шагов — заготовка для
   спекулятивной предгенерации;
3. сколько VRAM реально возвращает ``POST /free`` и за сколько секунд.

Перед запуском движок FreeToken должен быть остановлен: на 12 GB VRAM ему и
ComfyUI места нет.

Запуск::

    python bench\\phase3_image.py
    python bench\\phase3_image.py --no-stop-engine --skip-draft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.comfy import ComfyClient, ComfyError, GenerationResult, build_t2i_graph  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.engine import EngineController  # noqa: E402

PROMPT = (
    "dark elf barkeep behind a candlelit tavern counter, oil painting, "
    "dungeons and dragons fantasy, warm amber light, detailed face"
)


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def measured(label: str, action: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Выполняет действие, непрерывно замеряя память.

    @returns: ``(результат, сводка замеров)``.
    """
    print(f"  -> {label} ...", flush=True)
    sampler = metrics.Sampler()
    sampler.start()
    started = time.time()
    result = action()
    wall = time.time() - started
    sampler.stop()
    summary = sampler.summary().as_dict()
    summary["wall_s"] = round(wall, 3)
    print(
        f"     {wall:.1f} c, VRAM занято {summary['gpu_used_mb']['min']:.0f}"
        f"..{summary['gpu_used_mb']['max']:.0f} MB, "
        f"свободно минимум {summary['gpu_free_mb']['min']:.0f} MB, "
        f"RAM минимум {summary['ram_available_mb']['min']:.0f} MB",
        flush=True,
    )
    return result, summary


def generation_report(result: GenerationResult, summary: dict[str, Any]) -> dict[str, Any]:
    """Сводит результат генерации и замеры в один отчёт."""
    return {
        "prompt_id": result.prompt_id,
        "elapsed_s": round(result.elapsed_s, 2),
        "images": [str(path) for path in result.images],
        "image_exists": [path.exists() for path in result.images],
        "memory": summary,
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Замер генерации Qwen-Image-2.1")
    parser.add_argument("--full-size", type=int, default=1024)
    parser.add_argument("--full-steps", type=int, default=25)
    parser.add_argument("--draft-size", type=int, default=512)
    parser.add_argument("--draft-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--no-stop-engine", action="store_true", help="не останавливать FreeToken перед прогоном"
    )
    parser.add_argument("--skip-draft", action="store_true", help="пропустить черновой режим")
    args = parser.parse_args()

    report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds")}
    comfy = ComfyClient()

    print("\n=== Шаг 3: цена генерации изображения ===\n")

    if not args.no_stop_engine:
        controller = EngineController(port=1919)
        if controller.port_pid() is not None:
            print("Остановка движка FreeToken ...")
            stop = controller.stop(timeout_s=30.0)
            report["engine_stop"] = stop
            print(
                f"  остановлен за {stop['stop_seconds']} c, "
                f"свободно VRAM {stop['gpu_free_after_mb']} MB "
                f"(было {stop['gpu_free_before_mb']} MB)"
            )
        else:
            print("Движок FreeToken уже остановлен.")

    gpu = metrics.gpu_stats()
    report["gpu_before_comfy"] = gpu
    print(f"Перед запуском ComfyUI свободно VRAM {gpu['free_mb']} MB\n")

    if not comfy.is_alive():
        print("Запуск Comfy Desktop (это занимает 40-60 с) ...")
        started = time.time()
        alive, waited = comfy.ensure_running(timeout_s=240.0)
        report["comfy_start"] = {"alive": alive, "waited_s": round(waited, 1)}
        if not alive:
            print("  ComfyUI не поднялся за 240 с — прогон остановлен.")
            return 1
        print(f"  сервер ответил через {waited:.1f} c")
    else:
        report["comfy_start"] = {"alive": True, "waited_s": 0.0}
        print("ComfyUI уже запущен.")

    stats = comfy.system_stats()
    report["system_stats"] = {
        "comfyui_version": (stats.get("system") or {}).get("comfyui_version"),
        "device": ((stats.get("devices") or [{}])[0]).get("name"),
    }
    print(f"  ComfyUI {(stats.get('system') or {}).get('comfyui_version')} на "
          f"{((stats.get('devices') or [{}])[0]).get('name')}")

    models = comfy.check_models()
    report["models"] = {
        "ok": models["ok"],
        "missing": models["missing"],
        "available_counts": {key: len(value) for key, value in models["available"].items()},
    }
    if not models["ok"]:
        print(f"  !! не найдены модели: {models['missing']}")
        out = config.MEASUREMENTS_DIR / f"phase3_image_{stamp()}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    print("  все три модели видны загрузчикам")

    # --- полноразмерная генерация ------------------------------------------
    print(f"\nПолноразмерная генерация {args.full_size}x{args.full_size}, "
          f"{args.full_steps} шагов:")
    graph = build_t2i_graph(
        PROMPT,
        width=args.full_size,
        height=args.full_size,
        seed=args.seed,
        steps=args.full_steps,
        filename_prefix="novel_full",
    )
    report["full_graph"] = graph
    try:
        result, summary = measured(
            "t2i полный", lambda: comfy.generate(graph, timeout_s=1200.0)
        )
        report["full"] = generation_report(result, summary)
        print(f"     файл: {result.first_image}")
    except ComfyError as exc:
        report["full"] = {"error": str(exc)}
        print(f"  !! ошибка: {exc}")

    # --- освобождение VRAM --------------------------------------------------
    print("\nВыгрузка моделей ComfyUI:")
    gpu_before_free = metrics.gpu_stats()
    try:
        _, free_summary = measured("POST /free", lambda: comfy.free())
        time.sleep(3.0)
        gpu_after_free = metrics.gpu_stats()
        report["free_after_full"] = {
            "gpu_before_mb": gpu_before_free,
            "gpu_after_mb": gpu_after_free,
            "freed_mb": gpu_after_free["free_mb"] - gpu_before_free["free_mb"],
            "memory": free_summary,
        }
        print(
            f"     свободно VRAM {gpu_before_free['free_mb']} -> "
            f"{gpu_after_free['free_mb']} MB "
            f"(+{gpu_after_free['free_mb'] - gpu_before_free['free_mb']} MB)"
        )
    except ComfyError as exc:
        report["free_after_full"] = {"error": str(exc)}
        print(f"  !! ошибка: {exc}")

    # --- черновой режим -----------------------------------------------------
    if not args.skip_draft:
        print(f"\nЧерновой режим {args.draft_size}x{args.draft_size}, {args.draft_steps} шагов:")
        draft_graph = build_t2i_graph(
            PROMPT,
            width=args.draft_size,
            height=args.draft_size,
            seed=args.seed,
            steps=args.draft_steps,
            filename_prefix="novel_draft",
        )
        report["draft_graph"] = draft_graph
        try:
            result, summary = measured(
                "t2i черновик", lambda: comfy.generate(draft_graph, timeout_s=900.0)
            )
            report["draft"] = generation_report(result, summary)
            print(f"     файл: {result.first_image}")
        except ComfyError as exc:
            report["draft"] = {"error": str(exc)}
            print(f"  !! ошибка: {exc}")

        print("\nПовторная выгрузка моделей:")
        gpu_before_free = metrics.gpu_stats()
        try:
            _, free_summary = measured("POST /free", lambda: comfy.free())
            time.sleep(3.0)
            gpu_after_free = metrics.gpu_stats()
            report["free_after_draft"] = {
                "gpu_before_mb": gpu_before_free,
                "gpu_after_mb": gpu_after_free,
                "freed_mb": gpu_after_free["free_mb"] - gpu_before_free["free_mb"],
                "memory": free_summary,
            }
            print(
                f"     свободно VRAM {gpu_before_free['free_mb']} -> "
                f"{gpu_after_free['free_mb']} MB "
                f"(+{gpu_after_free['free_mb'] - gpu_before_free['free_mb']} MB)"
            )
        except ComfyError as exc:
            report["free_after_draft"] = {"error": str(exc)}
            print(f"  !! ошибка: {exc}")

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"phase3_image_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Итог ===")
    for label, key in (("Полная 1024²", "full"), ("Черновик 512²", "draft")):
        entry = report.get(key)
        if not entry:
            continue
        if "error" in entry:
            print(f"  {label}: ОШИБКА — {entry['error'][:120]}")
        else:
            mem = entry["memory"]
            print(
                f"  {label}: {entry['elapsed_s']} c, "
                f"пик VRAM {mem['gpu_used_mb']['max']:.0f} MB, "
                f"минимум свободной {mem['gpu_free_mb']['min']:.0f} MB"
            )
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
