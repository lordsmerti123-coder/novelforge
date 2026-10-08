"""Шаг 2: сколько стоит полная остановка движка и его холодный старт.

Сжатие пулов (Шаг 1) отдаёт максимум 1.6 GB — этого мало для Qwen-Image-2.1.
Значит, единственный способ освободить VRAM под картинки — остановить движок
целиком. Здесь измеряется цена такого решения:

* сколько VRAM и RAM освобождает остановка и за сколько секунд;
* сколько длится холодный старт до готовности и отдельно до первого токена;
* сколько RAM нужно движку в момент старта — это главный риск, потому что
  свободной оперативной памяти на машине около 7 GB, а банки экспертов
  занимают заметно больше.

Перед запуском движок должен быть остановлен: закрой FreeToken Desktop.

Запуск::

    python bench\\phase2_coldstart.py
    python bench\\phase2_coldstart.py --ratios 0.45,0.55 --restore-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.engine import EngineConfig, EngineController  # noqa: E402


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def describe_failure(report: dict[str, Any]) -> str:
    """Короткая причина отказа движка для консоли."""
    errors = report.get("timeline", {}).get("errors") or []
    if not errors:
        return "неизвестная причина"
    text = errors[-1]
    for marker in ("CUDA out of memory", "OutOfMemoryError", "ValueError", "RuntimeError"):
        if marker in text:
            index = text.rindex(marker)
            return text[index : index + 200].replace("\n", " ")
    return text[-200:].replace("\n", " ")


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Холодный старт движка FreeToken")
    parser.add_argument(
        "--ratios",
        default="0.45",
        help="значения --memory-ratio через запятую (по умолчанию 0.45)",
    )
    parser.add_argument(
        "--restore-ratio",
        type=float,
        default=0.9,
        help="конфигурация, в которой машина останется после прогона (по умолчанию 0.9)",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="предел ожидания старта, с")
    args = parser.parse_args()

    ratios = [float(item) for item in args.ratios.split(",") if item.strip()]
    controller = EngineController(port=1919)

    print("\n=== Шаг 2: остановка и холодный старт движка ===\n")

    if controller.port_pid() is not None:
        print(f"Порт 1919 занят процессом PID {controller.port_pid()}.")
        print("Закрой FreeToken Desktop (движок должен быть остановлен) и повтори запуск.")
        return 2

    idle = {
        "gpu": metrics.gpu_stats(),
        "ram": metrics.ram_stats(),
    }
    print(
        f"Движок остановлен. GPU свободно {idle['gpu']['free_mb']} MB, "
        f"RAM свободно {idle['ram']['ram_available_mb']} MB, "
        f"commit {idle['ram']['commit_used_mb']}/{idle['ram']['commit_limit_mb']} MB\n"
    )

    report: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "idle": idle,
        "starts": [],
    }

    for ratio in ratios:
        cfg = EngineConfig(name=f"ratio{int(ratio * 100):03d}", memory_ratio=ratio)
        print(f"--- Запуск memory-ratio={ratio} ---")
        started = time.time()
        result = controller.cold_start(cfg, timeout_s=args.timeout)
        result["wall_s"] = round(time.time() - started, 2)
        report["starts"].append(result)

        if result["ready"]:
            timeline = result["timeline"]
            phases = ", ".join(
                f"{name}@{value}c"
                for name, value in (timeline.get("phase_first_seen_s") or {}).items()
            )
            first = result.get("first_token") or {}
            print(f"    готов за {timeline.get('spawn_to_ready_s')} c (фазы: {phases})")
            print(f"    первый токен через {first.get('spawn_to_first_token_s')} c "
                  f"(ttft {first.get('ttft_s')} c)")
            print(f"    свободно VRAM {result.get('gpu_free_after_mb')} MB, "
                  f"RAM {result.get('ram_available_after_mb')} MB, "
                  f"RSS движка {(result.get('rss') or {}).get('sum_mb')} MB")
        else:
            print(f"    ОТКАЗ: {describe_failure(result)}")

        if ratio != ratios[-1] or ratio != args.restore_ratio:
            print("    остановка ...")
            stop = controller.stop(timeout_s=30.0)
            result["stop"] = stop
            print(f"    остановлен за {stop['stop_seconds']} c, "
                  f"свободно VRAM {stop['gpu_free_after_mb']} MB")
            if not stop["vram_settled"]:
                print("    !! VRAM не освободилась до целевого порога за 30 c")
        print()

    # Машина должна остаться в рабочем состоянии: поднимаем обычную конфигурацию,
    # если её ещё не поднимали в этом прогоне.
    if args.restore_ratio not in ratios:
        cfg = EngineConfig(
            name=f"ratio{int(args.restore_ratio * 100):03d}", memory_ratio=args.restore_ratio
        )
        print(f"--- Восстановление рабочей конфигурации memory-ratio={args.restore_ratio} ---")
        started = time.time()
        result = controller.cold_start(cfg, timeout_s=args.timeout)
        result["wall_s"] = round(time.time() - started, 2)
        result["is_restore"] = True
        report["starts"].append(result)
        if result["ready"]:
            print(f"    готов за {result['timeline'].get('spawn_to_ready_s')} c, "
                  f"первый токен через "
                  f"{(result.get('first_token') or {}).get('spawn_to_first_token_s')} c")
        else:
            print(f"    ОТКАЗ: {describe_failure(result)}")
            print("    Открой FreeToken Desktop заново, чтобы вернуть движок.")
        print()

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"phase2_coldstart_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Итог ===")
    for entry in report["starts"]:
        name = entry["config"]["name"]
        if entry["ready"]:
            print(
                f"  {name}: готов {entry['timeline']['spawn_to_ready_s']} c, "
                f"первый токен {(entry.get('first_token') or {}).get('spawn_to_first_token_s')} c, "
                f"VRAM движка {entry.get('engine_vram_mb')} MB, "
                f"RSS {(entry.get('rss') or {}).get('sum_mb')} MB"
            )
        else:
            print(f"  {name}: ОТКАЗ — {describe_failure(entry)}")
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
