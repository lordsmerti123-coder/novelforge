"""Шаг 1: сколько VRAM освобождает сжатие пулов кэша FreeToken и какой ценой.

Эксперимент полностью обратим: в конце пулы возвращаются к исходной геометрии.
Порядок:

1. снимок исходного состояния;
2. предсказание экономии по ``unit_bytes`` и ``limits``;
3. сжатие пулов до минимума, разрешённого сервером;
4. запрос чата на холодном кэше экспертов — это цена переключения;
5. восстановление исходной геометрии;
6. запрос чата на тёплом кэше — это цена возврата;
7. отчёт в ``bench/measurements``.

Запуск::

    python bench\\phase1_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.freetoken import ChatResult, FreeTokenClient, FreeTokenError  # noqa: E402

PROBE_MESSAGES = [
    {"role": "system", "content": "Ты — рассказчик тёмного фэнтези. Отвечай кратко."},
    {"role": "user", "content": "Опиши в двух предложениях вход в заброшенную шахту."},
]
PROBE_MAX_TOKENS = 64


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def state(client: FreeTokenClient) -> dict[str, Any]:
    """Текущее состояние памяти и движка."""
    gpu = metrics.gpu_stats()
    mem = metrics.ram_stats()
    doc: dict[str, Any] = {
        "gpu_used_mb": gpu["used_mb"],
        "gpu_free_mb": gpu["free_mb"],
        "ram_available_mb": mem["ram_available_mb"],
        "commit_used_mb": mem["commit_used_mb"],
    }
    try:
        stats = client.stats()
        doc["vram_bytes"] = stats.get("vram_bytes")
        doc["engine_vram_mb"] = round((stats.get("vram_bytes") or 0) / (1024 * 1024))
    except FreeTokenError as exc:
        doc["stats_error"] = str(exc)
    try:
        geometry = client.geometry()
        doc["geometry"] = {
            "moe_cache_size": geometry.moe_cache_size,
            "num_pages": geometry.num_pages,
            "num_swa_pages": geometry.num_swa_pages,
            "num_mamba_slots": geometry.num_mamba_slots,
        }
        doc["pool_bytes"] = geometry.bytes_for()
    except FreeTokenError as exc:
        doc["geometry_error"] = str(exc)
    return doc


def measured(label: str, action: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Выполняет действие, непрерывно замеряя память.

    @returns: ``(результат действия, сводка замеров)``.
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
        f"     {wall:.2f} c, VRAM занято {summary['gpu_used_mb']['min']:.0f}"
        f"..{summary['gpu_used_mb']['max']:.0f} MB, "
        f"свободно {summary['gpu_free_mb']['min']:.0f}..{summary['gpu_free_mb']['max']:.0f} MB",
        flush=True,
    )
    return result, summary


def chat_probe(client: FreeTokenClient, label: str) -> dict[str, Any]:
    """Один потоковый запрос чата с замером памяти и таймингов."""
    holder: dict[str, Any] = {}

    def run() -> ChatResult:
        result = client.chat_stream(PROBE_MESSAGES, max_tokens=PROBE_MAX_TOKENS, temperature=0.8)
        holder["chat"] = result
        return result

    _, summary = measured(label, run)
    chat: ChatResult = holder["chat"]
    return {
        "prompt_tokens": chat.prompt_tokens,
        "completion_tokens": chat.completion_tokens,
        "ttft_s": None if chat.ttft_s is None else round(chat.ttft_s, 3),
        "elapsed_s": round(chat.elapsed_s, 3),
        "decode_tps": round(chat.decode_tokens_per_second, 2),
        "text_head": chat.text[:160],
        "memory": summary,
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()
    client = FreeTokenClient(timeout_s=60.0)
    report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds")}

    print("\n=== Шаг 1: цена сжатия пулов кэша FreeToken ===\n")

    if not client.is_idle():
        print("Сервер занят активным запросом — эксперимент отменён.")
        return 2

    baseline = state(client)
    report["baseline"] = baseline
    geometry = client.geometry()
    limits = geometry.limits
    print(f"Исходно:  {geometry.describe()}")
    print(f"          GPU свободно {baseline['gpu_free_mb']} MB, "
          f"движок занимает {baseline.get('engine_vram_mb')} MB")

    minimum = {
        "moe": int(limits.get("moe_experts", {}).get("min", geometry.moe_cache_size)),
        "kv": int(limits.get("kv_tokens", {}).get("min", geometry.num_pages)),
        "swa": int(limits.get("swa_tokens", {}).get("min", geometry.num_swa_pages)),
    }
    predicted = geometry.bytes_for(
        moe=minimum["moe"], kv_tokens=minimum["kv"], swa_tokens=minimum["swa"]
    )
    current = geometry.bytes_for()
    report["limits"] = limits
    report["minimum_requested"] = minimum
    report["predicted"] = {
        "current_pool_bytes": current,
        "minimum_pool_bytes": predicted,
        "freed_bytes": current["total"] - predicted["total"],
        "freed_mb": round((current["total"] - predicted["total"]) / (1024 * 1024)),
    }
    print(f"Минимум по лимитам: moe={minimum['moe']} kv={minimum['kv']} swa={minimum['swa']}")
    print(f"Предсказанная экономия: {report['predicted']['freed_mb']} MB\n")

    # --- 3. сжатие ---------------------------------------------------------
    print("Сжатие пулов:")
    try:
        result, shrink_summary = measured(
            "cache rebuild -> минимум",
            lambda: client.cache_rebuild(
                moe_slots=minimum["moe"],
                kv_tokens=minimum["kv"],
                swa_tokens=minimum["swa"],
                wait_s=300.0,
            ),
        )
    except FreeTokenError as exc:
        report["shrink_error"] = str(exc)
        print(f"  !! ошибка: {exc}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    rebuild_doc, rebuild_elapsed = result
    time.sleep(2.0)
    after_shrink = state(client)
    report["shrink"] = {
        "requested": minimum,
        "rebuild_response": rebuild_doc,
        "rebuild_elapsed_s": round(rebuild_elapsed, 3),
        "memory_during": shrink_summary,
        "state_after": after_shrink,
    }
    print(f"  сервер вернул: moe={rebuild_doc.get('moe_cache_size')} "
          f"kv={rebuild_doc.get('num_pages')} swa={rebuild_doc.get('num_swa_pages')}")
    print(f"  GPU свободно теперь {after_shrink['gpu_free_mb']} MB "
          f"(было {baseline['gpu_free_mb']} MB), "
          f"движок занимает {after_shrink.get('engine_vram_mb')} MB")

    # --- 4. холодный чат ---------------------------------------------------
    print("\nЗапрос чата на холодном кэше экспертов:")
    try:
        report["cold_chat"] = chat_probe(client, "chat (холодный MoE-кэш)")
    except FreeTokenError as exc:
        report["cold_chat"] = {"error": str(exc)}
        print(f"  !! ошибка: {exc}")

    # --- 5. восстановление -------------------------------------------------
    print("\nВосстановление исходной геометрии:")
    try:
        result, restore_summary = measured(
            "cache rebuild -> исходная геометрия",
            lambda: client.cache_rebuild(
                moe_slots=geometry.moe_cache_size,
                kv_tokens=geometry.num_pages,
                swa_tokens=geometry.num_swa_pages,
                wait_s=300.0,
            ),
        )
        restore_doc, restore_elapsed = result
        time.sleep(2.0)
        after_restore = state(client)
        report["restore"] = {
            "requested": {
                "moe": geometry.moe_cache_size,
                "kv": geometry.num_pages,
                "swa": geometry.num_swa_pages,
            },
            "rebuild_response": restore_doc,
            "rebuild_elapsed_s": round(restore_elapsed, 3),
            "memory_during": restore_summary,
            "state_after": after_restore,
        }
        print(f"  сервер вернул: moe={restore_doc.get('moe_cache_size')} "
              f"kv={restore_doc.get('num_pages')} swa={restore_doc.get('num_swa_pages')}")
        print(f"  GPU свободно {after_restore['gpu_free_mb']} MB")
    except FreeTokenError as exc:
        report["restore_error"] = str(exc)
        print(f"  !! ошибка: {exc}")

    # --- 6. тёплый чат -----------------------------------------------------
    print("\nЗапрос чата после восстановления:")
    try:
        report["warm_chat"] = chat_probe(client, "chat (тёплый MoE-кэш)")
    except FreeTokenError as exc:
        report["warm_chat"] = {"error": str(exc)}
        print(f"  !! ошибка: {exc}")

    # --- 7. отчёт ----------------------------------------------------------
    actual_freed = after_shrink["gpu_free_mb"] - baseline["gpu_free_mb"]
    report["conclusion"] = {
        "gpu_free_before_mb": baseline["gpu_free_mb"],
        "gpu_free_after_shrink_mb": after_shrink["gpu_free_mb"],
        "actual_freed_mb": actual_freed,
        "predicted_freed_mb": report["predicted"]["freed_mb"],
        "shrink_seconds": report["shrink"]["rebuild_elapsed_s"],
        "restore_seconds": report.get("restore", {}).get("rebuild_elapsed_s"),
        "cold_ttft_s": (report.get("cold_chat") or {}).get("ttft_s"),
        "warm_ttft_s": (report.get("warm_chat") or {}).get("ttft_s"),
        "gpu_free_after_restore_mb": report.get("restore", {}).get("state_after", {}).get("gpu_free_mb"),
    }
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    out = config.MEASUREMENTS_DIR / f"phase1_cache_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Итог ===")
    c = report["conclusion"]
    print(f"Свободная VRAM:  {c['gpu_free_before_mb']} MB -> {c['gpu_free_after_shrink_mb']} MB "
          f"(освобождено {c['actual_freed_mb']} MB, предсказано {c['predicted_freed_mb']} MB)")
    print(f"Длительность:    сжатие {c['shrink_seconds']} c, "
          f"восстановление {c['restore_seconds']} c")
    print(f"Первый токен:    холодный {c['cold_ttft_s']} c, тёплый {c['warm_ttft_s']} c")
    print(f"Отчёт:           {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
