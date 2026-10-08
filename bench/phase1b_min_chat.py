"""Шаг 1b: какая минимальная геометрия пулов ещё обслуживает запросы.

Шаг 1 показала, что `kv=1` освобождает больше всего памяти, но запрос на такой
геометрии не обслуживается. Этот прогон ищет наименьшую геометрию, которая
работает, и заодно печатает настоящую ошибку движка.

Каждая геометрия проверяется дважды: первый запрос идёт на пустом кэше
экспертов, второй — на прогретом. Разница между ними и есть цена холодного
кэша.

Запуск::

    python bench\\phase1b_min_chat.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402

PROBE_MESSAGES = [
    {"role": "system", "content": "Ты — рассказчик. Отвечай одним коротким предложением."},
    {"role": "user", "content": "Что ты видишь у входа в шахту?"},
]

#: Значения KV-пула (в токенах), которые проверяются по возрастанию.
KV_SWEEP = [1, 256, 1024, 4096, 16384]


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def try_chat(client: FreeTokenClient) -> dict[str, Any]:
    """Один запрос чата; ошибка движка возвращается как данные, а не исключение."""
    started = datetime.now()
    try:
        result = client.chat_stream(PROBE_MESSAGES, max_tokens=16, temperature=0.7)
    except FreeTokenError as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round((datetime.now() - started).total_seconds(), 3)}
    return {
        "ok": True,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "ttft_s": None if result.ttft_s is None else round(result.ttft_s, 3),
        "elapsed_s": round(result.elapsed_s, 3),
        "decode_tps": round(result.decode_tokens_per_second, 2),
        "text_head": result.text[:100],
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()
    client = FreeTokenClient(timeout_s=120.0)
    report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds")}

    print("\n=== Шаг 1b: наименьшая рабочая геометрия пулов ===\n")

    if not client.is_idle():
        print("Сервер занят — прогон отменён.")
        return 2

    original = client.geometry()
    limits = original.limits
    min_moe = int(limits.get("moe_experts", {}).get("min", original.moe_cache_size))
    min_swa = int(limits.get("swa_tokens", {}).get("min", original.num_swa_pages))
    report["original"] = {
        "moe": original.moe_cache_size,
        "kv": original.num_pages,
        "swa": original.num_swa_pages,
    }
    print(f"Исходная геометрия: {original.describe()}")
    print(f"Пробуем moe={min_moe}, swa={min_swa}, kv из {KV_SWEEP}\n")

    rows: list[dict[str, Any]] = []
    for kv in KV_SWEEP:
        try:
            client.cache_rebuild(moe_slots=min_moe, kv_tokens=kv, swa_tokens=min_swa, wait_s=120.0)
        except FreeTokenError as exc:
            rows.append({"kv_tokens": kv, "rebuild_error": str(exc)})
            print(f"kv={kv:<6} пересборка не удалась: {exc}")
            continue

        geometry = client.geometry()
        free_mb = metrics.gpu_stats()["free_mb"]
        first = try_chat(client)
        second = try_chat(client)
        row = {
            "kv_requested": kv,
            "kv_actual": geometry.num_pages,
            "free_vram_mb": free_mb,
            "first_request": first,
            "second_request": second,
        }
        rows.append(row)

        status = "OK " if second["ok"] else "ОТКАЗ"
        detail = (
            f"ttft {second.get('ttft_s')} c, {second.get('decode_tps')} ток/с"
            if second["ok"]
            else second.get("error", "")[:90]
        )
        print(
            f"kv={kv:<6} -> фактически {geometry.num_pages:<6} "
            f"свободно VRAM {free_mb:>5} MB  {status}  {detail}"
        )
        if first["ok"] != second["ok"] or (
            first["ok"] and second["ok"] and abs((first.get("ttft_s") or 0) - (second.get("ttft_s") or 0)) > 0.5
        ):
            print(
                f"          первый запрос: "
                f"{'OK ttft ' + str(first.get('ttft_s')) + ' c' if first['ok'] else 'ОТКАЗ ' + first.get('error', '')[:70]}"
            )

    report["sweep"] = rows

    print("\nВосстановление исходной геометрии:")
    try:
        _, elapsed = client.cache_rebuild(
            moe_slots=original.moe_cache_size,
            kv_tokens=original.num_pages,
            swa_tokens=original.num_swa_pages,
            wait_s=120.0,
        )
        print(f"  восстановлено за {elapsed:.2f} c, свободно VRAM {metrics.gpu_stats()['free_mb']} MB")
        report["restored"] = True
    except FreeTokenError as exc:
        print(f"  !! не удалось восстановить: {exc}")
        report["restored"] = False
        report["restore_error"] = str(exc)

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"phase1b_min_chat_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
