"""Матрица моделей: холодный старт, скорость и соблюдение протокола.

Проверяются три модели, которые лежат локально и подходят движку по
архитектуре: Gemma-4, Qwen3.6 и gpt-oss. Для каждой измеряется холодный старт,
задержка до первого токена, скорость генерации и то, соблюдает ли модель
формат ответа с тегами.

Модели переключаются перезапуском движка: он обслуживает один чекпоинт за
запуск. Перед каждым запуском карта освобождается от ComfyUI.

Запуск::

    python bench\\model_matrix.py
    python bench\\model_matrix.py --only gemma4
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

MODELS_DIR = config.DEFAULT_MODELS_DIR
from novel.comfy import ComfyClient, ComfyError  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.engine import EngineConfig, EngineController, engine_rss_mb  # noqa: E402
from novel.formats import get_format  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402
from novel.models import preset_for  # noqa: E402
from novel.prompts import base_instruction  # noqa: E402
from novel.protocol import parse_reply  # noqa: E402

MODELS: list[tuple[str, Path]] = [
    ("gemma4", MODELS_DIR / "Gemma-4-26B-A4B-NVFP4"),
    ("qwen3_5_moe", MODELS_DIR / "Qwen3.6-35B-A3B-NVFP4"),
    ("gpt_oss", MODELS_DIR / "gpt-oss-20b"),
]

PROMPT = "Я вхожу в таверну и осматриваюсь. Что я вижу?"


def stamp() -> str:
    """Метка времени для имени файла отчёта."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def free_comfy() -> bool:
    """Выгружает модели ComfyUI, чтобы освободить карту под движок."""
    comfy = ComfyClient()
    if not comfy.is_alive():
        return False
    try:
        comfy.free()
    except ComfyError:
        return False
    metrics.wait_for_free_vram(threshold_mb=9000, timeout_s=30.0)
    return True


def probe(ft: FreeTokenClient, model_type: str) -> dict[str, Any]:
    """Один запрос к модели с проверкой формата ответа."""
    preset = preset_for(model_type)
    fmt = get_format("story")
    messages = [
        {"role": "system", "content": base_instruction(fmt)},
        {"role": "user", "content": PROMPT},
    ]
    started = time.time()
    try:
        result = ft.chat_stream(
            messages,
            max_tokens=int(preset.get("max_tokens", 500)),
            temperature=float(preset.get("temperature", 0.8)),
            timeout_s=600.0,
        )
    except FreeTokenError as exc:
        return {"ok": False, "error": str(exc)}

    parsed = parse_reply(result.text)
    return {
        "ok": True,
        "elapsed_s": round(result.elapsed_s, 2),
        "ttft_s": None if result.ttft_s is None else round(result.ttft_s, 2),
        "completion_tokens": result.completion_tokens,
        "decode_tps": round(result.decode_tokens_per_second, 2),
        "prompt_tokens": result.prompt_tokens,
        "format_ok": bool(parsed.prose.strip()),
        "has_scene": parsed.scene is not None,
        "parse_errors": parsed.errors,
        "prose_head": parsed.prose[:180].replace("\n", " "),
        "raw_head": result.text[:180].replace("\n", " "),
        "preset": {k: preset.get(k) for k in ("temperature", "top_p", "top_k", "note")},
        "probe_wall_s": round(time.time() - started, 2),
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Прогон трёх локальных моделей")
    parser.add_argument("--only", default=None, help="проверить только это семейство")
    parser.add_argument("--ratio", type=float, default=0.9, help="memory-ratio движка")
    parser.add_argument("--timeout", type=float, default=1200.0, help="предел ожидания старта")
    args = parser.parse_args()

    targets = [item for item in MODELS if args.only is None or item[0] == args.only]
    if not targets:
        print(f"неизвестное семейство: {args.only}")
        return 2

    controller = EngineController(port=1919)
    client = FreeTokenClient()
    report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds"), "runs": []}

    print("\n=== Матрица моделей ===\n")
    print(f"Свободно VRAM до начала: {metrics.gpu_stats()['free_mb']} MB")
    print(f"Свободно RAM до начала:  {metrics.ram_stats()['ram_available_mb']} MB\n")

    for model_type, path in targets:
        print(f"--- {path.name} ({model_type}) ---")
        if not path.exists():
            print("    каталог не найден, пропуск\n")
            report["runs"].append({"model": path.name, "error": "каталог не найден"})
            continue

        if controller.port_pid() is not None:
            stop = controller.stop(timeout_s=30.0)
            print(f"    движок остановлен за {stop['stop_seconds']} c")
        if free_comfy():
            print("    модели ComfyUI выгружены")

        # Драйвер отдаёт память не мгновенно; без паузы движок может не влезть.
        settled, free_mb = metrics.wait_for_free_vram(threshold_mb=9000, timeout_s=45.0)
        print(f"    свободно VRAM перед запуском: {free_mb} MB"
              + ("" if settled else " (мало, пробуем всё равно)"))

        cfg = EngineConfig(
            name=f"{model_type}_ratio{int(args.ratio * 100)}",
            memory_ratio=args.ratio,
            moe_strategy="offload",
            model_path=path,
        )
        started = time.time()
        result = controller.cold_start(cfg, timeout_s=args.timeout)
        result["wall_s"] = round(time.time() - started, 2)
        result["model"] = path.name
        result["model_type"] = model_type

        if not result.get("ready"):
            errors = (result.get("timeline") or {}).get("errors") or []
            reason = errors[-1][:300] if errors else "причина неизвестна"
            print(f"    ОТКАЗ: {reason}\n")
            result["error"] = reason
            report["runs"].append(result)
            continue

        timeline = result["timeline"]
        print(f"    готов за {timeline.get('spawn_to_ready_s')} c, "
              f"первый токен через {(result.get('first_token') or {}).get('spawn_to_first_token_s')} c")
        print(f"    кэш: {result.get('geometry')}")

        # Reasoning-модели по умолчанию тратят весь бюджет ответа на рассуждения
        # и до текста не доходят. Приложение выключает их тем же способом.
        reasoning = client.configure_reasoning("off")
        result["reasoning"] = reasoning
        print(f"    размышления: у модели по умолчанию {reasoning.get('model_default')}, "
              f"{'выключены' if reasoning.get('applied') else 'рычагов нет'}")

        result["probe"] = probe(client, model_type)
        if result["probe"].get("ok"):
            p = result["probe"]
            print(f"    ответ: {p['completion_tokens']} токенов за {p['elapsed_s']} c "
                  f"(ttft {p['ttft_s']} c, {p['decode_tps']} ток/с)")
            print(f"    формат: {'соблюдён' if p['format_ok'] else 'НАРУШЕН'}"
                  f"{', есть блок сцены' if p['has_scene'] else ''}"
                  f"{', ошибок разбора ' + str(len(p['parse_errors'])) if p['parse_errors'] else ''}")
            print(f"    текст: {p['prose_head'][:110]}")
        else:
            print(f"    ОШИБКА запроса: {result['probe'].get('error')}")

        result["rss"] = engine_rss_mb()
        result["gpu_free_after_mb"] = metrics.gpu_stats()["free_mb"]
        result["ram_available_after_mb"] = metrics.ram_stats()["ram_available_mb"]
        print(f"    RSS движка {result['rss']['sum_mb']} MB, "
              f"свободно VRAM {result['gpu_free_after_mb']} MB, "
              f"RAM {result['ram_available_after_mb']} MB\n")
        report["runs"].append(result)

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = config.MEASUREMENTS_DIR / f"model_matrix_{stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Итог ===")
    for run in report["runs"]:
        name = run.get("model", "?")
        if run.get("error") or not run.get("ready"):
            print(f"  {name:<28} ОТКАЗ: {str(run.get('error'))[:90]}")
            continue
        p = run.get("probe") or {}
        print(
            f"  {name:<28} старт {run['timeline']['spawn_to_ready_s']:>6} c  "
            f"ttft {str(p.get('ttft_s')):>5} c  "
            f"{str(p.get('decode_tps')):>6} ток/с  "
            f"формат {'ок' if p.get('format_ok') else 'НЕТ'}"
        )
    print(f"\nОтчёт: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
