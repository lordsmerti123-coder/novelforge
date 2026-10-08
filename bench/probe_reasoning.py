"""Диагностика reasoning-моделей: куда уходит ответ и как отключить размышления.

У reasoning-моделей (Qwen3.5/3.6, gpt-oss) часть вывода уходит в канал
размышлений. Если бюджет ответа мал, модель тратит его целиком на размышления и
до текста не доходит — снаружи это выглядит как пустой ответ при ненулевых
токенах.

Скрипт показывает, что именно вернул движок, какие рычаги управления
размышлениями он объявляет в ``/v1/stats``, и пробует отключить их.

Запуск::

    python bench\\probe_reasoning.py --model <каталог моделей>\\Qwen3.6-35B-A3B-NVFP4
    python bench\\probe_reasoning.py --model <каталог моделей>\\gpt-oss-20b --keep-running
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.comfy import ComfyClient, ComfyError  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.engine import EngineConfig, EngineController  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402

PROMPT = "Я вхожу в таверну и осматриваюсь. Что я вижу?"


def dump(label: str, value: Any) -> None:
    """Печатает блок с заголовком."""
    print(f"\n--- {label} ---")
    print(json.dumps(value, ensure_ascii=False, indent=2)[:2500])


def raw_request(
    client: FreeTokenClient,
    *,
    body_extra: dict[str, Any] | None = None,
    max_tokens: int = 400,
) -> dict[str, Any]:
    """Небуферный запрос с полным ответом движка."""
    body: dict[str, Any] = {
        "model": config.FREETOKEN_SERVED_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "stream": False,
    }
    if body_extra:
        body.update(body_extra)
    started = time.time()
    try:
        doc = client._request("POST", "/v1/chat/completions", body, timeout_s=600.0)
    except FreeTokenError as exc:
        return {"error": str(exc)}
    elapsed = time.time() - started
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = doc.get("usage") or {}
    return {
        "elapsed_s": round(elapsed, 2),
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(message.get("content") or ""),
        "reasoning_chars": len(
            message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or ""
        ),
        "message_keys": sorted(message.keys()),
        "usage": usage,
        "content_head": (message.get("content") or "")[:200].replace("\n", " "),
        "reasoning_head": str(
            message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or ""
        )[:200].replace("\n", " "),
    }


def main() -> int:
    setup_console()
    config.ensure_dirs()

    parser = argparse.ArgumentParser(description="Диагностика reasoning-модели")
    parser.add_argument("--model", required=True, help="путь к чекпоинту")
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--keep-running", action="store_true", help="не останавливать движок в конце")
    args = parser.parse_args()

    path = Path(args.model)
    controller = EngineController(port=1919)
    client = FreeTokenClient()

    print(f"\n=== Диагностика {path.name} ===\n")

    if controller.port_pid() is not None:
        stop = controller.stop(timeout_s=30.0)
        print(f"движок остановлен за {stop['stop_seconds']} c")
    comfy = ComfyClient()
    if comfy.is_alive():
        try:
            comfy.free()
            print("модели ComfyUI выгружены")
        except ComfyError:
            pass
    metrics.wait_for_free_vram(threshold_mb=9000, timeout_s=45.0)

    cfg = EngineConfig(name=f"diag_{path.name}", memory_ratio=args.ratio, model_path=path)
    started = time.time()
    result = controller.cold_start(cfg, timeout_s=1200.0)
    if not result.get("ready"):
        errors = (result.get("timeline") or {}).get("errors") or []
        print(f"ОТКАЗ: {errors[-1][:400] if errors else 'причина неизвестна'}")
        return 1
    print(f"готов за {result['timeline']['spawn_to_ready_s']} c, "
          f"первый токен через {(result.get('first_token') or {}).get('spawn_to_first_token_s')} c")
    print(f"кэш: {result.get('geometry')}")

    stats = client.stats()
    dump("model.sampling из чекпоинта", (stats.get("model") or {}).get("sampling"))
    try:
        geometry = client.cache_status().get("geometry") or {}
        dump("рычаги размышлений (geometry.reasoning)", geometry.get("reasoning"))
    except FreeTokenError as exc:
        print(f"не удалось прочитать geometry: {exc}")

    print("\n=== Запрос по умолчанию ===")
    print(json.dumps(raw_request(client, max_tokens=400), ensure_ascii=False, indent=2))

    print("\n=== С увеличенным бюджетом (1600 токенов) ===")
    print(json.dumps(raw_request(client, max_tokens=1600), ensure_ascii=False, indent=2))

    print("\n=== С отключёнными размышлениями ===")
    for extra in (
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"thinking_mode": "disabled"}},
        {"reasoning_effort": "low"},
    ):
        label = json.dumps(extra, ensure_ascii=False)
        print(f"\n  попытка {label}")
        outcome = raw_request(client, body_extra=extra, max_tokens=400)
        print("  " + json.dumps(outcome, ensure_ascii=False))

    if not args.keep_running:
        print("\nостанавливаю движок")
        controller.stop(timeout_s=30.0)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
