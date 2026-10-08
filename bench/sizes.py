"""Замер: сколько времени занимает кадр при разных размере и числе шагов.

Отвечает на вопрос «насколько меньше можно рисовать, чтобы стало заметно
быстрее». Каждый кадр пишется в ComfyUI как обычно, но в базу не попадает.

Сервер ComfyUI поднимается сам, если выключен.

Запуск::

    python bench\\sizes.py
    python bench\\sizes.py --only 512x512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402
from novel.comfy import ComfyClient, build_t2i_graph  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.machine import NovelMachine  # noqa: E402
from novel.models import ModelRegistry  # noqa: E402
from novel.settings import SettingsStore  # noqa: E402

#: (сторона, шагов) — от быстрого к обычному.
MATRIX: list[tuple[int, int]] = [
    (512, 8),
    (640, 16),
    (768, 20),
    (1024, 25),
]

PROMPT = ("interior of a dim, smoky medieval tavern, heavy wooden tables, "
          "glowing fireplace, cinematic lighting, detailed")


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Замер скорости кадров")
    parser.add_argument("--only", help="только один размер, например 512x512")
    parser.add_argument("--runs", type=int, default=1, help="повторов на размер")
    args = parser.parse_args()

    print("\n=== Скорость кадров: размер и шаги ===\n")
    db = config.DB_PATH
    machine = NovelMachine(
        __import__("novel.db", fromlist=["NovelDB"]).NovelDB(),
        SettingsStore(config.DATA_DIR / "settings.json"),
        ModelRegistry(),
    )
    print("поднимаю ComfyUI...")
    # Движок обязан быть погашен: вдвоём они не помещаются в видеопамять, и
    # ComfyUI уходит в системную память — шаг растёт с секунды до минуты с
    # лишним, а замер превращается в бессмыслицу. Один раз это уже случилось.
    engine = machine.stop_engine()
    if engine.get("stopped") or engine.get("already_stopped"):
        print(f"  движок погашен: {engine}")
    report = machine.ensure_comfy()
    if not report.get("ready"):
        print(f"  не удалось: {report.get('error') or report}")
        return 1
    print(f"  готов за {report.get('seconds', 0):.0f} c\n")

    client = ComfyClient()
    wanted = None
    if args.only:
        side = int(args.only.split("x")[0])
        wanted = side

    results: list[tuple[int, int, float]] = []
    for side, steps in MATRIX:
        if wanted and side != wanted:
            continue
        times: list[float] = []
        for _ in range(args.runs):
            graph = build_t2i_graph(
                PROMPT, width=side, height=side, seed=0, steps=steps,
                cfg=1.0, filename_prefix="novel_size_test",
            )
            started = time.time()
            result = client.generate(graph, timeout_s=1200.0)
            times.append(result.elapsed_s or (time.time() - started))
        seconds = min(times)
        megapixels = (side * side) / (1024 * 1024)
        print(f"  {side}x{side}, {steps:>2} шагов: {seconds:6.1f} c "
              f"({megapixels:.2f} Мпикс, {seconds / max(megapixels, 0.01) / 25 * 25:.0f} c на Мпикс)")
        results.append((side, steps, seconds))

    if results:
        base = next((r for r in results if r[0] == 1024), results[-1])
        print(f"\n  за базу берём {base[0]}x{base[0]} за {base[2]:.0f} c")
        for side, steps, seconds in results:
            gain = base[2] / seconds if seconds else 0
            print(f"    {side:>4}x{side:<4} {steps:>2} шагов — быстрее в {gain:.1f} раза "
                  f"({seconds:.0f} c вместо {base[2]:.0f} c)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
