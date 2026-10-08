"""Снимок состояния машины и обоих движков — по одной команде.

Запуск::

    python bench\\probe.py
    python bench\\probe.py --json

Ничего не меняет: только читает ``nvidia-smi``, ``/health``, ``/v1/stats``,
``/v1/cache/status`` и ``/system_stats``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config, metrics  # noqa: E402
from novel.console import setup_console  # noqa: E402
from novel.freetoken import FreeTokenClient, FreeTokenError  # noqa: E402


def comfy_probe() -> dict[str, object]:
    """Проверяет, отвечает ли ComfyUI, и возвращает его версию."""
    url = f"{config.COMFY_BASE_URL}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            doc = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"alive": False, "error": str(exc)}
    system = doc.get("system") or {}
    devices = doc.get("devices") or [{}]
    return {
        "alive": True,
        "comfyui_version": system.get("comfyui_version"),
        "python_version": (system.get("python_version") or "").split()[0],
        "device": devices[0].get("name"),
        "vram_total_mb": round((devices[0].get("vram_total") or 0) / (1024 * 1024)),
        "vram_free_mb": round((devices[0].get("vram_free") or 0) / (1024 * 1024)),
    }


def collect() -> dict[str, object]:
    """Собирает полный снимок состояния."""
    result: dict[str, object] = {"gpu": metrics.gpu_stats(), "memory": metrics.ram_stats()}

    client = FreeTokenClient()
    ft: dict[str, object] = {"url": config.FREETOKEN_BASE_URL}
    try:
        ft["health"] = client.health()
        ft["stats"] = client.stats()
        ft["cache"] = client.cache_status()
        geometry = client.geometry()
        sizes = geometry.bytes_for()
        ft["cache_sizes_mb"] = {key: round(value / (1024 * 1024)) for key, value in sizes.items()}
        ft["cache_human"] = geometry.describe()
    except FreeTokenError as exc:
        ft["error"] = str(exc)
    result["freetoken"] = ft

    result["comfyui"] = comfy_probe()

    ft_pid = metrics.pid_on_port(1919)
    comfy_pid = metrics.pid_on_port(8188)
    pids = {}
    if ft_pid:
        pids["freetoken"] = ft_pid
    if comfy_pid:
        pids["comfyui"] = comfy_pid
    result["pids"] = pids
    result["rss_mb"] = {name: metrics.process_rss_mb(pid) for name, pid in pids.items()}
    result["gpu_processes"] = [
        {"pid": pid, "name": name, "used": mem} for pid, name, mem in metrics.gpu_processes()
    ]
    return result


def render(doc: dict[str, object]) -> str:
    """Печатает снимок таблицей."""
    gpu = doc["gpu"]
    mem = doc["memory"]
    lines = [
        "",
        f"GPU    занято {gpu['used_mb']:>6} MB   свободно {gpu['free_mb']:>6} MB   "
        f"всего {gpu['total_mb']:>6} MB   загрузка {gpu['util_pct']}%",
        f"RAM    свободно {mem['ram_available_mb']:>6} MB   всего {mem['ram_total_mb']:>6} MB",
        f"COMMIT занято {mem['commit_used_mb']:>6} MB   предел {mem['commit_limit_mb']:>6} MB",
        "",
    ]

    ft = doc["freetoken"]
    if "error" in ft:
        lines.append(f"FreeToken  НЕДОСТУПЕН: {ft['error']}")
    else:
        health = ft["health"]
        stats = ft["stats"]
        lines.append(
            f"FreeToken  {ft['url']}  модель={health.get('model')}  "
            f"uptime={health.get('uptime_s')}s  версия={health.get('version')}"
        )
        lines.append(
            f"           VRAM движка {stats.get('vram_bytes', 0) / (1024 ** 3):.2f} GB   "
            f"ctx={stats.get('model', {}).get('ctx')}   "
            f"запросов активно={stats.get('requests', {}).get('active')}"
        )
        lines.append(f"           кэш: {ft['cache_human']}")
        rebuild = (ft["cache"].get("last_rebuild") or {})
        lines.append(
            f"           последняя пересборка: {rebuild.get('status')} "
            f"moe={rebuild.get('moe_cache_size')} kv={rebuild.get('num_pages')} "
            f"swa={rebuild.get('num_swa_pages')}"
        )

    comfy = doc["comfyui"]
    lines.append("")
    if comfy.get("alive"):
        lines.append(
            f"ComfyUI    {config.COMFY_BASE_URL}  версия={comfy['comfyui_version']}  "
            f"python={comfy['python_version']}  свободно {comfy['vram_free_mb']} MB"
        )
    else:
        lines.append(f"ComfyUI    НЕ ЗАПУЩЕН ({config.COMFY_BASE_URL}): {comfy.get('error')}")

    lines.append("")
    ours = [(pid, name, used) for pid, name, used in doc["gpu_processes"] if used.isdigit()]
    if ours:
        for pid, name, used in sorted(ours, key=lambda row: -int(row[2])):
            rss = doc["rss_mb"].get(name)
            lines.append(
                f"  pid {pid:<7} {int(used):>7} MB  {name}"
                f"{'' if rss is None else f'  RSS {rss} MB'}"
            )
    else:
        lines.append(
            f"  nvidia-smi не отдаёт объёмы по процессам "
            f"(видно {len(doc['gpu_processes'])} записей без прав)"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Снимок состояния NovelForge-окружения")
    parser.add_argument("--json", action="store_true", help="печатать сырой JSON")
    args = parser.parse_args()

    doc = collect()
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
