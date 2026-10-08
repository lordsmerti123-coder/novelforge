"""Запуск, остановка и наблюдение движка FreeToken.

Оркестратор владеет жизненным циклом движка: на 12 GB VRAM текстовая модель и
Qwen-Image-2.1 не помещаются одновременно, поэтому перед генерацией картинки
движок останавливается, а после — поднимается заново.

Остановка идёт по дереву процессов. Это важно: воркер, который держит банки
экспертов в RAM, запущен через ``multiprocessing.spawn`` и в своей командной
строке слова ``freetoken`` не содержит — по имени его не найти.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from novel import config, metrics
from novel.freetoken import FreeTokenClient, FreeTokenError

_CREATE_FLAGS = (
    subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
)


@dataclass
class EngineConfig:
    """Одна конфигурация запуска движка."""

    name: str
    memory_ratio: float
    extra_args: list[str] = field(default_factory=list)
    moe_strategy: str = "offload"
    model_path: Path = config.FREETOKEN_MODEL_PATH
    port: int = 1919
    host: str = "127.0.0.1"

    def argv(self, ft_exe: Path) -> list[str]:
        """Команда запуска движка.

        @param ft_exe: путь к ``ft.exe`` из установки FreeToken.
        @returns: список аргументов процесса.
        """
        return [
            str(ft_exe),
            "serve",
            "--model",
            str(self.model_path),
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--moe-strategy",
            self.moe_strategy,
            "--max-running-requests",
            "4",
            "--memory-ratio",
            str(self.memory_ratio),
            *self.extra_args,
        ]


@dataclass
class StartupTimeline:
    """Хронология холодного старта движка."""

    spawned_at: float
    first_http_at: float | None = None
    ready_at: float | None = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Представление для отчёта."""
        phase_times: dict[str, float] = {}
        for entry in self.phases:
            phase_times.setdefault(entry["phase"], entry["t"])
        doc = {
            "spawn_to_first_http_s": (
                None if self.first_http_at is None else round(self.first_http_at - self.spawned_at, 2)
            ),
            "spawn_to_ready_s": (
                None if self.ready_at is None else round(self.ready_at - self.spawned_at, 2)
            ),
            "phase_first_seen_s": {k: round(v, 2) for k, v in phase_times.items()},
            "phase_log": self.phases[-40:],
            "errors": self.errors,
        }
        return doc


def engine_processes() -> list[psutil.Process]:
    """Процессы движка вместе с потомками.

    @returns: список процессов; пустой, если движок не запущен.
    """
    roots: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
        except psutil.Error:
            continue
        if "freetoken" not in cmdline:
            continue
        if "serve" not in cmdline and "cli" not in cmdline:
            continue
        if "desktop" in (proc.info["name"] or "").lower():
            continue
        roots.append(proc)

    tree: dict[int, psutil.Process] = {}
    for root in roots:
        tree[root.pid] = root
        try:
            for child in root.children(recursive=True):
                tree[child.pid] = child
        except psutil.Error:
            continue
    return list(tree.values())


def engine_rss_mb() -> dict[str, int]:
    """Память всех процессов движка.

    Меряется по дереву целиком: сам ``ft.exe`` — это лёгкий лаунчер на несколько
    мегабайт, а банки экспертов держит вложенный воркер, поэтому RSS одного
    процесса ничего не говорит.

    @returns: ``sum_mb`` по всему дереву и ``max_mb`` самого тяжёлого процесса.
    """
    total = 0
    heaviest = 0
    for proc in engine_processes():
        try:
            rss = proc.memory_info().rss // (1024 * 1024)
        except psutil.Error:
            continue
        total += rss
        heaviest = max(heaviest, rss)
    return {"sum_mb": total, "max_mb": heaviest}


class EngineController:
    """Управляет одним экземпляром сервера FreeToken на заданном порту."""

    def __init__(
        self,
        ft_exe: Path = config.FREETOKEN_FT_EXE,
        port: int = 1919,
        log_dir: Path = config.LOGS_DIR,
    ) -> None:
        self.ft_exe = ft_exe
        self.port = port
        self.log_dir = log_dir
        self.client = FreeTokenClient(f"http://127.0.0.1:{port}", timeout_s=15.0)
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None

    # --- наблюдение ---------------------------------------------------------

    def is_healthy(self) -> bool:
        """Отвечает ли сервер на ``/health`` статусом ``ok``."""
        try:
            return self.client.health().get("status") == "ok"
        except (FreeTokenError, json.JSONDecodeError):
            return False

    def health(self) -> dict[str, Any]:
        """Сырой ответ ``/health``."""
        return self.client.health()

    def port_pid(self) -> int | None:
        """PID процесса, слушающего порт движка."""
        return metrics.pid_on_port(self.port)

    def stop(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Останавливает движок вместе со всеми потомками.

        @param timeout_s: сколько ждать штатного завершения перед принудительным.
        @returns: отчёт с числом убитых процессов и освободившейся памятью.
        """
        before = metrics.gpu_stats()
        procs = engine_processes()
        pids = [proc.pid for proc in procs]
        report: dict[str, Any] = {"pids": pids, "gpu_free_before_mb": before["free_mb"]}

        started = time.time()
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                continue

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not any(proc.is_running() for proc in procs):
                break
            time.sleep(0.25)

        killed: list[int] = []
        for proc in procs:
            try:
                if proc.is_running():
                    proc.kill()
                    killed.append(proc.pid)
            except psutil.Error:
                continue
        report["force_killed"] = killed
        report["stop_seconds"] = round(time.time() - started, 2)

        # Драйвер отдаёт VRAM не мгновенно после смерти процесса. Порог берётся с
        # оглядкой на общий объём: просить больше, чем есть на карте, бессмысленно.
        total_mb = before["total_mb"]
        target = min(before["free_mb"] + 6000, total_mb - 200)
        freed, last = metrics.wait_for_free_vram(threshold_mb=target, timeout_s=30.0)
        report["vram_target_mb"] = target
        report["vram_settled"] = freed
        report["gpu_free_after_mb"] = last
        if self.process is not None:
            self.process = None
        return report

    def wait_port_free(self, timeout_s: float = 60.0) -> bool:
        """Ждёт, пока порт движка перестанет быть занятым.

        @returns: ``True``, если порт освободился.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.port_pid() is None:
                return True
            time.sleep(0.5)
        return self.port_pid() is None

    # --- запуск -------------------------------------------------------------

    def start(self, cfg: EngineConfig) -> subprocess.Popen[bytes]:
        """Запускает движок отсоединённым процессом.

        Процесс переживает выход оркестратора, поэтому его PID нужно хранить:
        осиротевший движок продолжит держать VRAM.

        @param cfg: конфигурация запуска.
        @returns: handle запущенного процесса.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"engine_{cfg.name}.log"
        log_file = self.log_path.open("ab")
        argv = cfg.argv(self.ft_exe)
        self.process = subprocess.Popen(
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_CREATE_FLAGS,
            cwd=str(self.ft_exe.parent),
        )
        return self.process

    def wait_ready(
        self,
        spawned_at: float,
        timeout_s: float = 900.0,
        poll_s: float = 0.5,
        watch_memory: bool = True,
    ) -> StartupTimeline:
        """Ждёт готовности движка и собирает хронологию загрузки.

        @param spawned_at: момент запуска процесса, из :func:`time.time`.
        @param timeout_s: предел ожидания.
        @returns: хронология с временами по фазам загрузки.
        """
        timeline = StartupTimeline(spawned_at=spawned_at)
        sampler = metrics.Sampler(interval_s=2.0) if watch_memory else None
        if sampler is not None:
            sampler.start()

        deadline = time.time() + timeout_s
        last_phase: str | None = None
        saw_http = False
        try:
            while time.time() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    timeline.errors.append(
                        f"процесс движка завершился с кодом {self.process.returncode}"
                    )
                    if self.log_path and self.log_path.exists():
                        tail = self.log_path.read_text(encoding="utf-8", errors="replace")
                        timeline.errors.append("хвост лога: " + tail[-800:])
                    return timeline
                try:
                    doc = self.client.health()
                except FreeTokenError:
                    time.sleep(poll_s)
                    continue
                if not saw_http:
                    saw_http = True
                    timeline.first_http_at = time.time()
                status = doc.get("status")
                phase = doc.get("phase") or status
                if phase != last_phase:
                    last_phase = phase
                    progress = doc.get("progress") or {}
                    timeline.phases.append(
                        {
                            "t": round(time.time() - spawned_at, 2),
                            "phase": phase,
                            "status": status,
                            "done_bytes": progress.get("done_bytes", 0),
                            "total_bytes": progress.get("total_bytes", 0),
                        }
                    )
                if status == "ok":
                    timeline.ready_at = time.time()
                    return timeline
                if status == "error":
                    timeline.errors.append(f"движок сообщил об ошибке: {doc}")
                    return timeline
                time.sleep(poll_s)
            timeline.errors.append(f"движок не стал готов за {timeout_s} c")
            return timeline
        finally:
            if sampler is not None:
                sampler.stop()

    def cold_start(self, cfg: EngineConfig, timeout_s: float = 900.0) -> dict[str, Any]:
        """Полный цикл: запуск, ожидание готовности, первый токен.

        @returns: отчёт с хронологией, памятью на каждом этапе и временем до
            первого токена.
        """
        if self.port_pid() is not None:
            # Порт занят — обычно прежним движком. Возвращаем обычный отчёт об
            # ошибке, а не исключение: вызывающий код показывает причину в
            # интерфейсе, а не падает с трассировкой.
            return {
                "ready": False,
                "error": f"порт {self.port} занят — движок не остановлен",
                "seconds": 0.0,
                "timeline": {
                    "errors": [f"порт {self.port} занят другим процессом"],
                    "port_pid": self.port_pid(),
                },
            }

        gpu_before = metrics.gpu_stats()
        ram_before = metrics.ram_stats()
        spawned_at = time.time()
        self.start(cfg)
        pid = self.process.pid if self.process else None
        timeline = self.wait_ready(spawned_at, timeout_s=timeout_s)
        report: dict[str, Any] = {
            "config": {
                "name": cfg.name,
                "memory_ratio": cfg.memory_ratio,
                "extra_args": cfg.extra_args,
                "argv": cfg.argv(self.ft_exe),
            },
            "pid": pid,
            "log": str(self.log_path) if self.log_path else None,
            "gpu_free_before_mb": gpu_before["free_mb"],
            "ram_available_before_mb": ram_before["ram_available_mb"],
            "timeline": timeline.as_dict(),
            "ready": timeline.ready_at is not None,
        }
        if not report["ready"]:
            gpu_after = metrics.gpu_stats()
            report["gpu_free_after_mb"] = gpu_after["free_mb"]
            report["rss"] = engine_rss_mb()
            return report

        time.sleep(2.0)
        gpu_after = metrics.gpu_stats()
        ram_after = metrics.ram_stats()
        report["gpu_free_after_mb"] = gpu_after["free_mb"]
        report["gpu_used_after_mb"] = gpu_after["used_mb"]
        report["ram_available_after_mb"] = ram_after["ram_available_mb"]
        report["commit_used_after_mb"] = ram_after["commit_used_mb"]
        report["rss"] = engine_rss_mb()
        try:
            report["geometry"] = self.client.geometry().describe()
        except FreeTokenError as exc:
            report["geometry_error"] = str(exc)

        # Первый токен — это ещё и прогретый CUDA-граф, и прогрев кэша экспертов.
        try:
            first = self.client.chat_stream(
                [
                    {"role": "system", "content": "Отвечай одним коротким предложением."},
                    {"role": "user", "content": "Привет! Ты готов?"},
                ],
                max_tokens=16,
                temperature=0.7,
            )
            report["first_token"] = {
                "ttft_s": None if first.ttft_s is None else round(first.ttft_s, 3),
                "elapsed_s": round(first.elapsed_s, 3),
                "completion_tokens": first.completion_tokens,
                "spawn_to_first_token_s": round(time.time() - spawned_at, 2),
            }
        except FreeTokenError as exc:
            report["first_token"] = {"error": str(exc)}

        # vram_bytes движок заполняет только после первого запроса: до него поле
        # равно нулю, и замер по нему дал бы ложный ноль.
        try:
            report["engine_vram_mb"] = round(
                (self.client.stats().get("vram_bytes") or 0) / (1024 * 1024)
            )
        except FreeTokenError as exc:
            report["engine_vram_error"] = str(exc)

        return report


def start_engine_in_background(cfg: EngineConfig) -> threading.Thread:
    """Запускает движок в отдельном потоке, чтобы не блокировать GUI.

    @returns: поток, который завершится, когда движок станет готов.
    """
    controller = EngineController(port=cfg.port)
    thread = threading.Thread(target=controller.cold_start, args=(cfg,), daemon=True)
    thread.start()
    return thread
