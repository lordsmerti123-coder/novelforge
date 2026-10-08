"""Внешний движок: llama.cpp вместо FreeToken.

FreeToken не грузит GGUF, а на диске лежат десятки таких моделей. Рядом оказались
сборки llama.cpp — их кладёт LM Studio, — и они говорят на том же протоколе
``/v1/chat/completions``, что и FreeToken. Поэтому второй движок не требует
отдельного клиента: меняются адрес, имя модели и способ запуска.

Сборку приходится выбирать осторожно. У процессора Intel Core Ultra (Arrow Lake)
нет AVX-512, и сборка с ним падает с кодом ``0xC000001D`` — «недопустимая
инструкция». Поэтому кандидат не просто ищется по имени, а проверяется запуском
``--version``: молчащая сборка не годится, даже если файл на месте.

Модуль управляет только локальным llama.cpp. LM Studio поднимает свой сервер сам,
и для него нужен лишь адрес — см. :func:`is_external_url_alive`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from novel import config, metrics

_CREATE_FLAGS = (
    subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
)

#: Каталоги LM Studio со сборками llama.cpp. Версии Vulkan проверены на живом
#: запуске; сборки с AVX-512 молчат, поэтому проверяются, а не берутся на веру.
LMSTUDIO_BACKENDS = Path.home() / ".lmstudio" / "extensions" / "backends"

#: Сборки, которые стоит пробовать, в порядке предпочтения. Vulkan первым: он
#: использует видеокарту, а CUDA-сборки LM Studio оказались нерабочими.
BACKEND_PATTERNS = (
    "llama.cpp-win-x86_64-vulkan-avx2-*",
    "llama.cpp-win-x86_64-avx2-*",
)

#: Отдельные сборки вне LM Studio. Идут **последними**: проверка ``--version``
#: их пропускает, а падают они позже — при загрузке модели. Сборка с AVX-512 на
#: процессорах без него валится с ``0xC000001D`` уже на подборе памяти. Поэтому
#: сначала идут сборки, проверенные живой загрузкой. Свои пути задаются
#: переменной окружения ``NOVELFORGE_LLAMACPP_SERVERS`` через запятую.
STANDALONE_SERVERS = tuple(
    Path(p)
    for p in os.environ.get("NOVELFORGE_LLAMACPP_SERVERS", "").split(os.pathsep)
    if p.strip()
)


@dataclass
class ExternalConfig:
    """Одна конфигурация запуска llama.cpp."""

    model_path: Path
    port: int = 1919
    host: str = "127.0.0.1"
    #: Длина контекста. Больше — лучше помнит, но съедает видеопамять.
    context: int = 8192
    #: Сколько слоёв отдать видеокарте; 99 означает «все».
    gpu_layers: int = 99
    #: Одновременных предсказаний. Одно: каждая копия контекста ест память.
    parallel: int = 1
    #: Просить шаблон чата не размышлять. У Gemma-4 это убирает блок
    #: ``<|channel>thought`` целиком — проверено.
    disable_thinking: bool = True
    #: Бюджет размышлений в токенах. Нужен для моделей, чей шаблон не принимает
    #: ``enable_thinking``: у DeepSeek-R1 такого поля нет вовсе, и отключить
    #: размышления нельзя — только ограничить. ``-1`` без ограничений, ``0``
    #: обрывает сразу, но **ноль бесполезен**: размышления не исчезают, а
    #: переезжают в сам ответ («Хм, пользователь просит…»). Разумное значение —
    #: около 128: размышления укладываются в 387 символов, ответ приходит чистым,
    #: а время падает с 16.5 до 7.3 c.
    reasoning_budget: int = 256
    #: Имя, под которым модель видна в API.
    alias: str = "novelforge"
    extra_args: list[str] = field(default_factory=list)

    def argv(self, server: Path) -> list[str]:
        """Команда запуска сервера.

        @param server: путь к ``llama-server.exe``.
        @returns: список аргументов процесса.
        """
        args = [
            str(server),
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context),
            "-ngl", str(self.gpu_layers),
            "--parallel", str(self.parallel),
            # Шаблон чата из самой модели: без него сервер не знает её ролей.
            "--jinja",
            "-a", self.alias,
        ]
        if self.disable_thinking:
            args += ["--chat-template-kwargs", '{"enable_thinking": false}']
        if self.reasoning_budget >= 0:
            args += ["--reasoning-budget", str(self.reasoning_budget)]
        return args + list(self.extra_args)


def candidate_servers() -> list[Path]:
    """Все найденные ``llama-server.exe``: сначала проверенные, потом прочие.

    Проверенные — сборки LM Studio: на них живая загрузка модели проходила.
    Отдельные сборки идут последними, потому что ``--version`` их не разоблачает,
    а падают они уже при загрузке модели.

    @returns: пути к найденным сборкам.
    """
    found: list[Path] = []
    for pattern in BACKEND_PATTERNS:
        if not LMSTUDIO_BACKENDS.is_dir():
            break
        for folder in sorted(LMSTUDIO_BACKENDS.glob(pattern), reverse=True):
            server = folder / "llama-server.exe"
            if server.exists():
                found.append(server)
    found.extend(path for path in STANDALONE_SERVERS if path.exists())
    return found


def probe_server(server: Path, timeout_s: float = 20.0) -> bool:
    """Проверяет, что сборка запускается на этом процессоре.

    Сборка с AVX-512 на Arrow Lake молча падает, поэтому «файл есть» ничего не
    значит: нужен живой ответ на ``--version``.

    @param server: путь к ``llama-server.exe``.
    @param timeout_s: сколько ждать ответа.
    @returns: ``True``, если сборка рабочая.
    """
    try:
        done = subprocess.run(
            [str(server), "--version"],
            capture_output=True, timeout=timeout_s, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return b"version:" in (done.stdout or b"") + (done.stderr or b"")


def find_server(override: str | Path | None = None) -> Path | None:
    """Выбирает рабочую сборку llama.cpp.

    @param override: путь из настроек; если задан и рабочий, берётся он.
    @returns: путь к сборке либо ``None``, если ни одна не запускается.
    """
    if override:
        path = Path(override)
        if path.exists() and probe_server(path):
            return path
    for server in candidate_servers():
        if probe_server(server):
            return server
    return None


def llama_processes() -> list[psutil.Process]:
    """Процессы llama.cpp вместе с потомками.

    Имя процесса, а не командная строка: у llama.cpp нет вложенного воркера,
    который прячет своё имя, — в отличие от FreeToken.

    @returns: список процессов; пустой, если сервер не запущен.
    """
    tree: dict[int, psutil.Process] = {}
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info["name"] or "").lower()
        if not name.startswith("llama-server"):
            continue
        tree[proc.pid] = proc
        try:
            for child in proc.children(recursive=True):
                tree[child.pid] = child
        except psutil.Error:
            continue
    return list(tree.values())


def list_server_models(url: str, timeout_s: float = 15.0) -> list[str]:
    """Список моделей, который предлагает внешний сервер.

    У LM Studio свои имена моделей, и они не совпадают с путями к файлам: сервер
    ждёт ``имя-модели-в-нижнем-регистре``, а на диске лежит каталог с
    ``...\\Model-Name-...-Q8_K_P.gguf``. Поэтому список берётся у сервера, а не
    из обхода диска.

    @param url: базовый адрес сервера.
    @param timeout_s: сколько ждать ответа.
    @returns: идентификаторы моделей; пустой список, если сервер не ответил.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout_s) as response:
            doc = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return []
    data = doc.get("data")
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def is_external_url_alive(url: str, timeout_s: float = 5.0) -> bool:
    """Отвечает ли внешний сервер.

    Спрашивать только ``/health`` нельзя: у llama.cpp он есть, а **у LM Studio
    его нет** — тот отвечает ``Unexpected endpoint or method``. Проверено живьём,
    и на этом LM Studio выглядел бы мёртвым, хотя сервер работает. Поэтому
    вторым идёт ``/v1/models``, который есть у обоих.

    @param url: базовый адрес вида ``http://127.0.0.1:1234``.
    @returns: ``True``, если сервер готов.
    """
    base = url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
        if json.loads(body).get("status") == "ok":
            return True
    except (urllib.error.URLError, OSError, ValueError):
        pass
    except json.JSONDecodeError:
        pass

    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout_s) as response:
            doc = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(doc.get("data"), list)


class ExternalController:
    """Управляет одним экземпляром llama.cpp на заданном порту.

    Наружу выглядит как :class:`novel.engine.EngineController`: оркестратору
    важно уметь запустить сервер, дождаться готовности и остановить его перед
    генерацией картинок.
    """

    def __init__(self, server: Path, port: int = 1919, log_dir: Path | None = None) -> None:
        self.server = server
        self.port = port
        self.log_dir = log_dir or config.LOGS_DIR
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None

    # --- наблюдение ---------------------------------------------------------

    def is_healthy(self) -> bool:
        """Отвечает ли сервер на ``/health`` статусом ``ok``."""
        return is_external_url_alive(f"http://127.0.0.1:{self.port}")

    def health(self) -> dict[str, Any]:
        """Сырой ответ ``/health``."""
        url = f"http://127.0.0.1:{self.port}/health"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return {"status": "error", "error": str(exc)}

    def port_pid(self) -> int | None:
        """PID процесса, слушающего порт сервера."""
        return metrics.pid_on_port(self.port)

    def stop(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Останавливает сервер вместе с потомками.

        @param timeout_s: сколько ждать штатного завершения перед принудительным.
        @returns: отчёт с числом убитых процессов и освободившейся памятью.
        """
        before = metrics.gpu_stats()
        procs = llama_processes()
        report: dict[str, Any] = {
            "pids": [proc.pid for proc in procs],
            "gpu_free_before_mb": before["free_mb"],
        }
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

        total_mb = before["total_mb"]
        target = min(before["free_mb"] + 6000, total_mb - 200)
        freed, last = metrics.wait_for_free_vram(threshold_mb=target, timeout_s=30.0)
        report["vram_settled"] = freed
        report["gpu_free_after_mb"] = last
        self.process = None
        return report

    def wait_port_free(self, timeout_s: float = 60.0) -> bool:
        """Ждёт, пока порт перестанет быть занятым.

        @param timeout_s: предел ожидания.
        @returns: ``True``, если порт освободился.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.port_pid() is None:
                return True
            time.sleep(0.5)
        return self.port_pid() is None

    # --- запуск -------------------------------------------------------------

    def start(self, cfg: ExternalConfig) -> subprocess.Popen[bytes]:
        """Запускает сервер отсоединённым процессом.

        @param cfg: конфигурация запуска.
        @returns: handle запущенного процесса.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "engine_llamacpp.log"
        log_file = self.log_path.open("ab")
        self.process = subprocess.Popen(
            cfg.argv(self.server),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_CREATE_FLAGS,
            cwd=str(self.server.parent),
        )
        return self.process

    def wait_ready(
        self, spawned_at: float, timeout_s: float = 600.0, poll_s: float = 1.0,
    ) -> dict[str, Any]:
        """Ждёт готовности сервера.

        llama.cpp не рассказывает о фазах загрузки, как FreeToken, поэтому
        хронология короче: только время до первого ответа.

        @param spawned_at: момент запуска процесса, из :func:`time.time`.
        @param timeout_s: предел ожидания.
        @param poll_s: период опроса.
        @returns: хронология запуска.
        """
        timeline: dict[str, Any] = {"spawned_at": spawned_at, "phases": [], "errors": []}
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                text = ""
                if self.log_path and self.log_path.exists():
                    text = self.log_path.read_text(encoding="utf-8", errors="replace")[-800:]
                timeline["errors"].append(
                    f"процесс завершился с кодом {self.process.returncode}; хвост лога: {text}"
                )
                return timeline
            doc = self.health()
            if doc.get("status") == "ok":
                timeline["ready_at"] = time.time()
                timeline["ready_s"] = round(time.time() - spawned_at, 2)
                return timeline
            time.sleep(poll_s)
        timeline["errors"].append(f"сервер не стал готов за {timeout_s} c")
        return timeline

    def cold_start(self, cfg: ExternalConfig, timeout_s: float = 600.0) -> dict[str, Any]:
        """Полный цикл: запуск, ожидание готовности, первый токен.

        @param cfg: конфигурация запуска.
        @param timeout_s: предел ожидания готовности.
        @returns: отчёт с хронологией и временем до первого токена.
        """
        if self.port_pid() is not None:
            # См. EngineController.cold_start: занятый порт — обычная ошибка
            # запуска, о ней сообщают отчётом, а не исключением.
            return {
                "ready": False,
                "error": f"порт {self.port} занят — сервер не остановлен",
                "seconds": 0.0,
                "timeline": {
                    "errors": [f"порт {self.port} занят другим процессом"],
                    "port_pid": self.port_pid(),
                },
            }

        gpu_before = metrics.gpu_stats()
        spawned_at = time.time()
        self.start(cfg)
        timeline = self.wait_ready(spawned_at, timeout_s=timeout_s)
        report: dict[str, Any] = {
            "config": {"model_path": str(cfg.model_path), "argv": cfg.argv(self.server)},
            "server": str(self.server),
            "pid": self.process.pid if self.process else None,
            "log": str(self.log_path) if self.log_path else None,
            "gpu_free_before_mb": gpu_before["free_mb"],
            "timeline": timeline,
            "ready": "ready_at" in timeline,
        }
        if not report["ready"]:
            report["gpu_free_after_mb"] = metrics.gpu_stats()["free_mb"]
            return report
        report["gpu_free_after_mb"] = metrics.gpu_stats()["free_mb"]
        report["rss"] = sum(
            proc.memory_info().rss // (1024 * 1024) for proc in llama_processes()
            if proc.is_running()
        )
        return report
