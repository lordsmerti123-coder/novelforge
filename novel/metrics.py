"""Замеры видеопамяти, оперативной памяти и времени.

Все объёмы возвращаются целыми мегабайтами: так их можно сравнивать глазами и
писать в CSV без потери точности.

Наблюдение за VRAM идёт через ``nvidia-smi``: у NVML и WDDM на Windows 11
расходятся показания, а ``nvidia-smi`` — то, что видит и пользователь, и
планировщик памяти драйвера.
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
import time
from dataclasses import dataclass, field
from statistics import mean

import psutil

from novel import config

_CREATE_NO_WINDOW = 0x08000000


def _run(argv: list[str], timeout: float = 15.0) -> str:
    """Запускает процесс без мелькающего окна консоли и возвращает stdout."""
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} завершился с кодом {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


# --- GPU ---------------------------------------------------------------------


def gpu_stats() -> dict[str, int]:
    """Снимок занятой и свободной видеопамяти, мегабайты.

    @returns: ``total_mb``, ``used_mb``, ``free_mb``, ``util_pct``.
    @raises RuntimeError: если ``nvidia-smi`` недоступен или вернул ошибку.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=config.NVIDIA_SMI_TIMEOUT_S,
    )
    first = out.strip().splitlines()[0]
    total, used, free, util = (int(part.strip()) for part in first.split(","))
    return {"total_mb": total, "used_mb": used, "free_mb": free, "util_pct": util}


def gpu_processes() -> list[tuple[int, str, str]]:
    """Список процессов, держащих видеопамять: ``(pid, имя, занятая память)``.

    Для чужих процессов драйвер отдаёт ``[Insufficient Permissions]`` вместо
    объёма — такие строки остаются в списке как есть.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        timeout=config.NVIDIA_SMI_TIMEOUT_S,
    )
    rows: list[tuple[int, str, str]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, name, mem = (part.strip() for part in line.split(",", 2))
        rows.append((int(pid), name, mem))
    return rows


# --- RAM и commit ------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    """Структура Win32 ``MEMORYSTATUSEX`` из ``sysinfoapi.h``."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_stats() -> dict[str, int]:
    """Занятый и предельный объём зафиксированной памяти (commit charge).

    Именно этот счётчик упирается в потолок, когда FreeToken закрепляет веса
    экспертов в RAM: физическая память может выглядеть свободной, а commit —
    уже нет.

    ``MEMORYSTATUSEX`` выбран вместо ``PERFORMANCE_INFORMATION``: на этой
    сборке Windows 11 вторая структура возвращает меньше байт, чем объявлено в
    заголовке, и поля читаются со сдвигом.

    @returns: ``commit_used_mb``, ``commit_limit_mb``, ``commit_available_mb``.
    """
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    mb = 1024 * 1024
    return {
        "commit_used_mb": (status.ullTotalPageFile - status.ullAvailPageFile) // mb,
        "commit_limit_mb": status.ullTotalPageFile // mb,
        "commit_available_mb": status.ullAvailPageFile // mb,
    }


def ram_stats() -> dict[str, int]:
    """Снимок оперативной памяти, мегабайты.

    @returns: ``total_mb``, ``available_mb``, ``used_mb`` плюс поля commit.
    """
    vm = psutil.virtual_memory()
    mb = 1024 * 1024
    stats = {
        "ram_total_mb": vm.total // mb,
        "ram_available_mb": vm.available // mb,
        "ram_used_mb": (vm.total - vm.available) // mb,
    }
    stats.update(commit_stats())
    return stats


def process_rss_mb(pid: int) -> int | None:
    """Занятая процессом физическая память в мегабайтах, ``None`` если он умер."""
    try:
        return psutil.Process(pid).memory_info().rss // (1024 * 1024)
    except psutil.Error:
        return None


def find_pids(cmdline_contains: str) -> list[int]:
    """PID-ы процессов, в командной строке которых есть подстрока.

    @param cmdline_contains: подстрока для поиска, регистр не важен.
    @returns: отсортированный список PID-ов.
    """
    needle = cmdline_contains.lower()
    found: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except psutil.Error:
            continue
        if needle in cmdline.lower():
            found.append(proc.info["pid"])
    return sorted(found)


def pid_on_port(port: int) -> int | None:
    """PID процесса, слушающего локальный TCP-порт, или ``None``."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
            return conn.pid
    return None


# --- Снимок и замерщик -------------------------------------------------------


def snapshot(extra_pids: dict[str, int] | None = None) -> dict[str, object]:
    """Полный одномоментный снимок памяти.

    @param extra_pids: именованные PID-ы, для которых нужен RSS.
    @returns: словарь с полями GPU, RAM, commit и RSS процессов.
    """
    data: dict[str, object] = {"ts": time.time()}
    data.update(gpu_stats())
    data.update(ram_stats())
    if extra_pids:
        data["rss_mb"] = {
            name: process_rss_mb(pid) for name, pid in extra_pids.items()
        }
    return data


@dataclass
class Sample:
    """Один отсчёт фонового замерщика."""

    t: float
    gpu_used_mb: int
    gpu_free_mb: int
    gpu_util_pct: int
    ram_available_mb: int
    commit_used_mb: int


@dataclass
class Summary:
    """Сводка по серии отсчётов."""

    duration_s: float
    samples: int
    gpu_used_mb: dict[str, float] = field(default_factory=dict)
    gpu_free_mb: dict[str, float] = field(default_factory=dict)
    ram_available_mb: dict[str, float] = field(default_factory=dict)
    commit_used_mb: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Представление для записи в JSON."""
        return {
            "duration_s": round(self.duration_s, 3),
            "samples": self.samples,
            "gpu_used_mb": self.gpu_used_mb,
            "gpu_free_mb": self.gpu_free_mb,
            "ram_available_mb": self.ram_available_mb,
            "commit_used_mb": self.commit_used_mb,
        }


def _series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 1),
        "mean": round(mean(values), 1),
        "max": round(max(values), 1),
    }


class Sampler:
    """Опрашивает память в фоновом потоке, пока идёт измеряемая операция.

    Пример::

        with Sampler() as sampler:
            do_something_slow()
        print(sampler.summary().as_dict())
    """

    def __init__(self, interval_s: float = config.SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._errors: list[str] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                gpu = gpu_stats()
                stats = ram_stats()
                self.samples.append(
                    Sample(
                        t=time.time(),
                        gpu_used_mb=gpu["used_mb"],
                        gpu_free_mb=gpu["free_mb"],
                        gpu_util_pct=gpu["util_pct"],
                        ram_available_mb=stats["ram_available_mb"],
                        commit_used_mb=stats["commit_used_mb"],
                    )
                )
            except (RuntimeError, OSError) as exc:
                self._errors.append(str(exc))
            self._stop.wait(self.interval_s)

    def start(self) -> "Sampler":
        """Запускает фоновый опрос и делает первый отсчёт сразу."""
        self._started_at = time.time()
        self._take_one()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _take_one(self) -> None:
        try:
            gpu = gpu_stats()
            stats = ram_stats()
        except (RuntimeError, OSError) as exc:
            self._errors.append(str(exc))
            return
        self.samples.append(
            Sample(
                t=time.time(),
                gpu_used_mb=gpu["used_mb"],
                gpu_free_mb=gpu["free_mb"],
                gpu_util_pct=gpu["util_pct"],
                ram_available_mb=stats["ram_available_mb"],
                commit_used_mb=stats["commit_used_mb"],
            )
        )

    def stop(self) -> None:
        """Останавливает опрос и делает последний отсчёт."""
        self._stopped_at = time.time()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._take_one()

    def __enter__(self) -> "Sampler":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def errors(self) -> list[str]:
        """Ошибки опроса, накопленные за прогон."""
        return list(self._errors)

    def summary(self) -> Summary:
        """Пики и средние по всем отсчётам."""
        ended = self._stopped_at or time.time()
        return Summary(
            duration_s=ended - self._started_at,
            samples=len(self.samples),
            gpu_used_mb=_series([s.gpu_used_mb for s in self.samples]),
            gpu_free_mb=_series([s.gpu_free_mb for s in self.samples]),
            ram_available_mb=_series([s.ram_available_mb for s in self.samples]),
            commit_used_mb=_series([s.commit_used_mb for s in self.samples]),
        )


def wait_for_free_vram(threshold_mb: int, timeout_s: float) -> tuple[bool, int]:
    """Ждёт, пока свободной видеопамяти станет не меньше порога.

    @param threshold_mb: целевой объём свободной VRAM.
    @param timeout_s: сколько секунд ждать.
    @returns: ``(дождались, последнее наблюдённое значение)``.
    """
    deadline = time.time() + timeout_s
    last = 0
    while True:
        last = gpu_stats()["free_mb"]
        if last >= threshold_mb:
            return True, last
        if time.time() >= deadline:
            return False, last
        time.sleep(0.5)
