"""Скрытые запуски процессов и аварийная остановка всего.

Две задачи:

* ни один дочерний процесс не должен показывать окно консоли — иначе работа в
  браузере превращается в мельтешение чёрных прямоугольников;
* должен существовать один вызов, который гасит всё: движок текста, сервер
  ComfyUI и осиротевшие воркеры.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from novel import metrics

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def hidden_kwargs() -> dict[str, Any]:
    """Аргументы ``subprocess``, при которых дочерний процесс не показывает окно.

    ``CREATE_NO_WINDOW`` подавляет консоль, ``startupinfo`` со ``SW_HIDE``
    страхует случай, когда флаг игнорируется — например, при ``DETACHED_PROCESS``
    Windows явно разрешает не создавать консоль, но не запрещает показать окно.

    @returns: словарь для передачи в ``subprocess.run``/``Popen``.
    """
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": CREATE_NO_WINDOW, "startupinfo": startupinfo}


def run_hidden(argv: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Запускает процесс без окна и возвращает результат.

    @param argv: команда и аргументы.
    @param timeout: предел ожидания в секундах.
    @returns: завершённый процесс.
    """
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, **hidden_kwargs()
    )


def spawn_detached(argv: list[str], log_path: Path, cwd: Path | None = None) -> int:
    """Запускает процесс, переживающий родителя, с записью вывода в файл.

    Вывод идёт в файл, а не в поток: во-первых, иначе окно консоли всё-таки
    появится, во-вторых, лог движка потом нужен для разбора отказов.

    @param argv: команда и аргументы.
    @param log_path: файл для объединённого stdout и stderr.
    @param cwd: рабочий каталог процесса.
    @returns: PID запущенного процесса.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    kwargs = hidden_kwargs()
    kwargs["creationflags"] |= DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        argv,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=None if cwd is None else str(cwd),
        **kwargs,
    )
    return process.pid


@dataclass
class KillReport:
    """Что именно удалось остановить."""

    engine_pids: list[int]
    comfy_pids: list[int]
    engine_seconds: float
    comfy_seconds: float
    gpu_free_before_mb: int
    gpu_free_after_mb: int

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса и журнала."""
        return {
            "engine_pids": self.engine_pids,
            "comfy_pids": self.comfy_pids,
            "engine_seconds": round(self.engine_seconds, 2),
            "comfy_seconds": round(self.comfy_seconds, 2),
            "gpu_free_before_mb": self.gpu_free_before_mb,
            "gpu_free_after_mb": self.gpu_free_after_mb,
            "freed_mb": self.gpu_free_after_mb - self.gpu_free_before_mb,
        }


#: Номера процессов движка, погашенных за время работы приложения. Нужны,
#: чтобы находить рабочих, чей родитель умер раньше: сами они дерева не знают.
_KILLED_ENGINE_PIDS: set[int] = set()


def _declared_parent(proc: "psutil.Process") -> int | None:
    """Родитель, объявленный рабочим процессом в своей командной строке.

    Рабочие FreeToken запускаются через ``multiprocessing.spawn`` и объявляют
    родителя как ``spawn_main(parent_pid=N)``. Слова FreeToken в их командной
    строке нет, поэтому по имени их не найти.

    @param proc: процесс для осмотра.
    @returns: номер объявленного родителя; None, если его нет.
    """
    try:
        cmdline = " ".join(proc.cmdline())
    except psutil.Error:
        return None
    marker = "spawn_main(parent_pid="
    at = cmdline.find(marker)
    if at < 0:
        return None
    digits = ""
    for char in cmdline[at + len(marker):]:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def _sweep_workers(parent_pids: set[int]) -> list[int]:
    """Гасит рабочих, чей родитель входит в переданный набор.

    Набор собирается только из процессов движка: иначе под метлу попадут чужие
    приложения на Python, у которых тоже есть ``multiprocessing.spawn``.

    @param parent_pids: номера процессов движка, живых или только что погашенных.
    @returns: номера погашенных рабочих.
    """
    if not parent_pids:
        return []
    doomed = []
    for proc in psutil.process_iter(["pid"]):
        try:
            if proc.info["pid"] in parent_pids:
                continue
        except psutil.Error:
            continue
        if _declared_parent(proc) in parent_pids:
            doomed.append(proc)
    swept: list[int] = []
    for proc in doomed:
        try:
            proc.kill()
            swept.append(proc.pid)
        except psutil.Error:
            continue
    return swept


def engine_pids() -> list[int]:
    """PID-ы процессов движка FreeToken вместе с воркерами."""
    found: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
            name = (proc.info["name"] or "").lower()
        except psutil.Error:
            continue
        if "desktop" in name:
            continue
        if "freetoken" not in cmdline:
            continue
        if "serve" not in cmdline and "cli" not in cmdline:
            continue
        found.append(proc.info["pid"])
        try:
            found.extend(child.pid for child in psutil.Process(proc.info["pid"]).children(recursive=True))
        except psutil.Error:
            continue
    # Рабочие по имени не находятся: слова FreeToken в их командной строке нет.
    # Они опознаются по объявленному родителю — живому или недавно погашенному.
    known = set(found) | _KILLED_ENGINE_PIDS
    if known:
        for proc in psutil.process_iter(["pid"]):
            try:
                if proc.info["pid"] in known:
                    continue
            except psutil.Error:
                continue
            if _declared_parent(proc) in known:
                found.append(proc.info["pid"])
    return sorted(set(found))


def stop_engine_processes(timeout_s: float = 20.0) -> dict[str, Any]:
    """Гасит процессы движка FreeToken, освобождая видеопамять.

    Нужна при смене движка: контроллер к этому моменту уже смотрит на новый
    движок, и прежний процесс иначе остаётся держать карту — FreeToken и
    llama.cpp делят один порт, но это разные процессы.

    @param timeout_s: сколько ждать штатного завершения.
    @returns: отчёт с числом остановленных процессов и временем.
    """
    pids = engine_pids()
    if not pids:
        return {"stopped": 0, "seconds": 0.0}
    seconds = _terminate_tree(pids, timeout_s)
    # Родитель погашен, но рабочие могли не успеть за ним. Их находят не по
    # имени, а по объявленному родителю: без метлы они держат память.
    _KILLED_ENGINE_PIDS.update(pids)
    swept = _sweep_workers(set(pids))
    return {"stopped": len(pids), "swept": len(swept), "seconds": round(seconds, 2)}


def _terminate_tree(pids: list[int], timeout_s: float = 20.0) -> float:
    """Гасит перечисленные процессы, при необходимости силой.

    @param pids: PID-ы процессов.
    @param timeout_s: сколько ждать штатного завершения.
    @returns: длительность операции в секундах.
    """
    started = time.time()
    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.Error:
            continue
    for proc in procs:
        try:
            proc.terminate()
        except psutil.Error:
            continue
    deadline = time.time() + timeout_s
    while time.time() < deadline and any(proc.is_running() for proc in procs):
        time.sleep(0.2)
    for proc in procs:
        try:
            if proc.is_running():
                proc.kill()
        except psutil.Error:
            continue
    return time.time() - started


def _comfy_pids() -> list[int]:
    """PID-ы сервера ComfyUI, включая процесс с занятым портом 8188."""
    pids: set[int] = set()
    port_pid = metrics.pid_on_port(8188)
    if port_pid:
        pids.add(port_pid)
        try:
            pids.update(child.pid for child in psutil.Process(port_pid).children(recursive=True))
        except psutil.Error:
            pass
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
        except psutil.Error:
            continue
        if "comfyui\\main.py" in cmdline or "comfyui/main.py" in cmdline:
            pids.add(proc.info["pid"])
    return sorted(pids)


def kill_everything(stop_comfy: bool = True, timeout_s: float = 20.0) -> KillReport:
    """Аварийно гасит движок и, если попросят, сервер ComfyUI.

    Приложение Comfy Desktop не закрывается: сервер поднимается кнопкой Start, и
    терять это состояние пользователю не за что.

    @param stop_comfy: останавливать ли сервер ComfyUI.
    @param timeout_s: сколько ждать штатного завершения каждого дерева.
    @returns: отчёт о том, что освободилось.
    """
    try:
        before = metrics.gpu_stats()["free_mb"]
    except (RuntimeError, OSError):
        before = 0

    engine = engine_pids()
    engine_seconds = _terminate_tree(engine, timeout_s) if engine else 0.0
    # Те же рабочие, что и в обычной остановке: без метлы они остаются держать
    # память, а найти их штатным поиском нельзя.
    if engine:
        _KILLED_ENGINE_PIDS.update(engine)
        _sweep_workers(set(engine))

    comfy: list[int] = []
    comfy_seconds = 0.0
    if stop_comfy:
        comfy = _comfy_pids()
        if comfy:
            comfy_seconds = _terminate_tree(comfy, timeout_s)

    # Драйвер отдаёт память не мгновенно.
    deadline = time.time() + 20.0
    after = before
    while time.time() < deadline:
        try:
            after = metrics.gpu_stats()["free_mb"]
        except (RuntimeError, OSError):
            break
        if after > before + 500:
            break
        time.sleep(0.5)

    return KillReport(
        engine_pids=engine,
        comfy_pids=comfy,
        engine_seconds=engine_seconds,
        comfy_seconds=comfy_seconds,
        gpu_free_before_mb=before,
        gpu_free_after_mb=after,
    )
