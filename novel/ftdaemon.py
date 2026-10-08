"""Управление движком FreeToken через его службу-надзирателя.

FreeToken поднимает рабочие процессы, в командной строке которых нет слова
FreeToken: родитель умирает, а они остаются сиротами и держат видеопамять.
Родной поиск по имени их не находит, поэтому остановка движка, запущенного
напрямую через ``ft serve``, оставляет память занятой.

Служба решает это иначе. Она держит управление на своём порту и владеет деревом
процессов движка, поэтому остановка выходит чистой: рабочие не остаются
сиротами, и память возвращается.

Команд выгрузки отдельной модели у FreeToken нет: ``ft ctl`` умеет только
``health``, ``stats``, ``generate``, ``cache``, ``requests``. Поэтому смена
модели — это остановка движка и запуск с другой моделью.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from novel import config

#: Порт управления службы. Отдельный от порта движка.
DAEMON_PORT = 1900

#: Сколько ждать ответа службы на обычный запрос, секунды.
SHORT_TIMEOUT_S = 20.0

#: Сколько ждать остановки или смены модели, секунды.
LONG_TIMEOUT_S = 120.0


class DaemonError(RuntimeError):
    """Служба не ответила или отказала."""


def control_url(port: int = DAEMON_PORT) -> str:
    """Адрес управления службы.

    @param port: порт управления.
    @returns: адрес вида ``http://127.0.0.1:1900``.
    """
    return f"http://127.0.0.1:{port}"


def _call(verb: str, url: str, timeout_s: float) -> dict[str, Any]:
    """Зовёт службу через ``ft daemon``.

    @param verb: глагол службы, например ``status``.
    @param url: адрес управления.
    @param timeout_s: сколько ждать ответа.
    @raises DaemonError: если служба не ответила или вернула не разбор.
    @returns: разобранный ответ службы.
    """
    command = [str(config.FREETOKEN_FT_EXE), "daemon", verb, "--url", url]
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DaemonError(f"служба FreeToken не ответила: {exc}") from exc
    text = (done.stdout or "").strip()
    if not text:
        detail = (done.stderr or "").strip()[:200]
        raise DaemonError(f"служба FreeToken молчит: {detail}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DaemonError(f"ответ службы не разобран: {text[:200]}") from exc


def is_up(url: str = "") -> bool:
    """Отвечает ли служба.

    @param url: адрес управления; пустой означает адрес по умолчанию.
    @returns: признак того, что служба жива.
    """
    try:
        answer = _call("health", url or control_url(), SHORT_TIMEOUT_S)
    except DaemonError:
        return False
    return str(answer.get("daemon", "")).lower() == "up"


def spawn(url: str = "") -> None:
    """Поднимает службу, если она ещё не отвечает.

    Служба обязана пережить приложение: она владеет деревом процессов движка и
    потому останавливает его чисто. Без отсоединения служба уходит вместе с
    приложением, и движок остаётся без надзирателя — то есть ровно сиротой, ради
    чего служба и заведена.

    @param url: адрес управления; пустой означает адрес по умолчанию.
    """
    if is_up(url):
        return
    command = [str(config.FREETOKEN_FT_EXE), "daemon", "--port", str(DAEMON_PORT),
               "--setsid"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # Отсоединение от родителя: служба не должна уйти вместе с приложением.
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except OSError as exc:
        raise DaemonError(f"службу FreeToken поднять не удалось: {exc}") from exc


def wait_up(timeout_s: float = 60.0, url: str = "") -> bool:
    """Ждёт, пока служба начнёт отвечать.

    @param timeout_s: сколько ждать.
    @param url: адрес управления.
    @returns: признак готовности службы.
    """
    started = time.time()
    while time.time() - started < timeout_s:
        if is_up(url):
            return True
        time.sleep(1.5)
    return False


def engine(url: str = "") -> dict[str, Any]:
    """Что служба сообщает о движке.

    Поля ответа: ``running``, ``starting``, ``stopping``, ``pid``, ``port``,
    ``model``, ``uptimeS``, ``lastExitReason``.

    @param url: адрес управления.
    @raises DaemonError: если служба не ответила.
    @returns: состояние движка.
    """
    return _call("status", url or control_url(), SHORT_TIMEOUT_S)


def engine_running(url: str = "") -> bool:
    """Работает ли движок под службой.

    @param url: адрес управления.
    @returns: признак работающего движка.
    """
    try:
        return bool(engine(url).get("running"))
    except DaemonError:
        return False


def start(model_path: Path | str, *, port: int, serve_args: list[str] | None = None,
          url: str = "") -> dict[str, Any]:
    """Запускает движок под службой.

    Движок, уже обслуживающий ту же модель на том же порту, не перезапускается:
    служба отвечает признаком ``idempotent``.

    @param model_path: путь к модели.
    @param port: порт движка.
    @param serve_args: дополнительные параметры ``ft serve``.
    @param url: адрес управления.
    @raises DaemonError: если запустить не удалось.
    @returns: ответ службы.
    """
    command = [str(config.FREETOKEN_FT_EXE), "daemon", "start", str(model_path),
               "--url", url or control_url(), "--port", str(port)]
    if serve_args:
        command += ["--", *serve_args]
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=LONG_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DaemonError(f"движок не запущен: {exc}") from exc
    text = (done.stdout or "").strip()
    if not text:
        detail = (done.stderr or "").strip()[:200]
        raise DaemonError(f"служба не запустила движок: {detail}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def stop(url: str = "") -> dict[str, Any]:
    """Останавливает движок под службой.

    Остановка чистая: служба владеет деревом процессов, поэтому рабочие не
    остаются сиротами и память возвращается.

    @param url: адрес управления.
    @raises DaemonError: если остановить не удалось.
    @returns: ответ службы.
    """
    return _call("stop", url or control_url(), LONG_TIMEOUT_S)


def serve_args(memory_ratio: float, moe_strategy: str, host: str = "127.0.0.1",
               extra: list[str] | None = None) -> list[str]:
    """Собирает дополнительные параметры движка.

    Повторяет набор из прямого запуска, чтобы смена способа старта не меняла
    поведение модели.

    @param memory_ratio: доля памяти под движок.
    @param moe_strategy: способ размещения экспертов.
    @param host: адрес прослушивания.
    @param extra: дополнительные параметры.
    @returns: список параметров для ``ft serve``.
    """
    args = [
        "--host", host,
        "--moe-strategy", moe_strategy,
        "--max-running-requests", "4",
        "--memory-ratio", str(memory_ratio),
    ]
    return args + list(extra or [])


def port_answers(port: int, timeout_s: float = 4.0) -> bool:
    """Отвечает ли движок на своём порту.

    Порт открывается раньше, чем модель загружена, поэтому одного этого признака
    для готовности мало: запрос в этот момент получает отказ.

    @param port: порт движка.
    @param timeout_s: сколько ждать ответа.
    @returns: признак ответа.
    """
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                               timeout=timeout_s).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def probe(port: int, timeout_s: float = 180.0) -> bool:
    """Проверяет готовность движка коротким настоящим запросом.

    Занятой памяти мало: память занимается раньше, чем движок начинает принимать
    запросы — порог переваливается **во время заливки весов**, и ход в этот
    промежуток получает отказ.

    Запрос короткий: один токен. Он же слегка прогревает модель, поэтому время
    до первого слова относится уже к работе, а не к загрузке.

    @param port: порт движка.
    @param timeout_s: сколько ждать ответа.
    @returns: признак того, что движок отвечает.
    """
    body = {
        "model": "probe",
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as answer:
            json.loads(answer.read().decode("utf-8", "replace"))
        return True
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def wait_ready(port: int, timeout_s: float = 300.0,
               occupied_mb: int = 9000) -> float:
    """Ждёт, пока движок будет готов отвечать.

    Готовность подтверждается тремя признаками: порт отвечает, память занята
    весами и **пробный запрос проходит**. Одной памяти мало — она занимается
    раньше, чем движок начинает принимать запросы.

    @param port: порт движка.
    @param timeout_s: сколько ждать.
    @param occupied_mb: сколько памяти означает загруженную модель.
    @returns: время ожидания; минус единица, если движок не ответил.
    """
    started = time.time()
    weighted = False
    while time.time() - started < timeout_s:
        if port_answers(port):
            try:
                from novel import metrics

                if int(metrics.gpu_stats().get("used_mb") or 0) >= occupied_mb:
                    weighted = True
            except (RuntimeError, OSError, ImportError):
                # Без сведений о памяти остаётся проверка запросом.
                weighted = True
            if weighted and probe(port):
                return time.time() - started
        time.sleep(2)
    return -1.0
