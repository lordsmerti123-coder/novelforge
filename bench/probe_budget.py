r"""Проба: как регулируется бюджет размышлений у DeepSeek-R1 на llama.cpp.

Отключить размышления у DeepSeek нельзя — в его шаблоне чата нет ни
``enable_thinking``, ни бюджета. Это видно из самого GGUF: шаблон оперирует
маркерами ``<｜User｜>`` и ``</think>`` и никаких переменных не принимает.

Зато у llama.cpp есть свои ключи:

* ``--reasoning-budget N`` — ``-1`` без ограничений, ``0`` оборвать сразу,
  ``N`` — столько токенов на размышления;
* ``--reasoning on|off|auto`` — использовать ли размышления вообще;
* ``--reasoning-budget-message`` — что подставить, когда бюджет исчерпан.

Скрипт поднимает сервер по очереди с разными ключами и печатает, сколько модель
думала и что успела ответить. Проверяет не FreeToken, а llama.cpp: у FreeToken
для этого есть свой ``bench/probe_reasoning.py``.

Запуск::

    python bench\probe_budget.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

#: Пути задаются переменными окружения: они зависят от машины.
SERVER = Path(os.environ.get(
    "NOVELFORGE_LLAMACPP_SERVER",
    Path.home() / ".lmstudio" / "extensions" / "backends" / "llama-server.exe",
))
MODEL = Path(os.environ.get("NOVELFORGE_GGUF_MODEL",
                            Path.home() / "models" / "model.gguf"))
PORT = 1923
URL = f"http://127.0.0.1:{PORT}"

#: Что пробуем. Первый вариант — как есть, дальше по возрастанию настойчивости.
VARIANTS: list[tuple[str, list[str]]] = [
    ("без ограничений", []),
    ("--reasoning off", ["--reasoning", "off"]),
    ("бюджет 0", ["--reasoning-budget", "0"]),
    ("бюджет 128", ["--reasoning-budget", "128"]),
    ("бюджет 512", ["--reasoning-budget", "512"]),
    ("бюджет 128 + просьба прекратить",
     ["--reasoning-budget", "128",
      "--reasoning-budget-message", "Хватит думать, отвечай."]),
]

QUESTION = "Опиши одним абзацем таверну вечером."


def wait_ready(process: subprocess.Popen, timeout_s: int = 240) -> bool:
    """Ждёт готовности сервера.

    @param process: запущенный сервер.
    @param timeout_s: сколько ждать.
    @returns: ``True``, если сервер поднялся.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=4) as response:
                if b"ok" in response.read():
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def run_variant(title: str, extra: list[str]) -> None:
    """Поднимает сервер с набором ключей и печатает замер.

    @param title: название варианта.
    @param extra: дополнительные ключи запуска.
    """
    args = [SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
            "-c", "4096", "-ngl", "99", "--parallel", "1", "--jinja",
            "-a", "probe", *extra]
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=creation)
    print(f"\n=== {title} ===")
    try:
        if not wait_ready(process):
            print("  сервер не поднялся")
            return
        body = json.dumps({
            "model": "probe",
            "messages": [{"role": "user", "content": QUESTION}],
            "max_tokens": 900, "temperature": 0.8,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{URL}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
        started = time.time()
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read().decode("utf-8"))
        elapsed = time.time() - started

        message = (answer.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        usage = answer.get("usage") or {}
        print(f"  время:       {elapsed:6.1f} c")
        print(f"  размышления: {len(reasoning):5} символов")
        print(f"  ответ:       {len(content):5} символов")
        print(f"  токенов:     {usage.get('completion_tokens')}")
        if content:
            print(f"  -> {content[:100]}")
        if reasoning:
            print(f"  думал: {reasoning[:80]!r}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(2)


def main() -> int:
    setup_console()
    print("Бюджет размышлений DeepSeek-R1 на llama.cpp")
    print(f"модель: {Path(MODEL).name}")
    for title, extra in VARIANTS:
        run_variant(title, extra)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
