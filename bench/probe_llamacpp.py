r"""Проба llama.cpp: какой ручкой отключаются размышления модели.

Модель Gemma-4-E4B отвечает блоком ``<|channel>thought``. У llama.cpp
для этого есть два места: ``--reasoning-format`` (куда девать размышления) и
``--chat-template-kwargs`` (просьба к шаблону чата не думать вовсе).

Скрипт поднимает сервер по очереди с разными наборами ключей и печатает, что
оказалось в ``content`` и что в ``reasoning_content``. Картинки не рисуются,
модель не меняется — только пробные запросы.

Запуск::

    python bench\probe_llamacpp.py
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

#: Проверенная сборка: у неё нет AVX-512, которого лишён Arrow Lake.
#: Путь задаётся переменной окружения: он зависит от машины.
SERVER = Path(os.environ.get(
    "NOVELFORGE_LLAMACPP_SERVER",
    Path.home() / ".lmstudio" / "extensions" / "backends" / "llama-server.exe",
))
MODEL = Path(os.environ.get("NOVELFORGE_GGUF_MODEL",
                            Path.home() / "models" / "model.gguf"))
PORT = 1922
URL = f"http://127.0.0.1:{PORT}"

#: Наборы ключей: что пробуем и что ожидаем увидеть. Порядок важен: сначала
#: «как есть», потом попытки отключить — от самой простой к самой настойчивой.
VARIANTS: list[tuple[str, list[str]]] = [
    ("по умолчанию", []),
    ("--reasoning-format auto", ["--reasoning-format", "auto"]),
    ("шаблон: enable_thinking=false",
     ["--chat-template-kwargs", '{"enable_thinking": false}']),
]


def _all_variants() -> list[tuple[str, list[str]]]:
    """Полный набор для подробного разбора.

    @returns: все проверяемые наборы ключей.
    """
    return [
        ("--reasoning-format none", ["--reasoning-format", "none"]),
        ("--reasoning-format deepseek", ["--reasoning-format", "deepseek"]),
        ("оба: auto + enable_thinking=false",
         ["--reasoning-format", "auto",
          "--chat-template-kwargs", '{"enable_thinking": false}']),
    ]


def wait_ready(process: subprocess.Popen, timeout_s: int = 180) -> bool:
    """Ждёт, когда сервер ответит на /health.

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


def ask() -> dict:
    """Задаёт один короткий вопрос и возвращает разобранный ответ.

    @returns: ответ сервера целиком.
    """
    body = json.dumps({
        "model": "probe",
        "messages": [{"role": "user", "content": "Кто ты? Ответь одним предложением."}],
        "max_tokens": 200,
        "temperature": 0.7,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def run_variant(title: str, extra: list[str]) -> None:
    """Поднимает сервер с набором ключей и печатает, где оказался текст.

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
        answer = ask()
        message = (answer.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        print(f"  content:            {len(content):5} символов | {content[:90]!r}")
        print(f"  reasoning_content:  {len(reasoning):5} символов | {reasoning[:60]!r}")
        dirty = "thought" in content or "Thinking" in content or "<|channel" in content
        print(f"  ВЫВОД: {'размышления ЛЕЗУТ в ответ' if dirty else 'ответ ЧИСТЫЙ'}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(2)


def main() -> int:
    setup_console()
    print("Проверяю, чем отключаются размышления Gemma-4-E4B")
    print(f"сервер: {SERVER}")
    variants = VARIANTS + (_all_variants() if "--all" in sys.argv else [])
    for title, extra in variants:
        run_variant(title, extra)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
