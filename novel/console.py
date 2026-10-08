"""Настройка консольного вывода.

Без этого русский текст уезжает в кодировку текущей локали (cp1251), а читающая
сторона ждёт UTF-8 и получает мусор.
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    """Переключает стандартные потоки на UTF-8.

    Вызывается в начале каждого исполняемого скрипта проекта.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            continue
