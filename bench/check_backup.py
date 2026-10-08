"""Проверяет резервную копию проекта: что внутри и можно ли из неё восстановить.

Копия бесполезна, если из неё не поднимается рабочий проект, поэтому проверка
не ограничивается списком файлов: архив распаковывается во временную папку и
все модули компилируются.

Запуск::

    python bench\\check_backup.py
    python bench\\check_backup.py --restore <каталог для восстановления>
"""

from __future__ import annotations

import argparse
import compileall
import py_compile
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "backups"

#: Что обязано быть в копии, чтобы проект поднялся.
REQUIRED = ("novel/machine.py", "novel/db.py", "gui/app.py",
            "gui/static/index.html", "run_gui.py")

#: Чего в копии быть не должно: это данные пользователя.
FORBIDDEN_SUFFIXES = (".db", ".db-wal", ".db-shm")
FORBIDDEN_PREFIXES = ("data/",)


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Проверка резервной копии проекта")
    parser.add_argument("--archive", help="какой архив смотреть; по умолчанию свежий")
    parser.add_argument("--restore", metavar="ПАПКА",
                        help="распаковать сюда, а не во временную папку")
    args = parser.parse_args()

    archives = sorted(BACKUP_DIR.glob("novelforge-*.zip"))
    if not archives:
        print("\nкопий нет\n")
        return 1
    archive = Path(args.archive) if args.archive else archives[-1]

    print(f"\n=== Проверка копии {archive.name} ===\n")
    problems: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        print(f"  файлов: {len(names)}, размер: {archive.stat().st_size // 1024} КБ")
        sections = sorted({name.split("/")[0] for name in names})
        print(f"  разделы: {', '.join(sections)}")

        for name in names:
            if name.endswith(FORBIDDEN_SUFFIXES) or name.startswith(FORBIDDEN_PREFIXES):
                problems.append(f"данные пользователя в копии: {name}")
        print(f"  данных пользователя: "
              f"{'НЕТ' if not problems else 'НАЙДЕНЫ'}")

        missing = [item for item in REQUIRED if item not in names]
        if missing:
            problems.append(f"не хватает файлов: {', '.join(missing)}")
        print(f"  обязательные файлы: {'на месте' if not missing else 'НЕ ВСЕ'}")

        target = Path(args.restore) if args.restore else Path(tempfile.mkdtemp())
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
        print(f"  распаковано в: {target}")

    # Компиляция: если модуль не собирается, проект из копии не поднимется.
    bad: list[str] = []
    for path in sorted(target.rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True, cfile=str(path) + "c")
        except py_compile.PyCompileError as exc:
            bad.append(f"{path.relative_to(target)}: {exc.msg.splitlines()[-1][:80]}")
    print(f"  компилируется: {'всё' if not bad else f'НЕ ВСЁ ({len(bad)})'}")
    for item in bad[:5]:
        print(f"    {item}")
    if bad:
        problems.append(f"не компилируется файлов: {len(bad)}")

    # Страница интерфейса: без неё поднимать нечего.
    page = target / "gui" / "static" / "index.html"
    if page.exists():
        text = page.read_text(encoding="utf-8", errors="replace")
        print(f"  страница интерфейса: {len(text)} символов, "
              f"вкладок {text.count('class=\"tabbody')}")
        if "id=\"messages\"" not in text:
            problems.append("в странице нет списка сообщений")
    else:
        problems.append("нет страницы интерфейса")

    print()
    if problems:
        print("  ЗАМЕЧАНИЯ:")
        for item in problems:
            print(f"    {item}")
        print()
        return 1
    print("  копия пригодна: из неё поднимается проект\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
