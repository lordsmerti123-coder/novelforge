"""Резервная копия исходников NovelForge.

Копия хранит **только код**: `novel/`, `gui/`, `bench/`, `docs/` и корневые
файлы. Данных пользователя в ней нет и быть не должно: база содержит тексты
миров и партий, а удалённые строки остаются в свободных страницах файла и
находятся поиском по байтам. Миры выгружаются из интерфейса отдельно.

Смысл копии — восстановить проект, если правка что-то сломала. Поэтому она
кладётся в `backups/` рядом с проектом, а не в `data/`: ту папку очищают при
сбросе, и копии исчезли бы вместе с ней.

Запуск::

    python bench\\backup.py
    python bench\\backup.py --list
    python bench\\backup.py --fresh --keep 5
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "backups"

#: Что попадает в копию.
SOURCE_DIRS = ("novel", "gui", "bench", "docs")
SOURCE_FILES = ("run_gui.py", "launcher.py", "README.md")

#: Что не попадает: собирается заново или принадлежит пользователю.
SKIP_PARTS = {"__pycache__", "logs", "backups", "data", ".mypy_cache", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".db", ".db-wal", ".db-shm", ".zip", ".png", ".log"}


def human(size: float) -> str:
    """Размер в удобных единицах."""
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


def collect() -> list[Path]:
    """Файлы проекта, которые стоит сохранить."""
    files: list[Path] = []
    for name in SOURCE_DIRS:
        root = ROOT / name
        if not root.is_dir():
            continue
        files += [
            path for path in root.rglob("*")
            if path.is_file()
            and not any(part in SKIP_PARTS for part in path.parts)
            and path.suffix not in SKIP_SUFFIXES
        ]
    files += [ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file()]
    return sorted(files)


def existing() -> list[Path]:
    """Копии, лежащие в папке, свежие сверху."""
    return sorted(BACKUP_DIR.glob("novelforge-*.zip"))


def show(archives: list[Path]) -> None:
    """Печатает список копий."""
    if not archives:
        print("  копий нет")
        return
    for item in archives:
        stamp = time.strftime("%d.%m %H:%M", time.localtime(item.stat().st_mtime))
        print(f"    {item.name} — {human(item.stat().st_size)}, {stamp}")


def verify(archive: Path) -> list[str]:
    """Проверяет, что в копии только исходники.

    @param archive: готовый архив.
    @returns: замечания; пустой список означает, что всё в порядке.
    """
    problems: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if not names:
            problems.append("архив пуст")
        for name in names:
            suffix = Path(name).suffix
            if suffix in (".db", ".db-wal", ".db-shm"):
                problems.append(f"в копии база данных: {name}")
            if name.startswith("data/"):
                problems.append(f"в копии данные пользователя: {name}")
    if not any(name.startswith("novel/") for name in names):
        problems.append("в копии нет исходников novel/")
    return problems


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Резервная копия исходников NovelForge")
    parser.add_argument("--keep", type=int, default=10,
                        help="сколько последних копий оставлять (0 — все)")
    parser.add_argument("--fresh", action="store_true",
                        help="удалить прежние копии перед созданием новой")
    parser.add_argument("--list", action="store_true", help="только показать копии")
    args = parser.parse_args()

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("\n=== Копии проекта ===\n")
        show(existing())
        print()
        return 0

    files = collect()
    print("\n=== Резервная копия проекта ===\n")
    if not files:
        print("  нечего копировать: исходники не найдены")
        return 1

    if args.fresh:
        for old in existing():
            old.unlink()
            print(f"  удалена прежняя: {old.name}")

    archive = BACKUP_DIR / f"novelforge-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, str(path.relative_to(ROOT)))

    problems = verify(archive)
    print(f"  файлов: {len(files)}")
    print(f"  копия: {archive.name}, {human(archive.stat().st_size)}")
    if problems:
        print("  ЗАМЕЧАНИЯ:")
        for item in problems:
            print(f"    {item}")

    removed = []
    archives = existing()
    for old in archives[:-args.keep] if args.keep > 0 else []:
        removed.append(old.name)
        old.unlink()
    if removed:
        print(f"  удалены старые: {', '.join(removed)}")

    print(f"\n  всего копий: {len(existing())}")
    show(existing()[-3:])
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
