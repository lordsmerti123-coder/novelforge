"""Перенос мира вместе с картинками.

Обычная выгрузка мира — это JSON: в нём есть пути к файлам, но не сами файлы.
На другой машине или после переноса каталога такие пути повиснут, и мир останется
без иллюстраций.

Пакет решает это: это обычный ZIP, внутри `world.json` и папка `images` с кадрами
и вложениями. Пути в JSON заменяются на архивные, а при загрузке — на новые,
уже распакованные. Файлы кладутся в `data/bundled/<метка>/`, поэтому загруженный
мир самодостаточен и не зависит от того, где лежал исходный.

Имена файлов в архиве получают короткий хеш исходного пути: два кадра из разных
партий могут называться одинаково, и без этого один затёр бы другой.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel import config
from novel.db import NovelDB

MANIFEST_NAME = "world.json"
IMAGES_PREFIX = "images/"


@dataclass
class BundleReport:
    """Что получилось при выгрузке или загрузке пакета."""

    path: str
    images: int
    bytes: int
    world_id: int | None = None
    session_ids: list[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса."""
        return {
            "path": self.path,
            "images": self.images,
            "size_mb": round(self.bytes / (1024 * 1024), 2),
            "world_id": self.world_id,
            "session_ids": self.session_ids or [],
        }


def _archive_name(source: Path) -> str:
    """Имя файла внутри архива.

    @param source: исходный путь к изображению.
    @returns: имя вида ``images/<хеш>-<имя>``.
    """
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{IMAGES_PREFIX}{digest}-{source.name}"


def export_bundle(db: NovelDB, world_id: int, target: Path) -> BundleReport:
    """Выгружает мир в ZIP вместе с изображениями.

    @param db: хранилище.
    @param world_id: мир.
    @param target: путь к создаваемому архиву.
    @returns: отчёт о выгрузке.
    @raises ValueError: если мир не найден.
    """
    payload = db.export_world(world_id)
    collected: dict[str, Path] = {}

    def register(raw: Any) -> Any:
        """Заменяет путь на архивный, попутно собирая файлы."""
        if not raw:
            return raw
        source = Path(str(raw))
        if not source.is_file():
            return raw
        name = _archive_name(source)
        collected[name] = source
        return name

    for session in payload.get("sessions") or []:
        for scene in session.get("scenes") or []:
            scene["path"] = register(scene.get("path"))
        for message in session.get("messages") or []:
            message["attachments"] = [
                register(item) for item in (message.get("attachments") or [])
            ]

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(payload, ensure_ascii=False, indent=2))
        for name, source in collected.items():
            archive.write(source, name)

    return BundleReport(path=str(target), images=len(collected), bytes=target.stat().st_size)


def import_bundle(db: NovelDB, source: Path, folder: Path | None = None) -> BundleReport:
    """Загружает мир из ZIP вместе с изображениями.

    Файлы распаковываются в отдельный каталог, поэтому загруженный мир не зависит
    от расположения исходного.

    @param db: хранилище.
    @param source: путь к архиву.
    @param folder: куда распаковывать; по умолчанию ``data/bundled/<метка>``.
    @returns: отчёт о загрузке.
    @raises ValueError: если архив не является пакетом NovelForge.
    """
    if not source.is_file():
        raise ValueError(f"архив {source} не найден")

    with zipfile.ZipFile(source) as archive:
        if MANIFEST_NAME not in archive.namelist():
            raise ValueError("в архиве нет world.json — это не пакет NovelForge")
        try:
            payload = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"world.json повреждён: {exc}") from exc

        destination = folder or config.DATA_DIR / "bundled" / time.strftime("%Y%m%d-%H%M%S")
        extracted: dict[str, str] = {}
        for name in archive.namelist():
            if not name.startswith(IMAGES_PREFIX) or name.endswith("/"):
                continue
            # Имя берётся без каталогов: запись вида ../../ что-то распаковала бы
            # за пределами каталога проекта.
            target = destination / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            extracted[name] = str(target)

    def restore(raw: Any) -> Any:
        """Возвращает распакованный путь вместо архивного."""
        if not raw:
            return raw
        return extracted.get(str(raw), raw)

    for session in payload.get("sessions") or []:
        for scene in session.get("scenes") or []:
            scene["path"] = restore(scene.get("path"))
        for message in session.get("messages") or []:
            message["attachments"] = [
                restore(item) for item in (message.get("attachments") or [])
            ]

    world_id, session_ids = db.import_world(payload)
    return BundleReport(
        path=str(source),
        images=len(extracted),
        bytes=source.stat().st_size,
        world_id=world_id,
        session_ids=session_ids,
    )


def referenced_files(db: NovelDB, world_id: int | None = None) -> set[str]:
    """Файлы, на которые ссылаются записи базы.

    Нужно, чтобы отличить нужные изображения от осиротевших: удаление партии
    убирает записи, но не файлы.

    @param db: хранилище.
    @param world_id: ограничить одним миром; ``None`` — вся база.
    @returns: множество путей.
    """
    found: set[str] = set()
    sessions = [session.id for session in db.sessions(world_id)]
    for session_id in sessions:
        for scene in db.scenes(session_id):
            if scene.path:
                found.add(str(Path(scene.path).resolve()))
        for message in db.messages(session_id):
            for item in message.attachment_paths:
                found.add(str(Path(item).resolve()))
    for location in db.locations(world_id) if world_id else _all_locations(db):
        if location.reference_path:
            found.add(str(Path(location.reference_path).resolve()))
    return found


def _all_locations(db: NovelDB) -> list[Any]:
    """Все места всех миров — образцы тоже нельзя удалять."""
    rows = db.conn.execute("SELECT * FROM locations").fetchall()
    from novel.db import Location

    return [
        Location(
            id=row["id"], world_id=row["world_id"], name=row["name"], prompt=row["prompt"],
            style=row["style"], seed=row["seed"], reference_path=row["reference_path"],
            visits=row["visits"], created_at=row["created_at"], updated_at=row["updated_at"],
        )
        for row in rows
    ]


def session_weight(db: NovelDB, session_id: int) -> dict[str, Any]:
    """Сколько места занимает партия.

    Текст весит копейки: сотни сообщений — это доли мегабайта. Всё остальное —
    кадры, и именно их надо учитывать при переносе и удалении.

    @param db: хранилище.
    @param session_id: партия.
    @returns: счётчики сообщений, сцен и мегабайты файлов.
    """
    messages = db.messages(session_id)
    scenes = db.scenes(session_id)
    images = 0
    megabytes = 0.0
    for scene in scenes:
        if not scene.path:
            continue
        path = Path(scene.path)
        if path.is_file():
            images += 1
            megabytes += path.stat().st_size / (1024 * 1024)
    for message in messages:
        for item in message.attachment_paths:
            path = Path(item)
            if path.is_file():
                images += 1
                megabytes += path.stat().st_size / (1024 * 1024)
    text_bytes = sum(len(message.content.encode("utf-8")) for message in messages)
    return {
        "session_id": session_id,
        "messages": len(messages),
        "scenes": len(scenes),
        "images": images,
        "images_mb": round(megabytes, 2),
        "text_kb": round(text_bytes / 1024, 1),
    }


def orphan_files(
    db: NovelDB,
    *,
    uploads_dir: Path | None = None,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """Файлы NovelForge, на которые уже никто не ссылается.

    Удаление партии убирает записи из базы, но не файлы: они лежат вне неё. Здесь
    собирается всё, что осталось, по трём местам отдельно — чтобы было видно, что
    именно предлагается удалить.

    Каталог вывода генератора общий с ComfyUI, поэтому берутся только файлы с
    нашим префиксом ``novel_``; остальное не трогается никогда.

    @param db: хранилище.
    @param uploads_dir: каталог вложений; по умолчанию из настроек проекта.
    @param output_dir: каталог кадров; по умолчанию из настроек проекта.
    @param input_dir: каталог входов ComfyUI; по умолчанию из настроек проекта.
    @returns: словарь «категория — список путей».
    """
    used = referenced_files(db)
    result: dict[str, list[Path]] = {"uploads": [], "frames": [], "staged": []}

    uploads = uploads_dir or config.UPLOADS_DIR
    if uploads.is_dir():
        for path in uploads.iterdir():
            if path.is_file() and str(path.resolve()) not in used:
                result["uploads"].append(path)

    output = output_dir or config.comfy_output_dir()
    if output.is_dir():
        for path in output.glob("novel_*"):
            if path.is_file() and str(path.resolve()) not in used:
                result["frames"].append(path)

    # Копии образцов в каталоге входов: имя вида «хеш-исходное_имя».
    incoming = input_dir or config.comfy_input_dir()
    if incoming.is_dir():
        for path in incoming.iterdir():
            if not path.is_file() or str(path.resolve()) in used:
                continue
            head, dash, _ = path.name.partition("-")
            if dash and len(head) == 8 and all(char in "0123456789abcdef" for char in head):
                result["staged"].append(path)

    return {key: sorted(items) for key, items in result.items()}


def purge_orphans(
    db: NovelDB,
    *,
    uploads_dir: Path | None = None,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
) -> dict[str, Any]:
    """Удаляет осиротевшие файлы с диска.

    @param db: хранилище.
    @returns: сколько файлов удалено и сколько места освобождено.
    """
    removed = 0
    freed = 0
    by_kind: dict[str, int] = {}
    groups = orphan_files(
        db, uploads_dir=uploads_dir, output_dir=output_dir, input_dir=input_dir
    )
    for kind, paths in groups.items():
        for path in paths:
            try:
                freed += path.stat().st_size
                path.unlink()
                removed += 1
                by_kind[kind] = by_kind.get(kind, 0) + 1
            except OSError:
                continue
    return {
        "removed": removed,
        "freed_mb": round(freed / (1024 * 1024), 2),
        "by_kind": by_kind,
    }


def orphan_uploads(db: NovelDB) -> list[Path]:
    """Вложения, на которые уже никто не ссылается.

    Оставлено ради совместимости: полный список даёт :func:`orphan_files`.

    @param db: хранилище.
    @returns: список путей к осиротевшим вложениям.
    """
    return orphan_files(db)["uploads"]
