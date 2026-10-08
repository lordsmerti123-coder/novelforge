"""Картинки на вход языковой модели.

Движок принимает изображения как части сообщения в формате OpenAI:

    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

Base64 работает без дополнительных флагов запуска. Ссылки ``file://`` требуют
``--allowed-local-media-path``, а ``http(s)`` — интернета и разрешённого домена,
поэтому всё локальное передаётся встроенными данными.

Перед отправкой картинка уменьшается: у модели есть бюджет токенов на
изображение, и полноразмерный кадр 1024x1024 стоит заметно дороже, чем нужно
для описания сцены.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

from PIL import Image

#: Режимы применения зрения. Возможность модели видеть и необходимость
#: отправлять ей картинку — разные вещи: кадр 768x768 стоит около 258 токенов
#: входа и добавляет 1-4 секунды к ответу. Поэтому по умолчанию модель смотрит
#: только на то, что игрок приложил осознанно.
VISION_MODES: dict[str, str] = {
    "off": "выключено: вложения сохраняются в истории, но модели не отправляются",
    "on_attach": "только то, что приложил игрок",
    "on_attach_and_last_frame": "вложения игрока плюс последний сгенерированный кадр",
}

#: Режим по умолчанию: зрение применяется по необходимости, а не всегда.
DEFAULT_VISION_MODE = "on_attach"

#: Длинная сторона, до которой уменьшается изображение перед отправкой.
MAX_SIDE = 768

#: Формат и качество перекодирования.
JPEG_QUALITY = 88


class VisionError(RuntimeError):
    """Изображение не удалось подготовить."""


def prepare_image(path: Path, max_side: int = MAX_SIDE) -> tuple[str, int, int, tuple[int, int]]:
    """Готовит файл к отправке: уменьшает и кодирует в data URL.

    Прозрачность сохраняется: PNG с альфа-каналом остаётся PNG, остальное
    перекодируется в JPEG, потому что он заметно компактнее.

    @param path: путь к изображению на диске.
    @param max_side: предел длинной стороны в пикселях.
    @returns: ``(data URL, ширина, высота, исходный размер)``.
    @raises VisionError: если файл не читается как изображение.
    """
    try:
        with Image.open(path) as image:
            image.load()
            original = image.size
            has_alpha = image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            )
            if max(image.size) > max_side:
                ratio = max_side / max(image.size)
                image = image.resize(
                    (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                    Image.LANCZOS,
                )
            buffer = io.BytesIO()
            if has_alpha:
                image.convert("RGBA").save(buffer, format="PNG", optimize=True)
                mime = "image/png"
            else:
                image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
                mime = "image/jpeg"
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise VisionError(f"не удалось прочитать изображение {path}: {exc}") from exc

    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}", width, height, original


def image_part(data_url: str) -> dict[str, object]:
    """Часть сообщения с изображением.

    @param data_url: ссылка ``data:`` с встроенными данными.
    @returns: часть содержимого в формате OpenAI.
    """
    return {"type": "image_url", "image_url": {"url": data_url}}


def user_message(text: str, images: list[str] | None = None) -> dict[str, object]:
    """Сообщение пользователя, при необходимости с картинками.

    Текстовая часть без картинок остаётся обычной строкой: так запрос выглядит
    привычнее для модели и не тратит лишние токены на разметку.

    @param text: текст сообщения.
    @param images: ссылки ``data:`` с изображениями.
    @returns: сообщение в формате OpenAI.
    """
    if not images:
        return {"role": "user", "content": text}
    parts: list[dict[str, object]] = [{"type": "text", "text": text}]
    parts.extend(image_part(data_url) for data_url in images)
    return {"role": "user", "content": parts}


def save_upload(data_url: str, folder: Path, name: str) -> Path:
    """Сохраняет присланные интерфейсом данные в файл.

    @param data_url: ссылка ``data:`` с встроенными данными.
    @param folder: каталог для загрузок.
    @param name: желаемое имя файла без расширения.
    @returns: путь к сохранённому файлу.
    @raises VisionError: если данные не являются изображением в base64.
    """
    if not data_url.startswith("data:"):
        raise VisionError("ожидалась ссылка data: с изображением")
    header, _, payload = data_url.partition(",")
    if not payload:
        raise VisionError("в ссылке нет данных изображения")
    mime = header[5:].split(";", 1)[0] or "image/png"
    extension = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise VisionError(f"данные изображения повреждены: {exc}") from exc
    target = folder / f"{name}{extension}"
    target.write_bytes(raw)
    return target
