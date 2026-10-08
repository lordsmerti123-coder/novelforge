"""Типы повествования: история, чат и квест.

Тип задаёт только то, как ведущий пишет текст для игрока. Как рисуются кадры —
отдельный набор настроек: что в кадре, когда рисовать, какого размера. Раньше
это лежало внутри типа, и оттого половина типов оказалась одним и тем же с
разными настройками картинок.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueFormat:
    """Один тип повествования."""

    key: str
    title: str
    summary: str
    user_label: str
    #: Инструкция о том, как писать текст для игрока.
    prose_style: str
    #: Ожидается ли блок ``<scene>``. Даже если да, рисовать его велит политика.
    wants_scene: bool
    #: Роль картинки: иллюстрация к сцене или «фотография» от собеседника.
    scene_role: str
    #: Как рисовать текст: сплошной прозой или лентой сообщений.
    ui: str
    #: Нужно ли требовать блок предгенерации.
    wants_speculative: bool = False

    def as_dict(self) -> dict[str, object]:
        """Представление для интерфейса."""
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "user_label": self.user_label,
            "prose_is_messages": self.ui == "chat",
            "wants_scene": self.wants_scene,
            "scene_role": self.scene_role,
            "ui": self.ui,
        }


FORMATS: dict[str, DialogueFormat] = {
    "story": DialogueFormat(
        key="story",
        title="История",
        summary="Проза от второго лица: обстановка, персонажи, реплики.",
        user_label="Что делает игрок?",
        prose_style=(
            "Пиши живую прозу от второго лица, 2-4 абзаца. Описывай обстановку, "
            "телесные ощущения, реплики персонажей. Каждый ход должен сдвигать историю: "
            "появляется новая информация, персонаж действует, ситуация меняется. "
            "Не заканчивай ход вопросом «что ты будешь делать?» — ставь игрока перед фактом."
        ),
        wants_scene=True,
        scene_role="illustration",
        ui="prose",
        wants_speculative=True,
    ),
    "chat": DialogueFormat(
        key="chat",
        title="Чат",
        summary="Переписка с одним собеседником. Если кадры включены, он присылает снимки с телефона.",
        user_label="Твоё сообщение",
        prose_style=(
            "Ты ведёшь переписку с игроком от лица одного собеседника. Каждое "
            "сообщение — отдельная строка, не длиннее двух предложений, и начинается "
            "именем говорящего и двоеточием. Никаких описаний обстановки, действий и "
            "ощущений: только текст сообщений. Одно-три сообщения за ход.\n"
            "Собеседник пишет с телефона, поэтому он в курсе, где находится и что "
            "делает: если занят, так и говорит и отвечает коротко. Он не описывает "
            "себя со стороны — он пишет о себе словами."
        ),
        wants_scene=True,
        scene_role="photo",
        ui="chat",
    ),
    "quest": DialogueFormat(
        key="quest",
        title="Квест",
        summary="Проза по написанному сюжету: места, предметы и этапы заданы заранее.",
        user_label="Что делает игрок?",
        prose_style=(
            "Пиши живую прозу от второго лица, 2-4 абзаца. Описывай обстановку, "
            "телесные ощущения, реплики персонажей. Каждый ход двигай сюжет к цели "
            "этапа: не топчись на месте и не повторяй уже случившееся. "
            "Не заканчивай ход вопросом «что ты будешь делать?» — ставь игрока перед фактом."
        ),
        wants_scene=True,
        scene_role="illustration",
        ui="prose",
    ),
}

DEFAULT_FORMAT = "story"

#: Ключи типов, которые были раньше и различались только настройками картинок.
#: Ведутся на нынешние, чтобы миры и партии не остались без типа.
LEGACY_FORMATS: dict[str, str] = {
    "story_terse": "story",
    "chat_photo": "chat",
    "chat_scene": "chat",
}


def get_format(key: str) -> DialogueFormat:
    """Возвращает тип повествования по ключу, подставляя запасной.

    @param key: ключ типа; понимаются и прежние ключи.
    @returns: описание типа.
    """
    return FORMATS.get(normalize_format(key), FORMATS[DEFAULT_FORMAT])


def normalize_format(key: str) -> str:
    """Приводит ключ типа к нынешнему.

    @param key: ключ типа, в том числе прежний.
    @returns: нынешний ключ; пустая строка, если ключ незнаком.
    """
    cleaned = str(key or "").strip()
    if cleaned in FORMATS:
        return cleaned
    return LEGACY_FORMATS.get(cleaned, "")


def resolve_format(session_format: str, world_format: str) -> DialogueFormat:
    """Тип повествования для партии.

    Тип принадлежит партии, а не миру: один и тот же мир проходят и прозой, и
    перепиской. Пустая строка у партии означает «как у мира» — так ведут себя
    партии, заведённые до того, как тип переехал в партию. Незнакомый ключ тоже
    уступает миру: он ближе к делу, чем общий запасной.

    @param session_format: тип у партии.
    @param world_format: тип у мира.
    @returns: описание типа.
    """
    chosen = normalize_format(session_format)
    if chosen:
        return FORMATS[chosen]
    return get_format(world_format)


def effective_images(world: object, settings: object) -> tuple[str, str]:
    """Когда рисовать кадры и что в них показывать.

    Значения задаются у мира, а если у мира пусто — берутся из общих настроек.
    Пустое значение у мира означает «как в настройках»: так ведут себя миры,
    заведённые до того, как настройки картинок переехали в мир.

    @param world: мир; допускается ``None``.
    @param settings: настройки приложения.
    @returns: пара «когда рисовать» и «что в кадре».
    """
    policy = str(getattr(world, "image_policy", "") or "").strip()
    frame = str(getattr(world, "image_frame", "") or "").strip()
    return (
        policy or str(getattr(settings, "image_policy", "minimal") or "minimal"),
        frame or str(getattr(settings, "chat_frame", "portrait") or "portrait"),
    )


def list_formats() -> list[dict[str, object]]:
    """Все типы повествования для выпадающего списка в интерфейсе."""
    return [fmt.as_dict() for fmt in FORMATS.values()]
