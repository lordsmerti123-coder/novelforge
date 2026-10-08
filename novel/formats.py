"""Форматы диалога.

Формат определяет три вещи сразу: что именно модель пишет в ответе, как этот
ответ рисуется в интерфейсе и какую роль играет картинка. Поэтому формат — не
косметическая настройка, а часть протокола.

Общее для всех форматов: игроку показывается только содержимое ``<prose>``, а
служебные блоки ``<scene>`` и ``<speculative>`` уходят оркестратору и обратно в
контекст не возвращаются.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Политики частоты генерации изображений.
IMAGE_POLICIES: dict[str, str] = {
    "never": "никогда",
    "every_turn": "каждый ход",
    "on_scene_change": "при смене сцены",
    "every_n": "каждые N ходов",
    "manual": "только по кнопке",
    "idle": "в простое, если есть время",
}


@dataclass(frozen=True)
class DialogueFormat:
    """Один формат диалога."""

    key: str
    title: str
    summary: str
    user_label: str
    #: Инструкция о том, как писать текст для игрока.
    prose_style: str
    #: Ожидается ли блок ``<scene>``.
    wants_scene: bool
    #: Роль картинки: иллюстрация к сцене, «фотография» в переписке или никакой.
    scene_role: str
    #: Как рисовать текст: сплошной прозой, репликами или лентой сообщений.
    ui: str
    default_image_policy: str
    default_image_size: int
    default_image_steps: int
    #: Нужно ли требовать блок предгенерации.
    wants_speculative: bool = False

    def as_dict(self) -> dict[str, object]:
        """Представление для интерфейса."""
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "user_label": self.user_label,
            "prose_is_messages": self.ui in ("chat", "photo_chat"),
            "wants_scene": self.wants_scene,
            "scene_role": self.scene_role,
            "ui": self.ui,
            "default_image_policy": self.default_image_policy,
            "default_image_size": self.default_image_size,
            "default_image_steps": self.default_image_steps,
        }


FORMATS: dict[str, DialogueFormat] = {
    "story": DialogueFormat(
        key="story",
        title="История с описаниями",
        summary="Проза от второго лица: обстановка, персонажи, реплики. Иллюстрация на смену сцены.",
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
        default_image_policy="on_scene_change",
        default_image_size=1024,
        default_image_steps=25,
        wants_speculative=True,
    ),
    "story_terse": DialogueFormat(
        key="story_terse",
        title="История без описаний",
        summary="Только действия и реплики, без пейзажных описаний. Дешевле по токенам и быстрее.",
        user_label="Что делает игрок?",
        prose_style=(
            "Пиши скупо: только действия, реплики и то, что игрок обязан знать. "
            "Никаких описаний обстановки, погоды и одежды. Один-два абзаца, короткие фразы. "
            "Каждый ход двигает историю вперёд."
        ),
        wants_scene=True,
        scene_role="illustration",
        ui="prose",
        default_image_policy="on_scene_change",
        default_image_size=1024,
        default_image_steps=20,
    ),
    "chat": DialogueFormat(
        key="chat",
        title="Просто чат",
        summary="Мессенджер: короткие сообщения собеседника, без описаний вообще.",
        user_label="Твоё сообщение",
        prose_style=(
            "Ты пишешь в мессенджере. Каждое сообщение — отдельная строка, не длиннее "
            "двух предложений. Если говорит другой персонаж, начни строку его именем и "
            "двоеточием. Никаких описаний действий и обстановки: только текст сообщений. "
            "Одно-три сообщения за ход."
        ),
        wants_scene=False,
        scene_role="none",
        ui="chat",
        default_image_policy="manual",
        default_image_size=768,
        default_image_steps=20,
    ),
    "chat_photo": DialogueFormat(
        key="chat_photo",
        title="Чат с фотками",
        summary="Чатрулетка: собеседник присылает фотографию и комментирует её.",
        user_label="Твоё сообщение",
        prose_style=(
            "Ты общаешься в чате и время от времени присылаешь фотографии. "
            "Каждое сообщение — отдельная строка, не длиннее двух предложений, и "
            "начинается именем говорящего и двоеточием. Никаких описаний "
            "обстановки, действий и ощущений: только текст сообщений. Короткий "
            "ответ — одна строка, обычный — две или три. "
            "Когда присылаешь фотографию, опиши её в блоке scene — она покажется "
            "как вложение к последнему сообщению. Фотография должна быть бытовой "
            "и естественной, будто снята на телефон."
        ),
        wants_scene=True,
        scene_role="photo",
        ui="photo_chat",
        default_image_policy="every_turn",
        default_image_size=768,
        default_image_steps=20,
    ),
    "chat_scene": DialogueFormat(
        key="chat_scene",
        title="Диалоги с картинками",
        summary="Переписка персонажей, где картинка — кадр сцены, а не вложение.",
        user_label="Реплика",
        prose_style=(
            "Пиши переписку двух-трёх персонажей. Каждая реплика — отдельная "
            "строка, не длиннее двух предложений, в формате «Имя: текст». "
            "Описаний нет, кроме коротких авторских ремарок в скобках. "
            "Картинка иллюстрирует сцену, в которой идёт разговор."
        ),
        wants_scene=True,
        scene_role="illustration",
        ui="chat",
        default_image_policy="on_scene_change",
        default_image_size=1024,
        default_image_steps=25,
    ),
}

DEFAULT_FORMAT = "story"


def get_format(key: str) -> DialogueFormat:
    """Возвращает формат по ключу, подставляя формат по умолчанию.

    @param key: ключ формата.
    @returns: описание формата.
    """
    return FORMATS.get(key, FORMATS[DEFAULT_FORMAT])


def list_formats() -> list[dict[str, object]]:
    """Все форматы для выпадающего списка в интерфейсе."""
    return [fmt.as_dict() for fmt in FORMATS.values()]
