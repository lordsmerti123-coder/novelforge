"""Агент-редактор: выполняет указания по миру, написанные словами.

Пользователь пишет в отладке, что сделать: «добавь правило про цену магии»,
«перепиши правила мрачнее», «придумай двух персонажей», «создай мир про
затонувший город». Агент разбирает указание на действия, выполняет их и
показывает, что именно сделал.

Как это работает. Модель не вызывает инструменты напрямую: она возвращает JSON
со списком действий, оркестратор их выполняет и отдаёт результаты обратно. Такой
цикл повторяется, пока модель не скажет, что закончила, или пока не кончится
лимит шагов. Выбор в пользу протокола, а не нативных вызовов функций, сделан
потому, что разбор JSON у нас уже отлажен и одинаково работает на любой модели, а
качество нативных вызовов у локальных моделей непредсказуемо.

Разрушительные действия (удаление миров, правил, персонажей, очистка партий)
выполняются только при явном разрешении. Режим предпросмотра показывает план и
ничего не меняет.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from novel.authoring import Author, AuthoringError
from novel.db import Character, NovelDB, Rule
from novel.freetoken import FreeTokenClient, FreeTokenError
from novel.protocol import loads_json
from novel.settings import IMAGE_QUALITY

#: Как модель называет поля мира и как они зовутся на самом деле. Описание
#: операции перечисляет `brief`, но модель всё равно пишет `description` — так
#: естественнее. Отказывать из-за слова значит терять уже сделанную работу:
#: шагов мало, агент тратит их на ошибки — добавляет лишнее, получает отказ
#: из-за несуществующего поля и не успевает закончить.
WORLD_FIELD_ALIASES: dict[str, str] = {
    "description": "brief",
    "описание": "brief",
    "название": "name",
    "имя": "name",
    "жанр": "genre",
    "тон": "tone",
    "стиль": "style",
    "рассказчик": "narrator",
    "нарратор": "narrator",
    "скрытые_правила": "hidden_rules",
    "скрытые правила": "hidden_rules",
    "дополнение": "image_suffix",
    "суффикс": "image_suffix",
}


def _world_field(name: str) -> str:
    """Приводит название поля мира к тому, что есть в базе.

    @param name: как поле назвала модель.
    @returns: имя поля из базы; неизвестное возвращается без изменений, чтобы
        отказ остался внятным.
    """
    cleaned = str(name or "").strip().casefold()
    return WORLD_FIELD_ALIASES.get(cleaned, cleaned)


#: Сколько шагов подряд без единого изменения терпим. Ловит не повтор (отпечатки
#: у таких шагов разные), а именно топтание: модель зовёт `rewrite_text` раз за
#: разом, каждый раз с новым текстом, и не меняет ничего.
IDLE_STEPS_LIMIT = 3

#: Сколько шагов «подумал — сделал» допускается за одно указание.
MAX_STEPS = 8


def _signature(action: dict[str, Any]) -> str:
    """Отпечаток действия: имя и аргументы без учёта порядка ключей.

    Нужен, чтобы заметить повтор. Модель без флага ``done`` ходит по кругу и
    выполняет одно и то же: в мир ложатся одинаковые правила, а прогон кончается
    «не уложился в шаги».

    @param action: разобранное действие из ответа модели.
    @returns: строка-отпечаток.
    """
    name = str(action.get("op") or "")
    args = {key: value for key, value in action.items() if key != "op"}
    return name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)

AGENT_SYSTEM_PROMPT = """Ты редактор миров для текстовой игры. Пользователь даёт указание \
словами, ты превращаешь его в список действий.

Отвечай ТОЛЬКО JSON, без пояснений и без markdown:

{{
  "thought": "одна фраза: что собираешься сделать",
  "actions": [
    {{"op": "имя_действия", "аргументы...": "значения"}}
  ],
  "done": true
}}

Поле `done` ставь в true, когда после этих действий работа закончена. Если нужно \
сначала посмотреть результат — поставь false, и тебе вернут, что получилось.

Доступные действия:

{tools}

Правила:
- Все аргументы, которые описаны как обязательные, должны быть заполнены.
- Пиши по-русски. По-английски заполняется только `appearance` — оно уходит в \
генератор изображений.
- Не выдумывай номера миров и правил: бери их из состояния ниже.
- Если указание непонятно или для него нет подходящего действия, верни `actions` \
пустым, а в `thought` напиши, чего не хватает.
- **Работай в том мире, который открыт сейчас.** Если пользователь не просит \
создать новый мир, не вызывай `draft_world`: он заводит отдельный мир, и \
пользователь его не увидит. Чтобы дополнить открытый мир, вызывай `fill_world` \
или добавляй правила и персонажей по отдельности.
- Если номер мира не указан в аргументах, действие применится к открытому миру. \
Так и делай по умолчанию.
- **Про вид картинок.** Реалистичность и общий вид кадров задаёт стиль мира \
(`set_world_style`) — это слова, которые уходят генератору вместе с описанием: \
`hyper-realistic, photorealistic`, `oil painting`, `detailed skin texture`, \
`natural lighting`. Число шагов и размер (`set_image_settings`) влияют на \
чёткость и время, но не на стиль. Если просят «реалистичнее» — правь стиль, а не \
шаги.

Открыт сейчас: {current}

Состояние сейчас:

{state}
"""


@dataclass
class AgentStep:
    """Один шаг агента: что подумал, что сделал, что получилось."""

    thought: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса."""
        return {"thought": self.thought, "actions": self.actions, "results": self.results}


@dataclass
class AgentRun:
    """Один прогон агента по указанию пользователя."""

    instruction: str
    steps: list[AgentStep] = field(default_factory=list)
    finished: bool = False
    preview: bool = False
    error: str | None = None
    seconds: float = 0.0
    #: Прогон прерван: движок выгрузили под работающим агентом.
    interrupted: bool = False
    #: Движок был поднят заново посреди работы.
    revived: bool = False
    #: Действия, которые выполняются после прогона: смена модели перезапускает
    #: движок, на котором работает сам агент.
    pending: list[dict[str, Any]] = field(default_factory=list)
    #: Прогон закончился потому, что модель пошла по кругу, а не потому что
    #: сказала «готово».
    looped: bool = False

    @property
    def changed(self) -> int:
        """Сколько действий действительно изменило данные."""
        return sum(
            1
            for step in self.steps
            for result in step.results
            if result.get("ok") and result.get("changed")
        )

    def applied(self) -> list[str]:
        """Список выполненных действий — что именно успело примениться."""
        return [
            str(result.get("op"))
            for step in self.steps
            for result in step.results
            if result.get("ok") and result.get("changed")
        ]

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса."""
        return {
            "instruction": self.instruction,
            "steps": [step.as_dict() for step in self.steps],
            "finished": self.finished,
            "preview": self.preview,
            "error": self.error,
            "seconds": round(self.seconds, 1),
            "changed": self.changed,
            "interrupted": self.interrupted,
            "revived": self.revived,
            "applied": self.applied(),
            "pending": self.pending,
        }


@dataclass
class Operation:
    """Одно действие, доступное агенту."""

    name: str
    summary: str
    args: dict[str, str]
    handler: Callable[..., dict[str, Any]]
    destructive: bool = False
    previewable: bool = True
    #: Действие, которое нельзя выполнять внутри прогона: смена модели
    #: перезапускает движок, на котором работает сам агент. Такие действия
    #: записываются и применяются после завершения.
    deferred: bool = False

    def signature(self) -> str:
        """Строка описания действия для промпта."""
        if self.args:
            arguments = ", ".join(f'"{key}": {value}' for key, value in self.args.items())
        else:
            arguments = ""
        marker = " (выполнится после завершения работы)" if self.deferred else ""
        return f'- {{"op": "{self.name}"{"" if not arguments else ", " + arguments}}} — {self.summary}{marker}'


class WorldAgent:
    """Агент, который правит миры по указанию словами."""

    def __init__(
        self,
        db: NovelDB,
        client: FreeTokenClient,
        author: Author | None = None,
        registry: Any | None = None,
        current_world_id: int | None = None,
        settings_store: Any | None = None,
    ) -> None:
        self.db = db
        self.client = client
        self.author = author or Author(client)
        self.registry = registry
        #: Хранилище настроек: агенту нужно менять качество и размер кадров.
        self.settings_store = settings_store
        #: Мир, открытый в интерфейсе. Действия без явного номера мира
        #: применяются к нему, а не к первому попавшемуся.
        self.current_world_id = current_world_id
        self.operations: dict[str, Operation] = {}
        self._register()

    # --- доступные действия -------------------------------------------------

    def _register(self) -> None:
        """Объявляет действия, которые агент умеет выполнять."""
        self._add(Operation(
            "list_worlds", "показать миры с номерами и содержимым", {},
            lambda **_: {"worlds": worlds_digest(self.db, current=self.current_world_id)},
        ))
        self._add(Operation(
            "create_world",
            "создать новый мир",
            {"name": "строка", "brief": "строка", "genre": "строка",
             "tone": "строка", "style": "строка по-английски"},
            self._op_create_world,
        ))
        self._add(Operation(
            "draft_world",
            "СОЗДАТЬ НОВЫЙ ОТДЕЛЬНЫЙ МИР с нуля по короткой идее: описание, "
            "правила, персонажи, партия. Не вызывай это, чтобы дополнить уже "
            "существующий мир — для этого есть fill_world",
            {"idea": "строка"},
            self._op_draft_world,
        ))
        self._add(Operation(
            "fill_world",
            "дополнить существующий мир по идее: придумать и добавить правила, "
            "персонажей и описание, не создавая новый мир",
            {"idea": "строка", "world_id": "число, необязательно — по умолчанию открытый"},
            self._op_fill_world,
        ))
        self._add(Operation(
            "set_world",
            "изменить поля мира",
            {"world_id": "число", "field": "name|brief|genre|tone|style|narrator|format",
             "value": "строка"},
            self._op_set_world,
        ))
        self._add(Operation(
            "add_rule", "добавить правило в мир",
            {"world_id": "число", "title": "строка", "body": "строка"},
            self._op_add_rule,
        ))
        self._add(Operation(
            "update_rule", "изменить правило",
            {"rule_id": "число", "title": "строка, необязательно",
             "body": "строка, необязательно", "enabled": "true|false, необязательно"},
            self._op_update_rule,
        ))
        self._add(Operation(
            "rewrite_rule", "переписать правило по указанию",
            {"rule_id": "число", "instruction": "строка"},
            self._op_rewrite_rule,
        ))
        self._add(Operation(
            "delete_rule", "удалить правило", {"rule_id": "число"},
            self._op_delete_rule, destructive=True,
        ))
        self._add(Operation(
            "add_character", "добавить персонажа",
            {"world_id": "число", "name": "строка", "role": "строка",
             "description": "строка", "appearance": "строка по-английски", "speech": "строка"},
            self._op_add_character,
        ))
        self._add(Operation(
            "update_character", "изменить персонажа",
            {"character_id": "число", "name": "строка, необязательно",
             "role": "строка, необязательно", "description": "строка, необязательно",
             "appearance": "строка, необязательно", "speech": "строка, необязательно"},
            self._op_update_character,
        ))
        self._add(Operation(
            "delete_character", "удалить персонажа", {"character_id": "число"},
            self._op_delete_character, destructive=True,
        ))
        self._add(Operation(
            "add_scenarios", "придумать завязки для мира и вернуть их текстом",
            {"world_id": "число", "count": "число"},
            self._op_add_scenarios,
        ))
        self._add(Operation(
            "critique_world", "разобрать мир и найти слабые места",
            {"world_id": "число"}, self._op_critique,
        ))
        self._add(Operation(
            "rewrite_text",
            "переписать произвольный текст. Вспомогательное: возвращает готовый "
            "текст и НИЧЕГО не меняет — чтобы сохранить, передай его в другое действие",
            {"text": "строка", "instruction": "строка"}, self._op_rewrite_text,
        ))
        self._add(Operation(
            "create_session", "создать новую партию в мире, чтобы начать с чистого листа",
            {"world_id": "число", "title": "строка"},
            self._op_create_session,
        ))
        self._add(Operation(
            "show_image_settings",
            "показать, как сейчас рисуются кадры: стиль мира и настройки генератора",
            {}, self._op_show_image_settings,
        ))
        self._add(Operation(
            "set_world_style",
            "задать стиль изображений мира — то, что уходит генератору вместе с "
            "описанием кадра: реализм, живопись, освещение, палитра",
            {"world_id": "число, необязательно — по умолчанию открытый",
             "style": "строка по-английски, например hyper-realistic, oil painting"},
            self._op_set_world_style,
        ))
        self._add(Operation(
            "set_reply_length",
            "задать длину ответа ведущего и запас под размышления. Длина — это "
            "предел, из которого ведущий пишет текст хода; размышления идут по "
            "тому же счёту, и думающей модели нужен запас, иначе она не начнёт "
            "ответ. Нижняя граница — 200 токенов, верхняя — 8000",
            {"max_tokens": "200..8000, необязательно",
             "reasoning_reserve_tokens": "0..16000, необязательно"},
            self._op_set_reply_length,
        ))
        self._add(Operation(
            "set_image_settings",
            "изменить качество генерации кадров: ступень быстро/обычно/качественно, "
            "размер, число шагов; и что рисовать в переписке — портрет или сцену",
            {"image_quality": "fast | normal | quality, необязательно",
             "chat_frame": "portrait | scene, необязательно",
             "image_size": "256..2048, кратно 32, необязательно (переводит на custom)",
             "image_steps": "4..60, необязательно (переводит на custom)",
             "image_cfg": "1.0..4.0, необязательно; у Qwen-Image держи 1.0"},
            self._op_set_image_settings,
        ))
        self._add(Operation(
            "set_world_image_suffix",
            "дописать к каждому промпту кадра одно и то же. Уходит генератору "
            "дословно, ведущий этого не видит — так добавляют то, что модель "
            "писать отказывается",
            {"world_id": "номер мира, необязательно — по умолчанию открытый",
             "suffix": "строка по-английски; пустая строка снимает дополнение"},
            self._op_set_world_image_suffix,
        ))
        self._add(Operation(
            "set_scene_prompt",
            "переписать описание кадра и поставить его на перерисовку. Описание "
            "уходит генератору как есть, минуя ведущего",
            {"scene_id": "номер кадра",
             "prompt": "новое описание по-английски"},
            self._op_set_scene_prompt,
        ))
        self._add(Operation(
            "add_item",
            "положить вещь в мир: игроку или персонажу. Ведущий о ней узнает, но "
            "пользоваться за игрока не станет",
            {"world_id": "число, необязательно — по умолчанию открытый",
             "name": "строка", "properties": "строка: чем полезна или опасна",
             "character_id": "число, необязательно; без него вещь у игрока"},
            self._op_add_item,
        ))
        self._add(Operation(
            "delete_item", "убрать вещь из мира", {"item_id": "число"},
            self._op_delete_item, destructive=True,
        ))
        self._add(Operation(
            "list_models", "показать доступные модели и ту, что включена сейчас",
            {}, self._op_list_models,
        ))
        self._add(Operation(
            "switch_model",
            "переключить модель движка; выполняется после завершения работы, "
            "потому что движок придётся перезапустить",
            {"model": "часть имени модели, например gpt-oss или Qwen"},
            self._op_switch_model, deferred=True,
        ))

    def _add(self, operation: Operation) -> None:
        self.operations[operation.name] = operation

    def _current_world_line(self) -> str:
        """Строка про мир, открытый в интерфейсе, для промпта."""
        if self.current_world_id is None:
            return "ни один мир не открыт — обязательно указывай номер мира"
        world = self.db.world(self.current_world_id)
        if world is None:
            return f"мир #{self.current_world_id} не найден"
        return f"мир #{world.id} «{world.name}»"

    def tool_list(self) -> str:
        """Описание действий для промпта."""
        return "\n".join(operation.signature() for operation in self.operations.values())

    # --- выполнение действий ------------------------------------------------

    def _op_create_world(self, name: str = "", brief: str = "", genre: str = "",
                         tone: str = "", style: str = "", **_: Any) -> dict[str, Any]:
        if not str(name).strip():
            raise ValueError("нужно название мира")
        world_id = self.db.create_world(
            name=str(name).strip(), format="story", brief=str(brief).strip(),
            genre=str(genre).strip(), tone=str(tone).strip(), style=str(style).strip(),
        )
        session_id = self.db.create_session(world_id, "Первая партия")
        return {"world_id": world_id, "session_id": session_id, "changed": True}

    def _op_draft_world(self, idea: str = "", **_: Any) -> dict[str, Any]:
        if not str(idea).strip():
            raise ValueError("нужна идея мира")
        from novel.authoring import save_world

        drafted = self.author.draft_world(str(idea))
        world_id, session_id = save_world(self.db, drafted)
        return {
            "world_id": world_id,
            "session_id": session_id,
            "name": drafted.name,
            "rules": len(drafted.rules),
            "characters": len(drafted.characters),
            "note": "создан новый мир",
            "changed": True,
        }

    def _op_fill_world(self, idea: str = "", world_id: Any = None, **_: Any) -> dict[str, Any]:
        """Дополняет существующий мир, не создавая новый.

        Нужно для обычного случая: пользователь открыл свой мир и просит добавить
        туда лор, правила и персонажей. Придуманное берётся у модели, но остаётся
        в том же мире.

        @param idea: что именно нужно добавить, словами.
        @param world_id: мир; по умолчанию открытый в интерфейсе.
        @raises ValueError: если идея пуста.
        """
        if not str(idea).strip():
            raise ValueError("нужна идея: что добавить в мир")
        world = self._world(world_id)

        drafted = self.author.draft_world(str(idea))
        added_rules = 0
        for rule in drafted.rules:
            if rule.get("body"):
                self.db.add_rule(world.id, rule["body"], title=rule.get("title", ""))
                added_rules += 1
        added_characters = 0
        for character in drafted.characters:
            if character.get("name"):
                self.db.add_character(
                    world.id,
                    character["name"],
                    role=character.get("role", ""),
                    description=character.get("description", ""),
                    appearance=character.get("appearance", ""),
                    speech=character.get("speech", ""),
                )
                added_characters += 1

        filled: list[str] = []
        if not world.brief.strip() and drafted.brief:
            self.db.update_world(world.id, {"brief": drafted.brief})
            filled.append("brief")
        if not world.style.strip() and drafted.style:
            self.db.update_world(world.id, {"style": drafted.style})
            filled.append("style")
        if not world.genre.strip() and drafted.genre:
            self.db.update_world(world.id, {"genre": drafted.genre})
            filled.append("genre")

        return {
            "world_id": world.id,
            "rules": added_rules,
            "characters": added_characters,
            "filled": filled,
            "changed": added_rules + added_characters + len(filled) > 0,
        }

    def _op_set_world(self, world_id: Any = None, field: str = "", value: str = "", **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        wanted = _world_field(field)
        rejected = self.db.update_world(world.id, {wanted: value})
        if rejected:
            raise ValueError(
                f"у мира нет поля {field}; доступны: name, brief (описание), "
                "genre, tone, style, narrator, format"
            )
        return {"world_id": world.id, "field": wanted, "changed": True}

    def _op_add_rule(self, world_id: Any = None, title: str = "", body: str = "", **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        if not str(body).strip():
            raise ValueError("у правила нет текста")
        rule_id = self.db.add_rule(world.id, str(body).strip(), title=str(title).strip())
        return {"rule_id": rule_id, "world_id": world.id, "changed": True}

    def _op_update_rule(self, rule_id: Any = None, **changes: Any) -> dict[str, Any]:
        rule = self._rule(rule_id)
        payload = {key: value for key, value in changes.items() if value is not None and key != "world_id"}
        if "enabled" in payload:
            payload["enabled"] = 1 if payload["enabled"] in (True, "true", "да", 1, "1") else 0
        rejected = self.db.update_rule(rule.id, payload)
        if rejected:
            raise ValueError(f"у правила нет поля {rejected[0]}")
        return {"rule_id": rule.id, "changed": True}

    def _op_rewrite_rule(self, rule_id: Any = None, instruction: str = "", **_: Any) -> dict[str, Any]:
        rule = self._rule(rule_id)
        if not str(instruction).strip():
            raise ValueError("нужно указание, как переписать правило")
        source = f"{rule.title}: {rule.body}" if rule.title else rule.body
        rewritten = self.author.rewrite(source, str(instruction)).strip()
        title, body = _split_rule(rewritten, rule.title)
        self.db.update_rule(rule.id, {"title": title, "body": body})
        return {"rule_id": rule.id, "body": body, "changed": True}

    def _op_delete_rule(self, rule_id: Any = None, **_: Any) -> dict[str, Any]:
        rule = self._rule(rule_id)
        self.db.delete_rule(rule.id)
        return {"rule_id": rule.id, "deleted": True, "changed": True}

    def _op_add_character(self, world_id: Any = None, name: str = "", role: str = "",
                          description: str = "", appearance: str = "", speech: str = "",
                          **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        if not str(name).strip():
            raise ValueError("у персонажа нет имени")
        character_id = self.db.add_character(
            world.id, str(name).strip(), role=str(role).strip(),
            description=str(description).strip(), appearance=str(appearance).strip(),
            speech=str(speech).strip(),
        )
        return {"character_id": character_id, "world_id": world.id, "changed": True}

    def _op_update_character(self, character_id: Any = None, **changes: Any) -> dict[str, Any]:
        character = self._character(character_id)
        payload = {key: value for key, value in changes.items() if value is not None and key != "world_id"}
        if "enabled" in payload:
            payload["enabled"] = 1 if payload["enabled"] in (True, "true", "да", 1, "1") else 0
        rejected = self.db.update_character(character.id, payload)
        if rejected:
            raise ValueError(f"у персонажа нет поля {rejected[0]}")
        return {"character_id": character.id, "changed": True}

    def _op_delete_character(self, character_id: Any = None, **_: Any) -> dict[str, Any]:
        character = self._character(character_id)
        self.db.delete_character(character.id)
        return {"character_id": character.id, "deleted": True, "changed": True}

    def _op_add_scenarios(self, world_id: Any = None, count: Any = 3, **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        try:
            amount = max(1, min(int(count), 8))
        except (TypeError, ValueError):
            amount = 3
        scenarios = self.author.scenarios(world, self.db.rules(world.id), amount)
        return {"scenarios": scenarios, "changed": False}

    def _op_critique(self, world_id: Any = None, **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        report = self.author.critique(world, self.db.rules(world.id), self.db.characters(world.id))
        return {"critique": report, "changed": False}

    def _op_rewrite_text(self, text: str = "", instruction: str = "", **_: Any) -> dict[str, Any]:
        if not str(text).strip() or not str(instruction).strip():
            raise ValueError("нужны текст и указание")
        return {"text": self.author.rewrite(str(text), str(instruction)), "changed": False}

    def _op_create_session(self, world_id: Any = None, title: str = "", **_: Any) -> dict[str, Any]:
        world = self._world(world_id)
        session_id = self.db.create_session(world.id, str(title).strip() or "Новая партия")
        return {"session_id": session_id, "world_id": world.id, "changed": True}

    def _op_show_image_settings(self, **_: Any) -> dict[str, Any]:
        """Показывает, из чего складывается вид кадров.

        Два разных рычага, и путать их нельзя: стиль мира уходит генератору
        словами вместе с описанием кадра, а размер, шаги и cfg — это уже
        устройство самой генерации.
        """
        world = (
            self.db.world(self.current_world_id) if self.current_world_id else None
        )
        settings = self.settings_store.settings if self.settings_store else None
        result: dict[str, Any] = {"changed": False}
        if world is not None:
            result["world_id"] = world.id
            result["style"] = world.style or "(не задан)"
        if settings is not None:
            result["generator"] = {
                "image_size": settings.image_size,
                "image_steps": settings.image_steps,
                "image_cfg": settings.image_cfg,
            }
        result["подсказка"] = (
            "реалистичность задаётся стилем мира: hyper-realistic, photorealistic, "
            "detailed skin texture, natural lighting. Живописность — oil painting, "
            "watercolor. Качество и время — шагами: 25 обычно достаточно, 40 — "
            "предел разумного"
        )
        return result

    def _op_set_world_style(self, world_id: Any = None, style: str = "", **_: Any) -> dict[str, Any]:
        """Задаёт стиль изображений мира.

        Стиль уходит в каждый кадр этого мира, поэтому именно он отвечает за
        реалистичность и общий вид картинок.

        @param world_id: мир; по умолчанию открытый.
        @param style: набор ключевых слов по-английски.
        @raises ValueError: если стиль пуст.
        """
        cleaned = " ".join((style or "").split())
        if not cleaned:
            raise ValueError("пустой стиль: опиши, как должны выглядеть кадры")
        world = self._world(world_id)
        previous = (world.style or "").strip()
        rejected = self.db.update_world(world.id, {"style": cleaned})
        if rejected:
            raise ValueError(f"у мира нет поля {rejected[0]}")
        result: dict[str, Any] = {"world_id": world.id, "style": cleaned, "changed": True}
        if previous and previous != cleaned:
            # Образец места уходит генератору картинкой, а стиль — словами. Если
            # стиль сменился сильно, старые образцы будут тянуть кадры назад.
            result["note"] = (
                "новый стиль применится к следующим кадрам, но образцы уже "
                "нарисованных мест остались в прежнем виде и будут спорить с ним — "
                "если стиль сменился сильно, назначь образцы мест заново"
            )
        return result

    def _op_set_reply_length(
        self,
        max_tokens: Any = None,
        reasoning_reserve_tokens: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Задаёт длину ответа ведущего и запас под размышления.

        Длина ответа — это предел, из которого ведущий пишет текст хода.
        Размышления идут по тому же счёту, поэтому у думающих моделей запас
        задаётся отдельно: без него модель тратит весь предел на размышления и
        не начинает ответ.

        @param max_tokens: видимая длина ответа ведущего.
        @param reasoning_reserve_tokens: запас под размышления.
        @raises ValueError: если значение вне допустимых рамок.
        @returns: что записано.
        """
        if self.settings_store is None:
            raise ValueError("настройки недоступны")
        changes: dict[str, Any] = {}

        if max_tokens not in (None, ""):
            try:
                value = int(max_tokens)
            except (TypeError, ValueError):
                raise ValueError("длина ответа должна быть числом") from None
            if not 200 <= value <= 8000:
                raise ValueError(
                    "длина ответа от 200 до 8000 токенов: меньше — ответ обрывается "
                    "на полуслове, больше — не помещается в окно движка"
                )
            changes["max_tokens"] = value

        if reasoning_reserve_tokens not in (None, ""):
            try:
                reserve = int(reasoning_reserve_tokens)
            except (TypeError, ValueError):
                raise ValueError("запас под размышления должен быть числом") from None
            if not 0 <= reserve <= 16000:
                raise ValueError("запас под размышления от 0 до 16000 токенов")
            changes["reasoning_reserve_tokens"] = reserve

        if not changes:
            raise ValueError("не сказано, что менять: нужна длина ответа или запас")

        self.settings_store.settings.update(changes)
        self.settings_store.save()
        return {**changes, "changed": True}

    def _op_set_image_settings(
        self,
        image_quality: Any = None,
        image_size: Any = None,
        image_steps: Any = None,
        image_cfg: Any = None,
        chat_frame: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Меняет качество генерации кадров.

        Ступень качества (`fast`, `normal`, `quality`) задаёт размер и шаги
        готовым набором — это обычный способ. Точные числа (`image_size`,
        `image_steps`) переводят на ступень `custom`: они нужны редко, потому что
        размер обязан быть кратен 32, а время растёт как точки на шаги.

        @raises ValueError: если значение вне допустимых рамок.
        """
        if self.settings_store is None:
            raise ValueError("настройки недоступны")
        settings = self.settings_store.settings
        changes: dict[str, Any] = {}

        if image_quality not in (None, ""):
            quality = str(image_quality).strip().lower()
            if quality not in IMAGE_QUALITY:
                raise ValueError(
                    "ступень качества бывает: " + ", ".join(IMAGE_QUALITY)
                )
            changes["image_quality"] = quality

        if chat_frame not in (None, ""):
            frame = str(chat_frame).strip().lower()
            if frame not in ("portrait", "scene"):
                raise ValueError("кадр в переписке бывает portrait или scene")
            changes["chat_frame"] = frame

        if image_size not in (None, ""):
            try:
                size = int(image_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"непонятный размер кадра: {image_size}") from exc
            if not 256 <= size <= 2048:
                raise ValueError("размер кадра должен быть от 256 до 2048")
            if size % 32:
                raise ValueError("размер кадра должен быть кратен 32")
            changes["image_size"] = size

        if image_steps not in (None, ""):
            try:
                steps = int(image_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"непонятное число шагов: {image_steps}") from exc
            if not 4 <= steps <= 60:
                raise ValueError("число шагов должно быть от 4 до 60")
            changes["image_steps"] = steps

        if image_cfg not in (None, ""):
            try:
                cfg = float(image_cfg)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"непонятное значение cfg: {image_cfg}") from exc
            if not 1.0 <= cfg <= 4.0:
                raise ValueError("cfg должен быть от 1.0 до 4.0")
            changes["image_cfg"] = cfg

        # Точные числа без ступени означают, что набор больше не подходит.
        if ("image_size" in changes or "image_steps" in changes) \
                and "image_quality" not in changes:
            changes["image_quality"] = "custom"

        if not changes:
            raise ValueError("не указано, что менять")

        for key, value in changes.items():
            setattr(settings, key, value)
        self.settings_store.save()
        return {**changes, "changed": True, "note": "применится к следующим кадрам"}

    def _op_set_world_image_suffix(
        self, world_id: Any = None, suffix: str = "", **_: Any
    ) -> dict[str, Any]:
        """Задаёт дополнение, которое приписывается к каждому промпту кадра.

        Уходит генератору дословно и в самом конце промпта. Ведущий его не видит
        и переписать не может — так обходят отказ модели писать откровенное.

        @raises ValueError: если указан несуществующий мир.
        """
        world = self._world(world_id)
        text = " ".join(str(suffix or "").split())
        rejected = self.db.update_world(world.id, {"image_suffix": text})
        if rejected:
            raise ValueError(f"не принято: {', '.join(rejected)}")
        return {
            "world_id": world.id,
            "image_suffix": text,
            "changed": True,
            "note": "дополнение уходит генератору дословно" if text
                    else "дополнение снято",
        }

    def _op_set_scene_prompt(
        self, scene_id: Any = None, prompt: str = "", **_: Any
    ) -> dict[str, Any]:
        """Меняет описание кадра и ставит его в очередь на перерисовку.

        Описание уходит генератору как есть, минуя ведущего.

        @raises ValueError: если сцена не найдена или описание пустое.
        """
        try:
            number = int(scene_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("не указан номер кадра") from exc
        scene = self.db.scene(number)
        if scene is None:
            raise ValueError(f"кадр {number} не найден")
        text = " ".join(str(prompt or "").split())
        if not text:
            raise ValueError("пустое описание кадра")
        self.db.set_scene_prompt(number, text)
        return {
            "scene_id": number,
            "prompt": text,
            "status": "pending",
            "changed": True,
            "note": "кадр перерисуется при следующем запуске генерации",
        }

    def _op_add_item(self, world_id: Any = None, name: str = "", properties: str = "",
                     character_id: Any = None, **_: Any) -> dict[str, Any]:
        """Кладёт вещь в мир.

        @param world_id: мир; по умолчанию открытый.
        @param name: название вещи.
        @param properties: чем вещь полезна или опасна.
        @param character_id: владелец-персонаж; без него вещь у игрока.
        @raises ValueError: если нет названия или владелец не найден.
        """
        world = self._world(world_id)
        if not str(name).strip():
            raise ValueError("у вещи нет названия")
        owner = None
        if character_id not in (None, "", 0, "0"):
            try:
                owner = int(character_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"непонятный номер владельца: {character_id}") from exc
            if self._character(owner).world_id != world.id:
                raise ValueError(f"персонаж {owner} из другого мира")
        item_id = self.db.add_item(
            world.id, str(name).strip(), character_id=owner,
            properties=str(properties).strip(),
        )
        return {"item_id": item_id, "world_id": world.id, "changed": True}

    def _op_delete_item(self, item_id: Any = None, **_: Any) -> dict[str, Any]:
        try:
            index = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"непонятный номер вещи: {item_id}") from exc
        if self.db.item(index) is None:
            raise ValueError(f"вещь {index} не найдена")
        self.db.delete_item(index)
        return {"item_id": index, "deleted": True, "changed": True}

    def _op_list_models(self, **_: Any) -> dict[str, Any]:
        """Показывает, из чего выбирать при смене модели."""
        if self.registry is None:
            return {"models": [], "note": "реестр моделей недоступен", "changed": False}
        models = [
            {"name": item.name, "type": item.model_type, "size_gb": item.size_gb}
            for item in self.registry.all()
            if item.supported
        ]
        return {"models": models[:20], "changed": False}

    def _op_switch_model(self, model: str = "", **_: Any) -> dict[str, Any]:
        """Проверяет, что модель существует; сама смена идёт после прогона.

        @param model: часть имени модели.
        @raises ValueError: если подходящей модели нет.
        """
        if not str(model).strip():
            raise ValueError("не указана модель")
        if self.registry is None:
            raise ValueError("реестр моделей недоступен")
        found = find_model(self.registry, str(model))
        if found is None:
            raise ValueError(f"модель «{model}» не найдена среди поддерживаемых")
        return {"model": found.name, "path": found.path, "changed": False}

    def _world(self, world_id: Any) -> Any:
        """Находит мир, к которому относится действие.

        Если номер не указан, берётся мир, открытый в интерфейсе. Это главное
        лекарство от путаницы: без него агент, получив «добавь правило», не знает,
        о каком мире речь, и либо переспрашивает, либо трогает не тот мир.

        @param world_id: номер мира; пустой означает «тот, что открыт».
        @raises ValueError: если мира нет и открытого тоже.
        """
        if world_id in (None, "", 0, "0"):
            if self.current_world_id is None:
                raise ValueError("мир не указан, и ни один мир не открыт")
            world_id = self.current_world_id
        try:
            found = self.db.world(int(world_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"непонятный номер мира: {world_id}") from exc
        if found is None:
            raise ValueError(f"мир {world_id} не найден")
        return found

    def _rule(self, rule_id: Any) -> Rule:
        try:
            index = int(rule_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"непонятный номер правила: {rule_id}") from exc
        row = self.db.conn.execute("SELECT * FROM world_rules WHERE id = ?", (index,)).fetchone()
        if row is None:
            raise ValueError(f"правило {index} не найдено")
        return Rule(**{key: row[key] for key in row.keys()})

    def _character(self, character_id: Any) -> Character:
        try:
            index = int(character_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"непонятный номер персонажа: {character_id}") from exc
        row = self.db.conn.execute("SELECT * FROM characters WHERE id = ?", (index,)).fetchone()
        if row is None:
            raise ValueError(f"персонаж {index} не найден")
        return Character(**{key: row[key] for key in row.keys()})

    # --- цикл ---------------------------------------------------------------

    def execute(self, action: dict[str, Any], *, allow_destructive: bool) -> dict[str, Any]:
        """Выполняет одно действие.

        @param action: словарь вида ``{"op": "add_rule", "world_id": 3, ...}``.
        @param allow_destructive: разрешены ли удаления.
        @returns: отчёт о выполнении; ``ok`` показывает, удалось ли.
        """
        name = str(action.get("op") or "").strip()
        operation = self.operations.get(name)
        if operation is None:
            return {"op": name, "ok": False, "error": f"действия «{name}» не существует"}
        if operation.destructive and not allow_destructive:
            return {
                "op": name,
                "ok": False,
                "error": "удаление запрещено: включи разрешение на удаление",
            }
        payload = {key: value for key, value in action.items() if key != "op"}
        try:
            result = operation.handler(**payload)
        except (ValueError, AuthoringError, FreeTokenError) as exc:
            return {"op": name, "ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — действие не должно ронять агента
            return {"op": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("ok", True)
        return {"op": name, **result}

    def run(
        self,
        instruction: str,
        *,
        preview: bool = False,
        allow_destructive: bool = False,
        max_steps: int = MAX_STEPS,
        should_continue: Callable[[], bool] | None = None,
        revive: Callable[[], bool] | None = None,
        world_id: int | None = None,
    ) -> AgentRun:
        """Выполняет указание пользователя.

        Движок может быть выгружен прямо посреди работы: его останавливают перед
        генерацией картинки, при смене модели и кнопкой «Стоп всё». Поэтому между
        шагами проверяется, не пора ли остановиться, а неудачный запрос к движку
        один раз повторяется после попытки поднять движок заново. Уже выполненные
        действия при этом сохраняются: каждое пишется в базу отдельно, и отчёт
        показывает, что именно успело примениться.

        @param instruction: что нужно сделать, словами.
        @param preview: показать план и ничего не менять.
        @param allow_destructive: разрешить удаления.
        @param max_steps: предел шагов «подумал — сделал».
        @param should_continue: проверка перед каждым шагом; ``False`` означает
            остановку по внешней команде.
        @param revive: попытка поднять движок заново; возвращает ``True`` при успехе.
        @param world_id: мир, открытый в интерфейсе. Действия без явного номера
            мира применяются к нему.
        @returns: отчёт со всеми шагами и результатами.
        """
        if world_id is not None:
            self.current_world_id = world_id
        started = time.time()
        run = AgentRun(instruction=instruction.strip(), preview=preview)
        if not run.instruction:
            run.error = "пустое указание"
            return run

        system = AGENT_SYSTEM_PROMPT.format(
            tools=self.tool_list(),
            state=worlds_digest(self.db, current=self.current_world_id),
            current=self._current_world_line(),
        )
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": run.instruction},
        ]

        try:
            self.client.configure_reasoning("off")
        except FreeTokenError:
            pass

        #: Отпечатки уже выполненных действий: по ним ловится хождение по кругу.
        applied_signatures: set[str] = set()
        #: Прозу вместо JSON просим переоформить один раз за прогон.
        asked_again = False
        #: Шаги подряд, не изменившие ничего: по ним ловится топтание.
        idle_steps = 0
        for _ in range(max(1, max_steps)):
            if should_continue is not None and not should_continue():
                run.interrupted = True
                run.error = "остановлено: движок выгружается"
                break

            duplicates = 0

            reply, revived, failure, interrupted = self._ask(
                conversation, revive, should_continue
            )
            run.revived = run.revived or revived
            if reply is None:
                run.error = failure
                run.interrupted = interrupted
                break

            payload = _parse_reply(reply.text)
            if payload is None:
                # Модель ответила прозой вместо JSON. У reasoning-моделей это
                # типичный срыв: размышления обрываются по бюджету и продолжаются
                # уже в ответе. Один раз просим переоформить — обычно этого
                # хватает, а без повтора прогон обрывается сразу.
                if not asked_again:
                    asked_again = True
                    conversation.append({"role": "assistant", "content": reply.text})
                    conversation.append({
                        "role": "user",
                        "content": "Ответ не разобран: нужен ТОЛЬКО JSON, без пояснений "
                                   "и без markdown. Повтори его в виде "
                                   '{"thought": "...", "actions": [...], "done": true}.',
                    })
                    continue
                run.error = "модель вернула не JSON"
                run.steps.append(AgentStep(thought=reply.text.strip()[:300]))
                break

            thought = str(payload.get("thought") or "").strip()
            actions = payload.get("actions")
            if not isinstance(actions, list):
                actions = []
            step = AgentStep(thought=thought, actions=[a for a in actions if isinstance(a, dict)])

            if preview:
                for action in step.actions:
                    step.results.append({"op": str(action.get("op") or ""), "preview": True})
                run.steps.append(step)
                run.finished = bool(payload.get("done"))
                break

            for action in step.actions:
                name = str(action.get("op") or "")
                signature = _signature(action)
                if signature in applied_signatures:
                    duplicates += 1
                    step.results.append({
                        "op": name, "ok": False, "changed": False, "duplicate": True,
                        "note": "такое действие уже выполнено — пропускаю повтор",
                    })
                    continue
                operation = self.operations.get(name)
                if operation is not None and operation.deferred:
                    # Действие проверяется сразу, а выполняется после прогона:
                    # смена модели перезапускает движок, на котором агент сейчас
                    # работает. Проверка нужна, чтобы агент узнал об ошибке —
                    # скажем, опечатке в имени модели — ещё внутри прогона и мог
                    # её исправить.
                    check = self.execute(action, allow_destructive=allow_destructive)
                    if not check.get("ok"):
                        step.results.append(check)
                        continue
                    run.pending.append(action)
                    applied_signatures.add(signature)
                    step.results.append({
                        "op": name,
                        "ok": True,
                        "changed": False,
                        "deferred": True,
                        "note": "выполнится после завершения работы",
                        **{key: value for key, value in check.items()
                           if key not in ("op", "ok", "changed")},
                    })
                    continue
                step.results.append(self.execute(action, allow_destructive=allow_destructive))
                applied_signatures.add(signature)
            run.steps.append(step)

            # Шаг, который ничего не изменил, — это или подготовка, или топтание.
            # Одно-два таких терпим: модель могла получить текст и собраться его
            # куда-то вложить. Три подряд означают, что она ходит по кругу.
            if any(result.get("changed") for result in step.results):
                idle_steps = 0
            else:
                idle_steps += 1
            if idle_steps >= IDLE_STEPS_LIMIT:
                run.error = (
                    f"{IDLE_STEPS_LIMIT} шага подряд без единого изменения — "
                    "похоже, модель ходит по кругу"
                )
                break

            # Модель повторила ровно то, что уже сделала: дальше она будет
            # повторяться и дальше, а мир заполнится копиями. Считаем работу
            # законченной — это честнее, чем гонять её до предела шагов.
            if step.actions and duplicates == len(step.actions):
                run.finished = True
                run.looped = True
                run.error = "модель повторила те же действия — работа остановлена"
                break

            if payload.get("done") or not step.actions:
                run.finished = bool(payload.get("done")) or not step.actions
                break

            conversation.append({"role": "assistant", "content": reply.text})
            conversation.append({
                "role": "user",
                "content": "Результаты выполнения:\n"
                + _format_results(step.results)
                + "\n\nПродолжай или заверши, вернув JSON.",
            })

        run.seconds = time.time() - started
        if not run.finished and not run.error and not run.interrupted:
            # Сообщение говорит и о сделанном: «не уложился» само по себе читается
            # как «ничего не вышло», хотя изменения в мире уже есть.
            done = run.changed
            run.error = (
                f"не уложился в {max_steps} шагов; применено изменений: {done}"
                if done else f"не уложился в {max_steps} шагов; ничего не изменено"
            )
        return run

    def _ask(
        self,
        conversation: list[dict[str, str]],
        revive: Callable[[], bool] | None,
        should_continue: Callable[[], bool] | None,
    ) -> tuple[Any, bool, str | None, bool]:
        """Запрашивает у модели следующий шаг, переживая выгрузку движка.

        Движок останавливают штатно — перед генерацией картинки, при смене модели
        и кнопкой «Стоп всё». Для агента это выглядит как обрыв связи на середине
        работы, поэтому один раз делается попытка поднять движок и повторить
        запрос: чаще всего выгрузка временная.

        Если остановку запросил пользователь, движок не поднимается: это стоило бы
        около минуты и всей видеопамяти ради того, чтобы сразу же остановиться.

        @param conversation: переписка с моделью.
        @param revive: попытка поднять движок; ``None`` — не пытаться.
        @param should_continue: проверка внешней остановки.
        @returns: ``(ответ или None, поднимали ли движок, текст ошибки, прервано ли)``.
        """
        revived = False
        for attempt in (1, 2):
            try:
                return (
                    self.client.chat(
                        conversation, max_tokens=1600, temperature=0.3, timeout_s=900.0
                    ),
                    revived,
                    None,
                    False,
                )
            except FreeTokenError as exc:
                if should_continue is not None and not should_continue():
                    return None, revived, "остановлено: движок выгружается", True
                if attempt == 2 or revive is None:
                    return None, revived, str(exc), False
                if not revive():
                    return None, revived, f"движок выгружен и не поднялся: {exc}", False
                revived = True
        return None, revived, "не удалось получить ответ модели", False


def find_model(registry: Any, query: str) -> Any | None:
    """Ищет модель по части имени или пути.

    Ведущий называет модель как удобно — «gpt-oss», «квен», «джемма», — поэтому
    сначала ищется точное совпадение, затем совпадение по началу, затем по
    вхождению подстроки.

    @param registry: реестр моделей.
    @param query: то, что назвал пользователь.
    @returns: найденная модель либо ``None``.
    """
    needle = query.strip().casefold()
    if not needle:
        return None
    models = [item for item in registry.all() if item.supported]
    for item in models:
        if needle in (item.name.casefold(), item.path.casefold()):
            return item
    for item in models:
        if item.name.casefold().startswith(needle):
            return item
    for item in models:
        if needle in item.name.casefold() or needle in item.path.casefold():
            return item
    return None


def _split_rule(text: str, fallback_title: str) -> tuple[str, str]:
    """Разбирает переписанное правило на заголовок и текст.

    @param text: ответ модели.
    @param fallback_title: заголовок, если модель его не повторила.
    @returns: ``(заголовок, текст)``.
    """
    stripped = text.strip().strip('"')
    head, sep, tail = stripped.partition(":")
    if sep and len(head) <= 40 and tail.strip():
        return head.strip(), tail.strip()
    return fallback_title, stripped


def _parse_reply(text: str) -> dict[str, Any] | None:
    """Разбирает ответ модели, возвращая ``None`` при неудаче."""
    try:
        payload = loads_json(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _format_results(results: list[dict[str, Any]]) -> str:
    """Собирает результаты действий в текст для следующего шага модели."""
    lines: list[str] = []
    for result in results:
        op = result.get("op", "?")
        if result.get("ok"):
            details = ", ".join(
                f"{key}={value}" for key, value in result.items()
                if key not in ("op", "ok", "changed") and not isinstance(value, (dict, list))
            )
            lines.append(f"- {op}: выполнено" + (f" ({details})" if details else ""))
        else:
            lines.append(f"- {op}: ошибка — {result.get('error')}")
    return "\n".join(lines) or "- действий не было"


def worlds_digest(db: NovelDB, *, max_worlds: int = 6, current: int | None = None) -> str:
    """Краткая сводка миров для промпта агента.

    Показывает номера, которые нужны агенту, чтобы не выдумывать идентификаторы,
    и не разрастается: правила и персонажи обрезаются. Открытый мир идёт первым и
    помечается — иначе агент, глядя на восемь миров, хватается не за тот.

    @param db: хранилище.
    @param max_worlds: сколько миров показывать.
    @param current: номер мира, открытого в интерфейсе.
    @returns: текст сводки.
    """
    worlds = db.worlds()
    if current is not None:
        worlds.sort(key=lambda item: (item.id != current, item.id))
    worlds = worlds[:max_worlds]
    if not worlds:
        return "миров пока нет"
    lines: list[str] = []
    for world in worlds:
        rules = db.rules(world.id)
        characters = db.characters(world.id)
        sessions = db.sessions(world.id)
        mark = "  ← ОТКРЫТ СЕЙЧАС" if world.id == current else ""
        lines.append(
            f"#{world.id} «{world.name}» ({world.format}) — правил {len(rules)}, "
            f"персонажей {len(characters)}, партий {len(sessions)}{mark}"
        )
        if world.brief:
            lines.append(f"    описание: {world.brief[:160]}")
        if rules:
            listing = " | ".join(f"#{rule.id} {rule.title or rule.body[:25]}" for rule in rules[:8])
            lines.append(f"    правила: {listing}")
        if characters:
            listing = " | ".join(f"#{item.id} {item.name}" for item in characters[:8])
            lines.append(f"    персонажи: {listing}")
    return "\n".join(lines)
