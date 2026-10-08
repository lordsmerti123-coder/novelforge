"""Сочинение силами локальной модели: миры, правила, персонажи, сценарии.

Модуль превращает запущенный движок в писателя. Он не участвует в игровом ходе
и ничего не хранит сам: на вход идут данные из хранилища, на выходе — структура,
которую вызывает и сохраняет тот, кто попросил.

Зачем это отдельно от игрового протокола: у сочинения другая задача. Ведущий
обязан двигать сцену и держаться формата с тегами, а писатель должен придумать
стройную структуру и вернуть её одним JSON без служебных блоков. Смешивать эти
роли в одном запросе — значит получить посредственное и то и другое.

Размышления модели здесь выключаются так же, как в игре: для сочинения они не
нужны, а бюджет тратят.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from novel.db import Character, NovelDB, Rule, World
from novel.freetoken import FreeTokenClient, FreeTokenError
from novel.protocol import loads_json

DRAFT_WORLD_PROMPT = """Ты придумываешь мир для текстовой игры с ИИ-ведущим.

Идея автора:
{idea}

Верни только JSON, без пояснений и без markdown:

{{
  "name": "короткое название мира",
  "genre": "жанр в двух-трёх словах",
  "tone": "тон повествования",
  "narrator": "от какого лица ведётся повествование",
  "style": "стиль иллюстраций по-английски: oil painting, dark fantasy, ...",
  "brief": "описание мира на 4-6 предложений: место, время, расстановка сил, положение игрока",
  "rules": [
    {{"title": "короткое имя", "body": "что запрещено или обязательно"}}
  ],
  "characters": [
    {{"name": "имя", "role": "кто это", "description": "характер и цели",
      "appearance": "внешность по-английски, для генератора изображений",
      "speech": "как говорит"}}
  ]
}}

Требования:
- Пиши по-русски. По-английски заполняется только поле appearance: оно уходит \
прямо в генератор изображений, и переводить его нельзя.
- rules — от 4 до 7 правил, которые действительно ограничивают ведущего: чего в мире \
не бывает, как работает сила или магия, за что персонажи платят. Не пересказывай ими описание.
- characters — от 3 до 5 ключевых персонажей с заполненным appearance.
- Опирайся на идею автора, но дополняй её конкретикой: имена, места, конфликты.
- Никаких общих слов вроде «мрачный мир, полный опасностей».
"""

MORE_RULES_PROMPT = """Ты дописываешь правила для уже существующего мира.

Мир: {name}
Жанр: {genre}. Тон: {tone}
Описание: {brief}

Уже есть правила:
{existing}

Придумай ещё {count} правил, которые не повторяют существующие и закрывают то, \
о чём ведущий мог бы поспорить сам с собой: границы возможного, цена силы, \
последствия, запреты. Пиши по-русски.

Верни только JSON-массив:
[{{"title": "короткое имя", "body": "что запрещено или обязательно"}}]
"""

MORE_CHARACTERS_PROMPT = """Ты дописываешь персонажей для существующего мира.

Мир: {name}
Описание: {brief}

Уже есть:
{existing}

Придумай ещё {count} персонажей, которые создают напряжение рядом с уже \
существующими: соперник, должник, свидетель, тот, кому нельзя отказать. \
Пиши по-русски; по-английски только appearance.

Верни только JSON-массив:
[{{"name": "имя", "role": "кто это", "description": "характер и цели",
  "appearance": "внешность по-английски, для генератора изображений",
  "speech": "как говорит"}}]
"""

SCENARIOS_PROMPT = """Ты придумываешь завязки для игры в готовом мире.

Мир: {name}
Жанр: {genre}. Тон: {tone}
Описание: {brief}

Правила:
{rules}

Придумай {count} разных завязок. Каждая должна начинаться с конкретного события, \
ставить игрока перед выбором и опираться на правила мира. Никаких «ты просыпаешься \
в незнакомом месте». Пиши по-русски.

Верни только JSON-массив:
[{{"title": "короткое название", "opening": "первая реплика ведущего, 3-5 предложений",
  "hook": "в чём конфликт и что на кону", "twist": "что может выясниться позже"}}]
"""

REWRITE_PROMPT = """Перепиши текст по указанию автора.

Указание: {instruction}

Текст:
{text}

Верни только переписанный текст, без пояснений и без кавычек вокруг него.
"""

CRITIQUE_PROMPT = """Ты редактор. Оцени мир для текстовой игры и найди слабые места.

Мир: {name}
Жанр: {genre}. Тон: {tone}
Описание: {brief}

Правила:
{rules}

Персонажи:
{characters}

Верни только JSON:
{{
  "verdict": "одна фраза: годится или нет",
  "problems": ["конкретная проблема", "..."],
  "fixes": ["что именно сделать", "..."],
  "missing": ["чего не хватает: завязка, конфликт, цена, ..."]
}}

Ищи конкретное: правила, которые ничего не запрещают; персонажи без цели; \
описание, из которого непонятно, что игрок делает в этом мире; отсутствие цены \
за силу; отсутствие конфликта.
"""


class AuthoringError(RuntimeError):
    """Модель не вернула пригодную структуру."""


@dataclass
class AuthoredWorld:
    """Мир, придуманный моделью."""

    name: str
    brief: str
    genre: str = ""
    tone: str = ""
    narrator: str = ""
    style: str = ""
    rules: list[dict[str, str]] = field(default_factory=list)
    characters: list[dict[str, str]] = field(default_factory=list)
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Представление для отчёта и записи в файл."""
        return {
            "name": self.name,
            "brief": self.brief,
            "genre": self.genre,
            "tone": self.tone,
            "narrator": self.narrator,
            "style": self.style,
            "rules": self.rules,
            "characters": self.characters,
        }


class Author:
    """Обёртка над движком для сочинения."""

    def __init__(self, client: FreeTokenClient, *, max_tokens: int = 2000) -> None:
        self.client = client
        self.max_tokens = max_tokens
        client.configure_reasoning("off")

    def _ask(self, prompt: str, *, temperature: float = 0.8, max_tokens: int | None = None) -> str:
        """Один запрос к модели.

        @param prompt: текст задания.
        @param temperature: температура; для сочинения выше, для разбора ниже.
        @param max_tokens: предел ответа.
        @returns: текст ответа.
        @raises AuthoringError: если движок недоступен или ответ пуст.
        """
        try:
            result = self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                timeout_s=900.0,
            )
        except FreeTokenError as exc:
            raise AuthoringError(str(exc)) from exc
        text = result.text.strip()
        if not text:
            raise AuthoringError("модель вернула пустой ответ")
        return text

    def _ask_json(self, prompt: str, *, temperature: float = 0.8) -> Any:
        """Запрос с разбором JSON-ответа.

        @raises AuthoringError: если разобрать не удалось.
        """
        text = self._ask(prompt, temperature=temperature)
        try:
            return loads_json(text)
        except ValueError as exc:
            raise AuthoringError(f"не удалось разобрать JSON: {exc}") from exc

    # --- сочинение ----------------------------------------------------------

    def draft_world(self, idea: str) -> AuthoredWorld:
        """Придумывает мир по короткой идее автора.

        @param idea: одна-две фразы о том, какой мир нужен.
        @returns: готовый мир с правилами и персонажами.
        """
        payload = self._ask_json(DRAFT_WORLD_PROMPT.format(idea=idea.strip()))
        if not isinstance(payload, dict):
            raise AuthoringError("модель вернула не объект JSON")
        name = str(payload.get("name") or "").strip()
        brief = str(payload.get("brief") or "").strip()
        if not name or not brief:
            raise AuthoringError("в ответе нет названия или описания мира")
        return AuthoredWorld(
            name=name,
            brief=brief,
            genre=str(payload.get("genre") or "").strip(),
            tone=str(payload.get("tone") or "").strip(),
            narrator=str(payload.get("narrator") or "").strip(),
            style=str(payload.get("style") or "").strip(),
            rules=_clean_rules(payload.get("rules")),
            characters=_clean_characters(payload.get("characters")),
            raw=json.dumps(payload, ensure_ascii=False),
        )

    def more_rules(self, world: World, existing: list[Rule], count: int = 3) -> list[dict[str, str]]:
        """Дописывает правила к существующему миру."""
        text = "\n".join(f"- {rule.title}: {rule.body}" for rule in existing) or "- правил пока нет"
        payload = self._ask_json(
            MORE_RULES_PROMPT.format(
                name=world.name, genre=world.genre or "не задан", tone=world.tone or "не задан",
                brief=world.brief or "не задано", existing=text, count=count,
            )
        )
        return _clean_rules(payload)

    def more_characters(
        self, world: World, existing: list[Character], count: int = 2
    ) -> list[dict[str, str]]:
        """Дописывает персонажей к существующему миру."""
        text = "\n".join(f"- {item.name} ({item.role})" for item in existing) or "- пока никого"
        payload = self._ask_json(
            MORE_CHARACTERS_PROMPT.format(
                name=world.name, brief=world.brief or "не задано", existing=text, count=count
            )
        )
        return _clean_characters(payload)

    def scenarios(self, world: World, rules: list[Rule], count: int = 5) -> list[dict[str, str]]:
        """Придумывает завязки для мира."""
        rules_text = "\n".join(f"- {rule.title}: {rule.body}" for rule in rules) or "- правил нет"
        payload = self._ask_json(
            SCENARIOS_PROMPT.format(
                name=world.name, genre=world.genre or "не задан", tone=world.tone or "не задан",
                brief=world.brief or "не задано", rules=rules_text, count=count,
            )
        )
        if not isinstance(payload, list):
            raise AuthoringError("модель вернула не массив завязок")
        cleaned: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            opening = str(item.get("opening") or "").strip()
            if not opening:
                continue
            cleaned.append({
                "title": str(item.get("title") or "").strip() or "Без названия",
                "opening": opening,
                "hook": str(item.get("hook") or "").strip(),
                "twist": str(item.get("twist") or "").strip(),
            })
        return cleaned

    def rewrite(self, text: str, instruction: str) -> str:
        """Переписывает произвольный текст по указанию.

        @param text: исходный текст.
        @param instruction: что именно изменить.
        @returns: переписанный текст.
        """
        return self._ask(
            REWRITE_PROMPT.format(instruction=instruction.strip(), text=text.strip()),
            temperature=0.6,
            max_tokens=max(400, len(text) // 2),
        )

    def critique(
        self, world: World, rules: list[Rule], characters: list[Character]
    ) -> dict[str, Any]:
        """Разбирает мир и находит слабые места."""
        payload = self._ask_json(
            CRITIQUE_PROMPT.format(
                name=world.name, genre=world.genre or "не задан", tone=world.tone or "не задан",
                brief=world.brief or "не задано",
                rules="\n".join(f"- {r.title}: {r.body}" for r in rules) or "- правил нет",
                characters="\n".join(f"- {c.name} ({c.role}): {c.description}" for c in characters)
                or "- никого",
            ),
            temperature=0.4,
        )
        return payload if isinstance(payload, dict) else {"raw": payload}


def _clean_rules(payload: Any) -> list[dict[str, str]]:
    """Приводит правила из ответа модели к единому виду."""
    if not isinstance(payload, list):
        return []
    rules: list[dict[str, str]] = []
    for item in payload:
        if isinstance(item, str):
            body = item.strip()
            if body:
                rules.append({"title": "", "body": body})
            continue
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or "").strip()
        if not body:
            continue
        rules.append({"title": str(item.get("title") or "").strip(), "body": body})
    return rules


def _clean_characters(payload: Any) -> list[dict[str, str]]:
    """Приводит персонажей из ответа модели к единому виду."""
    if not isinstance(payload, list):
        return []
    characters: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        characters.append({
            "name": name,
            "role": str(item.get("role") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "appearance": str(item.get("appearance") or "").strip(),
            "speech": str(item.get("speech") or "").strip(),
        })
    return characters


def save_world(db: NovelDB, authored: AuthoredWorld, *, title: str | None = None) -> tuple[int, int]:
    """Записывает придуманный мир в хранилище вместе с первой партией.

    @param db: хранилище.
    @param authored: результат работы :class:`Author`.
    @param title: название мира; по умолчанию берётся из ответа модели.
    @returns: ``(идентификатор мира, идентификатор партии)``.
    """
    world_id = db.create_world(
        name=title or authored.name,
        format="story",
        brief=authored.brief,
        genre=authored.genre,
        tone=authored.tone,
        narrator=authored.narrator,
        style=authored.style,
    )
    for rule in authored.rules:
        db.add_rule(world_id, rule["body"], title=rule["title"])
    for character in authored.characters:
        db.add_character(
            world_id,
            character["name"],
            role=character["role"],
            description=character["description"],
            appearance=character["appearance"],
            speech=character["speech"],
        )
    session_id = db.create_session(world_id, title="Первая партия")
    return world_id, session_id
