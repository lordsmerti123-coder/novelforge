"""Разбор ответа LLM на теги ``<prose>``, ``<scene>`` и ``<speculative>``.

Ответ модели — гибрид: обычный текст для игрока плюс машинные блоки для
оркестратора. Игроку уходит **только** ``<prose>``; ``<scene>`` запускает
генерацию картинки, ``<speculative>`` кладётся в очередь на простой. Обратно в
контекст модели не возвращается ни то, ни другое — это односторонний поток.

Разбор намеренно терпимый. Модель может не закрыть тег, обернуть JSON в
markdown-ограждение, поставить висячую запятую или выдать половину полей;
единственный по-настоящему плохой исход — потерять текст для игрока, поэтому
``prose`` возвращается даже тогда, когда машинные блоки разобрать не удалось.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PROSE = "prose"
SCENE = "scene"
SPECULATIVE = "speculative"
#: Блок с номерами отложенных заметок, которые ведущий считает сбывшимися.
NOTES = "notes"
#: Блок с тем, как персонажи выглядят теперь: переоделся, ранен, загримирован.
LOOKS = "looks"

_TAG_RE = re.compile(
    # Список блоков собирается из констант: добавив блок, нельзя забыть про
    # разбор — иначе он молча не находится.
    r"<(?P<name>" + "|".join((PROSE, SCENE, SPECULATIVE, NOTES, LOOKS)) + r")\b[^>]*>"
    r"(?P<body>.*?)(?:</(?P=name)\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class SceneSpec:
    """Описание сцены для генерации изображения."""

    image_prompt: str
    style: str = ""
    location: str = ""
    npc: list[str] = field(default_factory=list)
    seed: int | None = None
    refs: list[str] = field(default_factory=list)
    #: Резкая перемена обстановки на уже знакомом месте — гроза, пожар, бой,
    #: ночь вместо дня. Единственный случай, когда кадр рисуется не из-за нового
    #: места или нового лица.
    sudden: bool = False

    @property
    def full_prompt(self) -> str:
        """Промпт для генератора: описание сцены плюс стиль."""
        parts = [self.image_prompt.strip()]
        if self.style.strip():
            parts.append(self.style.strip())
        return ", ".join(part for part in parts if part)


@dataclass
class SpeculativeSpec:
    """Один вариант предгенерации, привязанный к вероятному действию игрока."""

    trigger: str
    prompt: str
    seed: int | None = None


@dataclass
class ParsedReply:
    """Результат разбора ответа модели."""

    prose: str
    scene: SceneSpec | None = None
    speculative: list[SpeculativeSpec] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: str = ""
    used_fallback: bool = False
    #: Номера отложенных заметок, которые ведущий считает сбывшимися.
    fulfilled_notes: list[int] = field(default_factory=list)
    #: Номера задумок, которые больше не подходят: ведущий отбросил их сам,
    #: потому что игрок свернул в сторону.
    dropped_notes: list[int] = field(default_factory=list)
    #: Как персонажи выглядят теперь: имя — описание.
    looks: dict[str, str] = field(default_factory=dict)

    @property
    def has_machine_block(self) -> bool:
        """Есть ли что-то для оркестратора, кроме текста для игрока."""
        return (self.scene is not None or bool(self.speculative)
                or bool(self.fulfilled_notes) or bool(self.dropped_notes)
                or bool(self.looks))

    def as_dict(self) -> dict[str, Any]:
        """Представление для журнала и отчёта."""
        return {
            "prose": self.prose,
            "scene": None
            if self.scene is None
            else {
                "image_prompt": self.scene.image_prompt,
                "style": self.scene.style,
                "location": self.scene.location,
                "npc": self.scene.npc,
                "seed": self.scene.seed,
                "sudden": self.scene.sudden,
                "refs": self.scene.refs,
            },
            "speculative": [
                {"trigger": item.trigger, "prompt": item.prompt, "seed": item.seed}
                for item in self.speculative
            ],
            "errors": self.errors,
            "used_fallback": self.used_fallback,
            "fulfilled_notes": self.fulfilled_notes,
            "dropped_notes": self.dropped_notes,
            "looks": self.looks,
        }


def _strip_fence(text: str) -> str:
    """Снимает markdown-ограждение, в которое модель иногда заворачивает JSON."""
    return _FENCE_RE.sub("", text.strip())


def _slice_outermost(text: str, opening: str, closing: str) -> str | None:
    """Вырезает от первой открывающей скобки до последней закрывающей."""
    start = text.find(opening)
    end = text.rfind(closing)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def loads_json(body: str) -> Any:
    """Разбирает JSON с починкой типовых огрехов модели.

    Порядок попыток: как есть, затем без markdown-ограждения, затем с
    вырезанными висячими запятыми, затем по внешним скобкам, затем с заменой
    одинарных кавычек на двойные.

    @param body: текст, который должен содержать JSON.
    @returns: разобранное значение.
    @raises json.JSONDecodeError: если ни одна попытка не удалась.
    """
    candidates: list[str] = []
    stripped = body.strip()
    candidates.append(stripped)
    unfenced = _strip_fence(stripped)
    if unfenced != stripped:
        candidates.append(unfenced)
    for candidate in list(candidates):
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if repaired != candidate:
            candidates.append(repaired)
    for candidate in list(candidates):
        for opening, closing in (("{", "}"), ("[", "]")):
            sliced = _slice_outermost(candidate, opening, closing)
            if sliced and sliced != candidate:
                candidates.append(sliced)
                candidates.append(_TRAILING_COMMA_RE.sub(r"\1", sliced))
    candidates.append(unfenced.replace("'", '"'))

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise last_error or json.JSONDecodeError("пустой блок", "", 0)


def _as_str(value: Any) -> str:
    """Приводит значение к строке, разворачивая список в перечисление."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _as_str_list(value: Any) -> list[str]:
    """Приводит значение к списку строк."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_seed(value: Any) -> int | None:
    """Приводит зерно к целому; нечисловое значение отбрасывается."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _first_key(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Значение первого подходящего ключа, без учёта регистра.

    @param payload: разобранный объект.
    @param keys: допустимые написания ключа.
    @returns: значение либо ``None``.
    """
    lowered = {str(key).strip().casefold(): value for key, value in payload.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _as_id_list(value: Any) -> list[int]:
    """Приводит блок заметок к списку номеров.

    Блок приходит по-разному: ``[1, 2]``, ``{"fulfilled": [1]}`` или строка
    «1, 2». Всё, что не похоже на номер, отбрасывается молча — из-за одного
    мусорного значения терять остальные нельзя.

    @param value: разобранный JSON блока.
    @returns: номера заметок по возрастанию, без повторов.
    """
    if isinstance(value, dict):
        value = value.get("fulfilled") or value.get("notes") or value.get("ids") or []
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list):
        return []
    found: set[int] = set()
    for item in value:
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, int):
            found.add(item)
            continue
        for piece in str(item).replace(";", ",").split(","):
            # Ведущий может дописать к номеру пояснение — «4 (стражник)».
            # Берём ведущее число и не теряем остальные номера из-за этого.
            leading = re.match(r"\s*(\d+)", piece)
            if leading:
                found.add(int(leading.group(1)))
    return sorted(found)


def _as_looks(value: Any) -> dict[str, str]:
    """Приводит блок внешности к словарю «имя — как выглядит».

    Ведущий может прислать объект или список пар. Пустые значения и пустые имена
    отбрасываются молча: из-за одной оговорки терять остальные нельзя.

    @param value: разобранный JSON блока.
    @returns: словарь описаний внешности.
    """
    if isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("who")
                look = item.get("look") or item.get("appearance") or item.get("description")
                if name and look:
                    pairs.append((name, look))
        value = dict(pairs)
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for name, look in value.items():
        clean_name = str(name).strip()
        clean_look = " ".join(str(look).split())
        if clean_name and clean_look:
            result[clean_name] = clean_look
    return result


def looks_from_text(text: str) -> dict[str, str]:
    """Вылавливает изменения внешности из вольного ответа модели.

    Отдельный вопрос про внешность задаётся без формата блоков, и модель отвечает
    как хочет: с пояснениями до и после, в ```json, иногда списком. Поэтому ищем
    первый похожий на объект или массив кусок и разбираем его мягко.

    @param text: ответ модели.
    @returns: словарь «имя — как выглядит», возможно пустой.
    """
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    for candidate in (cleaned, _first_json_object(cleaned), _first_json_array(cleaned)):
        if not candidate:
            continue
        try:
            parsed = loads_json(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        found = _as_looks(parsed)
        if found:
            return found
    return {}


def _first_json_object(text: str) -> str:
    """Первый сбалансированный объект ``{...}`` в тексте."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return ""


def _first_json_array(text: str) -> str:
    """Первый сбалансированный массив ``[...]`` в тексте."""
    start = text.find("[")
    if start == -1:
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def parse_scene(payload: Any, errors: list[str]) -> SceneSpec | None:
    """Собирает сцену из разобранного блока.

    @param payload: словарь из ``<scene>``.
    @param errors: список, куда дописываются замечания разбора.
    @returns: описание сцены или ``None``, если промпта в блоке нет.
    """
    if not isinstance(payload, dict):
        errors.append("блок scene не является объектом JSON")
        return None
    prompt = _as_str(payload.get("image_prompt") or payload.get("prompt"))
    if not prompt:
        errors.append("в блоке scene нет image_prompt")
        return None
    return SceneSpec(
        image_prompt=prompt,
        style=_as_str(payload.get("style")),
        location=_as_str(payload.get("location")),
        npc=_as_str_list(payload.get("npc")),
        seed=_as_seed(payload.get("seed")),
        refs=_as_str_list(payload.get("refs")),
        sudden=bool(payload.get("sudden")),
    )


def parse_speculative(payload: Any, errors: list[str]) -> list[SpeculativeSpec]:
    """Собирает список вариантов предгенерации.

    @param payload: массив из ``<speculative>`` либо один объект.
    @param errors: список, куда дописываются замечания разбора.
    @returns: разобранные варианты; блоки без промпта пропускаются.
    """
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        errors.append("блок speculative не является массивом")
        return []
    items: list[SpeculativeSpec] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        prompt = _as_str(entry.get("prompt") or entry.get("image_prompt"))
        if not prompt:
            continue
        items.append(
            SpeculativeSpec(
                trigger=_as_str(entry.get("trigger")) or "не указан",
                prompt=prompt,
                seed=_as_seed(entry.get("seed")),
            )
        )
    return items


def _parse_json_reply(text: str, result: ParsedReply) -> bool:
    """Разбирает ответ, целиком оформленный объектом JSON.

    Ожидаются ключи ``prose`` и необязательный ``scene``. Разбор принимается,
    только если текста для игрока действительно есть: иначе лучше отдать всё
    как есть, чем показать пустоту.

    @param text: сырой ответ модели.
    @param result: заполняемый разбор.
    @returns: ``True``, если ответ разобран как объект.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = loads_json(stripped)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    prose = payload.get(PROSE)
    if not isinstance(prose, str) or not prose.strip():
        return False

    result.prose = prose.strip()
    result.used_fallback = True
    scene = payload.get(SCENE)
    if isinstance(scene, dict):
        try:
            result.scene = parse_scene(scene, result.errors)
        except (TypeError, ValueError) as exc:
            result.errors.append(f"scene: некорректный объект ({exc})")
    return True


def parse_reply(text: str) -> ParsedReply:
    """Разбирает ответ модели целиком.

    Если тегов нет вовсе, весь ответ считается текстом для игрока: так ведёт
    себя модель, забывшая формат, и терять её ответ нельзя.

    @param text: сырой ответ модели.
    @returns: текст для игрока и разобранные машинные блоки.
    """
    result = ParsedReply(prose="", raw=text)
    blocks: dict[str, str] = {}
    for match in _TAG_RE.finditer(text):
        name = match.group("name").lower()
        blocks.setdefault(name, match.group("body"))

    if PROSE in blocks:
        result.prose = blocks[PROSE].strip()
    elif not blocks:
        # Модель иногда отвечает не тегами, а целым объектом JSON:
        # ``{"prose": "...", "scene": {...}}``. Без этого разбора объект уходил
        # игроку как текст, а сцена внутри него пропадала — на практике в чат
        # попал сырой JSON, и кадр не нарисовался.
        if _parse_json_reply(text, result):
            return result
        result.prose = text.strip()
        result.used_fallback = True
    else:
        # Машинные блоки есть, текста для игрока нет — отдаём всё, кроме блоков.
        result.prose = _TAG_RE.sub("", text).strip()
        result.used_fallback = True

    if SCENE in blocks:
        try:
            result.scene = parse_scene(loads_json(blocks[SCENE]), result.errors)
        except json.JSONDecodeError as exc:
            result.errors.append(f"scene: не удалось разобрать JSON ({exc.msg})")
        except (TypeError, ValueError) as exc:
            result.errors.append(f"scene: некорректный блок ({exc})")

    if SPECULATIVE in blocks:
        try:
            result.speculative = parse_speculative(loads_json(blocks[SPECULATIVE]), result.errors)
        except json.JSONDecodeError as exc:
            result.errors.append(f"speculative: не удалось разобрать JSON ({exc.msg})")
        except (TypeError, ValueError) as exc:
            result.errors.append(f"speculative: некорректный блок ({exc})")

    if NOTES in blocks:
        # Блок необязательный: ведущий ставит его, только когда что-то из
        # задуманного на потом сбылось или перестало подходить. Понимается и
        # голый список номеров, и пара «сбылось / отброшено».
        try:
            payload = loads_json(blocks[NOTES])
            if isinstance(payload, dict):
                # Ключи ведущий пишет как придётся: принимаем все разумные
                # написания, чтобы блок не пропадал из-за одного слова.
                result.fulfilled_notes = _as_id_list(
                    _first_key(payload, ("done", "fulfilled", "notes", "closed"))
                )
                result.dropped_notes = _as_id_list(
                    _first_key(payload, ("dropped", "drop", "cancelled", "canceled"))
                )
            else:
                result.fulfilled_notes = _as_id_list(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            result.errors.append(f"notes: некорректный блок ({exc})")

    if LOOKS in blocks:
        # Необязательный блок: ведущий ставит его, только когда внешность
        # кого-то изменилась — переоделся, ранен, загримирован.
        try:
            result.looks = _as_looks(loads_json(blocks[LOOKS]))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            result.errors.append(f"looks: некорректный блок ({exc})")

    return result
