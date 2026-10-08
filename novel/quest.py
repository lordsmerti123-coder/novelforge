"""Состояние квеста: переходы, вещи, этапы и исход.

Квест — это мир с заранее написанным сюжетом: места, предметы, этапы и условия
победы и поражения известны заранее. Ведущий ведёт по этому сюжету, а программа
следит, чтобы он не выдумывал лишнего: нельзя взять предмет, которого нет на
месте, нельзя пройти сквозь стену и нельзя перескочить этап.

Ведущий сообщает об изменениях отдельным блоком ``<quest>``, а всё недопустимое
отклоняется и запоминается. Отклонённое видно ведущему следующим ходом, поэтому
он поправляется сам, без вмешательства игрока.
"""

from __future__ import annotations

from typing import Any

from novel.protocol import loads_json

#: Ключ состояния партии, под которым лежит состояние квеста.
STATE_KEY = "quest"

#: Ключ состояния партии, под которым записан играемый квест.
QUEST_KEY = "quest_key"

#: Допустимые исходы.
OUTCOMES = ("win", "lose")

#: Сколько последних замечаний держать в памяти. Больше не нужно: ведущему
#: важно то, что он сделал не так только что.
LOG_LIMIT = 8


def list_quests() -> list[dict[str, Any]]:
    """Краткие описания квестов для интерфейса.

    @returns: список описаний; пустой, если квестов нет.
    """
    try:
        from novel.quests import list_quests as _list
    except ImportError:
        return []
    return _list()


def get_quest(key: str) -> dict[str, Any] | None:
    """Квест по ключу.

    @param key: ключ квеста, например ``office``.
    @returns: словарь квеста либо ``None``, если такого нет.
    """
    if not key:
        return None
    try:
        from novel.quests import get_quest as _get
    except ImportError:
        return None
    return _get(key)


def initial_state(quest: dict[str, Any]) -> dict[str, Any]:
    """Начальное состояние квеста.

    Предметы без места игрок получает сразу: так описываются вещи, с которыми
    он пришёл.

    @param quest: словарь квеста.
    @returns: состояние.
    """
    carried = [
        str(item.get("id"))
        for item in quest.get("items") or []
        if item.get("id") and not str(item.get("where") or "").strip()
    ]
    return {
        "stage": 0,
        "location": str(quest.get("start") or ""),
        "carried": carried,
        "used": [],
        "flags": [],
        "status": "active",
        "log": [],
    }


def read_state(db: Any, session_id: int, quest: dict[str, Any]) -> dict[str, Any]:
    """Читает состояние квеста, дополняя недостающее начальным.

    @param db: хранилище.
    @param session_id: партия.
    @param quest: словарь квеста.
    @returns: состояние.
    """
    stored = db.get_state(session_id, STATE_KEY, None)
    state = initial_state(quest)
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key in state:
                state[key] = value
    # Локация могла исчезнуть из квеста после правки: тогда игрок возвращается
    # на старт, а не остаётся в несуществующем месте.
    if not _location(quest, state.get("location")):
        state["location"] = str(quest.get("start") or "")
    return state


def write_state(db: Any, session_id: int, state: dict[str, Any]) -> None:
    """Записывает состояние квеста.

    @param db: хранилище.
    @param session_id: партия.
    @param state: состояние.
    """
    db.set_state(session_id, STATE_KEY, state)


def _location(quest: dict[str, Any], key: Any) -> dict[str, Any] | None:
    """Локация квеста по id.

    @param quest: словарь квеста.
    @param key: id локации.
    @returns: описание локации либо ``None``.
    """
    wanted = str(key or "")
    if not wanted:
        return None
    for place in quest.get("locations") or []:
        if str(place.get("id")) == wanted:
            return place
    return None


def _item(quest: dict[str, Any], key: Any) -> dict[str, Any] | None:
    """Предмет квеста по id.

    @param quest: словарь квеста.
    @param key: id предмета.
    @returns: описание предмета либо ``None``.
    """
    wanted = str(key or "")
    if not wanted:
        return None
    for thing in quest.get("items") or []:
        if str(thing.get("id")) == wanted:
            return thing
    return None


def _names(quest: dict[str, Any], keys: list[Any]) -> list[str]:
    """Называет предметы так, как их видит игрок.

    @param quest: словарь квеста.
    @param keys: id предметов.
    @returns: названия; для незнакомого id — сам id.
    """
    out: list[str] = []
    for key in keys:
        thing = _item(quest, key)
        out.append(str(thing.get("name")) if thing else str(key))
    return out


def _as_keys(value: Any) -> list[str]:
    """Приводит поле отчёта к списку id.

    Ведущий пишет то строкой, то списком, поэтому принимается и то, и другое.

    @param value: значение поля.
    @returns: список id.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def parse_report(body: str) -> dict[str, Any]:
    """Разбирает блок ``<quest>``.

    @param body: содержимое блока.
    @returns: отчёт; пустой словарь, если разобрать не удалось.
    """
    if not body or not body.strip():
        return {}
    try:
        payload = loads_json(body)
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_count(quest: dict[str, Any]) -> int:
    """Сколько этапов в квесте.

    @param quest: словарь квеста.
    @returns: число этапов.
    """
    return len(quest.get("stages") or [])


def apply_report(
    db: Any,
    session_id: int,
    quest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Применяет отчёт ведущего, отклоняя недопустимое.

    @param db: хранилище.
    @param session_id: партия.
    @param quest: словарь квеста.
    @param report: разобранный блок ``<quest>``.
    @returns: словарь с ключами ``state``, ``applied`` и ``rejected``.
    """
    state = read_state(db, session_id, quest)
    applied: list[str] = []
    rejected: list[str] = []

    if state.get("status") in OUTCOMES:
        # Квест кончился: дальше менять нечего, и ведущему об этом сообщается.
        return {"state": state, "applied": applied, "rejected": rejected,
                "finished": True}

    _apply_location(quest, state, report, applied, rejected)
    _apply_take(quest, state, report, applied, rejected)
    _apply_use(quest, state, report, applied, rejected)
    _apply_flags(state, report, applied)
    _apply_stage(quest, state, report, applied, rejected)
    _apply_end(state, report, applied, rejected)

    if applied or rejected:
        state["log"] = (list(state.get("log") or []) + applied + rejected)[-LOG_LIMIT:]
    write_state(db, session_id, state)
    return {"state": state, "applied": applied, "rejected": rejected,
            "finished": state.get("status") in OUTCOMES}


def _apply_location(
    quest: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    applied: list[str],
    rejected: list[str],
) -> None:
    """Переносит игрока, если переход возможен."""
    raw = str(report.get("location") or "").strip()
    if not raw:
        return
    here = str(state.get("location") or "")
    if raw == here:
        return
    # Ведущий пишет то код места, то его название — принимается и то, и другое.
    # По одному коду он промахивается: в отчёте стоит «переговорная», а в квесте
    # место записано как ``conference_room``, и переход отклонялся с издевательским
    # «такого места в квесте нет», хотя место есть.
    place = _location(quest, raw) or match_location(quest, raw)
    if place is None:
        rejected.append(f"переход в «{raw}» невозможен: такого места в квесте нет")
        return
    target = str(place.get("id"))
    if target == here:
        return
    current = _location(quest, here)
    exits = [str(x) for x in (current or {}).get("exits") or []]
    if target not in exits:
        here_name = str((current or {}).get("name") or here)
        rejected.append(
            f"переход в «{place.get('name')}» невозможен: из «{here_name}» туда нет пути"
        )
        return
    state["location"] = target
    applied.append(f"игрок перешёл в «{place.get('name')}»")


def _apply_take(
    quest: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    applied: list[str],
    rejected: list[str],
) -> None:
    """Кладёт предмет в руки, если он лежит на месте."""
    carried = [str(x) for x in state.get("carried") or []]
    used = [str(x) for x in state.get("used") or []]
    here = str(state.get("location") or "")
    for key in _as_keys(report.get("take")):
        thing = _item(quest, key)
        label = str(thing.get("name")) if thing else key
        if thing is None:
            rejected.append(f"взять «{key}» нельзя: такого предмета в квесте нет")
            continue
        if key in carried or key in used:
            rejected.append(f"«{label}» уже у игрока")
            continue
        where = str(thing.get("where") or "").strip()
        if where and where != here:
            place = _location(quest, where)
            here_name = str((_location(quest, here) or {}).get("name") or here)
            rejected.append(
                f"взять «{label}» нельзя: он лежит не здесь"
                f" (ищите в «{(place or {}).get('name') or where}», а игрок в «{here_name}»)"
            )
            continue
        carried.append(key)
        applied.append(f"игрок взял «{label}»")
    state["carried"] = carried


def _apply_use(
    quest: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    applied: list[str],
    rejected: list[str],
) -> None:
    """Тратит предмет, если он у игрока."""
    carried = [str(x) for x in state.get("carried") or []]
    used = [str(x) for x in state.get("used") or []]
    for key in _as_keys(report.get("use")):
        thing = _item(quest, key)
        label = str(thing.get("name")) if thing else key
        if key not in carried:
            if key in used:
                rejected.append(f"«{label}» уже потрачен")
            else:
                rejected.append(f"применить «{label}» нельзя: его нет у игрока")
            continue
        carried = [x for x in carried if x != key]
        used.append(key)
        applied.append(f"игрок применил «{label}»")
    state["carried"] = carried
    state["used"] = used


def _apply_flags(state: dict[str, Any], report: dict[str, Any], applied: list[str]) -> None:
    """Запоминает отметки о случившемся."""
    flags = [str(x) for x in state.get("flags") or []]
    for mark in _as_keys(report.get("flag")):
        if mark not in flags:
            flags.append(mark)
            applied.append(f"отмечено: {mark}")
    state["flags"] = flags


def _apply_stage(
    quest: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    applied: list[str],
    rejected: list[str],
) -> None:
    """Двигает этап не больше чем на один вперёд и только по улике."""
    raw = report.get("stage")
    if raw is None or raw == "":
        return
    try:
        wanted = int(raw)
    except (TypeError, ValueError):
        rejected.append(f"номер этапа «{raw}» не число")
        return
    current = int(state.get("stage") or 0)
    total = _stage_count(quest)
    if wanted == current:
        return
    if wanted < 0 or wanted >= max(total, 1):
        rejected.append(f"этапа с номером {wanted + 1} в квесте нет")
        return
    if wanted < current:
        rejected.append("назад по этапам не возвращаются")
        return
    if wanted > current + 1:
        rejected.append(
            f"через этап перескочить нельзя: сейчас {current + 1} из {total}, "
            f"а ведущий назвал {wanted + 1}"
        )
        return
    # Этап движется уликой, а не словом ведущего: он называл следующий номер
    # каждый ход, и квест добегал до последнего этапа, пока игрок стоял на месте.
    # Улика — любое другое изменение в том же отчёте: переход, взятое,
    # применённое или отметка.
    if not applied:
        rejected.append(
            "этап не принят: в этом ходу ничего не изменилось — ни перехода, "
            "ни находки, ни применения"
        )
        return
    state["stage"] = wanted
    stages = quest.get("stages") or []
    title = str((stages[wanted] or {}).get("title") or wanted + 1) if wanted < len(stages) else wanted
    applied.append(f"начался этап {wanted + 1}: «{title}»")


def _apply_end(
    state: dict[str, Any],
    report: dict[str, Any],
    applied: list[str],
    rejected: list[str],
) -> None:
    """Записывает исход, если он назван."""
    raw = report.get("end")
    if raw is None or raw == "":
        return
    outcome = str(raw).strip().lower()
    if outcome not in OUTCOMES:
        rejected.append(f"исход «{raw}» неизвестен: бывает только win или lose")
        return
    state["status"] = outcome
    applied.append("квест выигран" if outcome == "win" else "квест проигран")


def sync_inventory(
    db: Any,
    world_id: int,
    quest: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Приводит инвентарь игрока к состоянию квеста.

    Вещи квеста живут в его состоянии, а панель инвентаря читает таблицу вещей:
    без этой сверки игрок не видел бы, что у него в руках. Метка в свойствах
    отличает вещи квеста от тех, что заведены вручную, — чужие не трогаются.

    @param db: хранилище.
    @param world_id: мир.
    @param quest: словарь квеста.
    @param state: состояние квеста.
    """
    carried = [str(x) for x in state.get("carried") or []]
    definition = {str(x.get("id")): x for x in quest.get("items") or []}
    prefix = f"квест:{quest.get('key')}:"

    existing: dict[str, Any] = {}
    try:
        rows = db.items(world_id)
    except (AttributeError, TypeError):
        return
    for row in rows:
        properties = str(getattr(row, "properties", "") or "")
        if properties.startswith(prefix):
            existing[properties[len(prefix):]] = row

    for key in carried:
        if key in existing:
            continue
        thing = definition.get(key) or {}
        db.add_item(
            world_id,
            str(thing.get("name") or key),
            description=str(thing.get("description") or ""),
            properties=prefix + key,
        )

    for key, row in existing.items():
        if key not in carried:
            db.delete_item(int(getattr(row, "id")))


#: Буквы, которые ведущий пишет по-разному в одном и том же слове.
#:
#: Только «ё»: она в наборе часто заменяется на «е». Заменять «й» на «и» нельзя —
#: от этого «серверной» перестаёт сходиться с «Серверная».
_LETTER_SWAP = str.maketrans({"ё": "е"})

#: Короче этого начала слова места не сравниваются: «зал» совпал бы с «залом»,
#: «заливом» и чем угодно ещё.
MATCH_MIN_LENGTH = 4

#: Какая доля более короткого названия обязана совпасть. Меньше половины —
#: и «коридор» сошёлся бы с «котельной».
MATCH_SHARE = 0.6


def _stem(name: str) -> str:
    """Приводит название места к виду, годному для сравнения.

    Ведущий называет одно место по-разному: «Ресепшн» в квесте и «в ресепшене»
    в ответе. Убираются регистр, знаки и короткие слова-предлоги; окончание не
    срезается — его учитывает :func:`_same_place`.

    @param name: название места.
    @returns: основа слова; пустая строка, если сравнивать нечего.
    """
    cleaned = "".join(ch for ch in str(name or "").lower() if ch.isalnum() or ch == " ")
    # «в», «на», «из» и прочие предлоги к названию места не относятся, а сверку
    # ломают: «в коридоре» и «Коридор» иначе не сойдутся никогда.
    words = [word for word in cleaned.split() if len(word) > 2]
    return " ".join(words).translate(_LETTER_SWAP)


def _common_prefix(left: str, right: str) -> int:
    """Сколько знаков у двух названий совпадает с начала.

    @param left: первое название.
    @param right: второе название.
    @returns: число совпадающих знаков.
    """
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _same_place(left: str, right: str) -> bool:
    """Считает два названия одним местом.

    Сравнивается общее начало, а не всё слово: у русского прилагательного на
    конце меняется не одна буква («серверная» и «серверной» расходятся в двух
    знаках), и проверка «всё, кроме последней» их разводит. Доля совпадения
    берётся от более короткого названия.

    @param left: первое название.
    @param right: второе название.
    @returns: ``True``, если это одно и то же место.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    shorter = min(len(left), len(right))
    needed = max(MATCH_MIN_LENGTH, int(shorter * MATCH_SHARE))
    return _common_prefix(left, right) >= needed


def match_location(quest: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Ищет локацию квеста по названию от ведущего.

    Сравнивается начало слова, а не всё название: ведущий пишет «ресепшен» там,
    где в квесте «Ресепшн», и точного совпадения не будет никогда. Проверка
    идёт по обоим направлениям — что названо, и как записано в квесте.

    @param quest: словарь квеста.
    @param name: как место назвал ведущий.
    @returns: описание локации либо ``None``, если такой в квесте нет.
    """
    wanted = _stem(name)
    if len(wanted) < MATCH_MIN_LENGTH:
        return None
    for place in quest.get("locations") or []:
        if _same_place(wanted, _stem(str(place.get("name") or ""))):
            return place
    return None


def sync_location(
    db: Any,
    session_id: int,
    quest: dict[str, Any],
    state: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Переносит игрока по названию, которое назвал ведущий.

    Доверять блоку ``<quest>`` целиком нельзя: ведущий часто его не ставит, и
    тогда состояние застывает, хотя история уже ушла вперёд. Зато место он
    называет всегда — по нему и сверяемся. Переход принимается только если он
    возможен, поэтому выдуманное место состояние не сдвинет.

    @param db: хранилище.
    @param session_id: партия.
    @param quest: словарь квеста.
    @param state: состояние квеста.
    @param name: место от ведущего.
    @returns: словарь с ключами ``changed`` и ``state``.
    """
    if is_finished(state):
        return {"changed": False, "state": state}
    place = match_location(quest, name)
    if place is None:
        return {"changed": False, "state": state}
    target = str(place.get("id"))
    if target == str(state.get("location") or ""):
        return {"changed": False, "state": state}

    here = _location(quest, state.get("location"))
    exits = [str(x) for x in (here or {}).get("exits") or []]
    if target not in exits:
        # Место названо, но из текущего туда нет хода. Состояние не двигается, а
        # отказ запоминается: следующим ходом ведущий увидит его и поправится.
        here_name = str((here or {}).get("name") or state.get("location") or "")
        note = (f"переход в «{place.get('name')}» невозможен: "
                f"из «{here_name}» туда нет пути")
        state["log"] = (list(state.get("log") or []) + [note])[-LOG_LIMIT:]
        write_state(db, session_id, state)
        return {"changed": False, "state": state, "unreachable": str(place.get("name"))}

    applied: list[str] = []
    rejected: list[str] = []
    _apply_location(quest, state, {"location": target}, applied, rejected)
    if applied:
        state["log"] = (list(state.get("log") or []) + applied)[-LOG_LIMIT:]
        write_state(db, session_id, state)
    return {"changed": bool(applied), "state": state}


def is_finished(state: dict[str, Any]) -> bool:
    """Кончился ли квест.

    @param state: состояние квеста.
    @returns: ``True``, если есть исход.
    """
    return str(state.get("status") or "") in OUTCOMES


def outcome(state: dict[str, Any]) -> str:
    """Исход квеста.

    @param state: состояние квеста.
    @returns: ``win``, ``lose`` или пустая строка.
    """
    status = str(state.get("status") or "")
    return status if status in OUTCOMES else ""


def carried_names(quest: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Что у игрока в руках, названиями.

    @param quest: словарь квеста.
    @param state: состояние квеста.
    @returns: названия предметов.
    """
    return _names(quest, list(state.get("carried") or []))


def current_stage(quest: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Текущий этап квеста.

    @param quest: словарь квеста.
    @param state: состояние квеста.
    @returns: описание этапа; пустой словарь, если этапов нет.
    """
    stages = quest.get("stages") or []
    index = int(state.get("stage") or 0)
    if 0 <= index < len(stages):
        return dict(stages[index])
    return {}


def current_location(quest: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Где игрок сейчас.

    @param quest: словарь квеста.
    @param state: состояние квеста.
    @returns: описание локации; пустой словарь, если её нет.
    """
    return dict(_location(quest, state.get("location")) or {})
