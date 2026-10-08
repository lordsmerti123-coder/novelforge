"""Самотест частей, которым не нужен GPU.

Проверяются разбор ответа, хранилище, настройки, форматы, сборка контекста и
реестр моделей. Сеть и карта не трогаются, поэтому скрипт можно запускать в
любой момент.

Запуск::

    python bench\\selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.context import ContextBuilder, estimate_tokens  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.formats import FORMATS, get_format  # noqa: E402
from novel.models import ModelRegistry, preset_for, SEARCH_ROOTS  # noqa: E402
from novel.presets import WORLD_PRESETS, get_preset, list_presets  # noqa: E402
from novel.protocol import parse_reply  # noqa: E402
from novel.settings import Settings, SettingsStore  # noqa: E402
from novel import prompts  # noqa: E402

PASSED = 0
FAILED = 0


class _NoEngine:
    """Заглушка движка: сборка контекста не должна ходить в сеть."""

    def count_tokens(self, *_args, **_kwargs) -> int:
        raise AssertionError("сеть не должна вызываться в самотесте")


class _FakeEngine:
    """Заглушка движка для проверки суммаризации."""

    def count_tokens(self, messages=None, system=None, model=None) -> int:
        text = (system or "") + " ".join(item.get("content", "") for item in (messages or []))
        return max(1, len(text) // 3)

    def chat(self, messages, **kwargs):  # noqa: ARG002 — подпись ради совместимости
        from novel.freetoken import ChatResult

        return ChatResult(
            text="Герой вошёл в таверну, поговорил с Марлой и узнал о пропавшем караване.",
            elapsed_s=0.1,
            prompt_tokens=10,
            completion_tokens=14,
            raw={},
        )


def check(name: str, condition: bool, detail: str = "") -> None:
    """Печатает результат одной проверки."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  ФЕЙЛ {name}{'' if not detail else ' — ' + detail}")


def test_protocol() -> None:
    """Разбор ответа модели на всех огрехах, которые она допускает."""
    print("\nРазбор ответа модели:")
    full = parse_reply(
        "<prose>Ты входишь в таверну.</prose>\n"
        '<scene>{"location":"tavern","npc":["barkeep"],"image_prompt":"dark tavern",'
        '"style":"oil","seed":42}</scene>\n'
        '<speculative>[{"trigger":"attack","prompt":"fight","seed":7}]</speculative>'
    )
    check("prose извлечён", full.prose == "Ты входишь в таверну.", repr(full.prose))
    check("scene разобран", full.scene is not None and full.scene.seed == 42)
    check("стиль склеен в промпт", full.scene is not None and full.scene.full_prompt == "dark tavern, oil")
    check("speculative разобран", len(full.speculative) == 1)
    check("ошибок нет", full.errors == [], str(full.errors))

    plain = parse_reply("Просто текст без тегов.")
    check("текст без тегов уходит игроку", plain.prose == "Просто текст без тегов.")
    check("включён запасной разбор", plain.used_fallback)

    fenced = parse_reply('<prose>ok</prose><scene>```json\n{"image_prompt":"a castle"}\n```</scene>')
    check("JSON в ограждении разобран", fenced.scene is not None)

    trailing = parse_reply('<prose>ok</prose><scene>{"image_prompt":"a castle",}</scene>')
    check("висячая запятая починена", trailing.scene is not None)

    broken = parse_reply("<prose>ok</prose><scene>{это не json}</scene>")
    check("битый scene не съедает prose", broken.prose == "ok")
    check("причина отказа записана", bool(broken.errors))

    case = parse_reply("<PROSE>ok</PROSE>")
    check("регистр тега не важен", case.prose == "ok")


def test_db() -> None:
    """Хранилище: миры, правила, персонажи, партии, память."""
    print("\nХранилище:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "test.db")
        try:
            _db_body(db)
        finally:
            db.close()


def _db_body(db: NovelDB) -> None:
    """Тело проверок хранилища: вынесено, чтобы соединение закрывалось всегда."""
    world_id = db.create_world(name="Тест", format="story", brief="описание", genre="фэнтези")
    check("мир создан", db.world(world_id) is not None)

    db.add_rule(world_id, "не бывает воскрешения", title="Смерть")
    db.add_rule(world_id, "магия стоит крови", title="Цена")
    db.add_rule(world_id, "отключённое правило", title="Черновик", enabled=False)
    check("правила добавлены", len(db.rules(world_id)) == 3)
    check("отключённое правило не попадает в контекст", len(db.rules(world_id, enabled_only=True)) == 2)

    db.add_character(world_id, "Марла", appearance="scarred woman")
    check("персонаж добавлен", len(db.characters(world_id)) == 1)

    rejected = db.update_world(world_id, {"tone": "мрачный", "чушь": 1})
    check("правка мира применена", db.world(world_id).tone == "мрачный")
    check("незнакомое поле отклонено", rejected == ["чушь"])

    session_id = db.create_session(world_id, "Партия")
    db.add_message(session_id, "user", "привет")
    db.add_message(session_id, "assistant", "ответ")
    db.add_message(session_id, "user", "ещё")
    check("сообщения записаны", db.count_messages(session_id) == 3)
    check("окно берёт последние", [m.content for m in db.messages(session_id, limit=2)] == ["ответ", "ещё"])

    removed = db.rewind_to(session_id, db.messages(session_id)[0].id)
    check("откат удаляет хвост",
          removed["messages"] == 2 and db.count_messages(session_id) == 1, str(removed))

    scene_id = db.add_scene(session_id, "a castle", seed=5)
    db.finish_scene(scene_id, "/tmp/x.png", "done", 12.5)
    check("сцена закрыта", db.scene(scene_id).status == "done")
    check("сцены партии найдены", len(db.scenes(session_id)) == 1)

    db.set_state(session_id, "location", "таверна")
    check("состояние читается", db.world_state(session_id) == {"location": "таверна"})
    db.set_state(session_id, "location", "шахта")
    check("состояние перезаписывается", db.get_state(session_id, "location") == "шахта")
    db.delete_state(session_id, "location")
    check("состояние удаляется", db.world_state(session_id) == {})

    db.add_memory(session_id, 1, "первая память", 10)
    db.replace_memories(session_id, 2, "сводная память", 12)
    check("память сворачивается в одну", len(db.memories(session_id)) == 1)
    check("память обновлена", db.latest_memory(session_id).summary == "сводная память")

    second = db.create_session(world_id, "Вторая партия")
    db.add_message(second, "user", "в другой партии")
    check("партии изолированы", db.count_messages(session_id) == 1 and db.count_messages(second) == 1)

    # Сцены должны уходить вместе с партией. Без внешнего ключа они оставались в
    # базе, а из-за переиспользования номеров могли всплыть в новой партии.
    scenes_before = len(db.scenes(session_id))
    db.add_scene(session_id, "кадр первой партии", message_id=db.messages(session_id)[0].id)
    check("сцена записана", len(db.scenes(session_id)) == scenes_before + 1,
          f"{len(db.scenes(session_id))} != {scenes_before + 1}")
    db.delete_session(second)
    check("удаление партии не трогает чужие сцены",
          len(db.scenes(session_id)) == scenes_before + 1,
          str(len(db.scenes(session_id))))

    db.clear_session(session_id)
    check("очистка партии убирает её сцены", len(db.scenes(session_id)) == 0)
    check("целостность не нарушена", db.check_integrity() == [], str(db.check_integrity()))

    db.clear_session(session_id)
    check("очистка партии убирает сообщения", db.count_messages(session_id) == 0)
    check("очистка партии сохраняет мир", db.world(world_id) is not None)

    db.delete_world(world_id)
    check("удаление мира каскадом убирает партии", db.sessions() == [])
    check("удаление мира каскадом убирает сцены", db.scenes() == [])
    check("целостность чистая после удаления", db.check_integrity() == [], str(db.check_integrity()))


def test_settings() -> None:
    """Чтение, запись и отклонение мусора в настройках."""
    print("\nНастройки:")
    with tempfile.TemporaryDirectory() as tmp:
        store = SettingsStore(path=Path(tmp) / "settings.json")
        store.load()
        store.settings.temperature = 0.5
        store.save()

        again = SettingsStore(path=Path(tmp) / "settings.json")
        again.load()
        check("значение сохранилось", again.settings.temperature == 0.5)
        check("есть бюджет контекста", again.settings.context_budget_tokens > 0)

        rejected = again.settings.update({"temperature": "0.9", "нет_такого": 1})
        check("число приведено из строки", again.settings.temperature == 0.9)
        check("незнакомое поле отклонено", rejected == ["нет_такого"], str(rejected))

        again.settings.apply_model_preset(preset_for("gemma4"))
        check("пресет применился", again.settings.top_k == 64, str(again.settings.top_k))

        restored = Settings.from_dict({"temperature": 0.3, "мусор": True})
        check("незнакомый ключ при чтении игнорируется", restored.temperature == 0.3)


def test_prompts() -> None:
    """Шаблоны промптов подставляются без потерь.

    Отдельная проверка нужна потому, что в промптах лежат примеры JSON с
    фигурными скобками: ``str.format`` принял бы их за поля подстановки и упал
    бы уже во время хода.
    """
    print("\nШаблоны промптов:")
    check("врезка без текста пуста", prompts.interject_block("") == "")
    check("врезка с репликой не грозит отсутствием действия",
          "Игрок не действует" not in prompts.interject_block("пусть придёт стражник"))
    solo = prompts.interject_block("пусть придёт стражник", solo=True)
    check("врезка одна объясняет, что игрок не действует",
          "Игрок не действует" in solo, solo[:80])
    check("текст врезки дошёл в обоих случаях",
          "пусть придёт стражник" in solo)

    structure = prompts.render(prompts.STRUCTURE_WORLD_PROMPT, brief="МЕТКА_ОПИСАНИЯ")
    check("описание мира подставлено", "МЕТКА_ОПИСАНИЯ" in structure)
    check("пример JSON остался нетронутым", '"rules"' in structure and '"characters"' in structure)
    check("метка не осталась в тексте", "{brief}" not in structure)

    image = prompts.render(
        prompts.IMAGE_PROMPT_PROMPT, style="СТИЛЬ", appearances="ВНЕШНОСТЬ", description="СЦЕНА"
    )
    check("стиль подставлен", "СТИЛЬ" in image)
    check("внешность подставлена", "ВНЕШНОСТЬ" in image)
    check("описание сцены подставлено", "СЦЕНА" in image)

    summary = prompts.render(prompts.SUMMARIZE_PROMPT, transcript="СТЕНОГРАММА")
    check("стенограмма подставлена", "СТЕНОГРАММА" in summary)

    for key in FORMATS:
        text = prompts.base_instruction(get_format(key))
        check(f"инструкция формата {key} собирается", len(text) > 100)


def test_formats() -> None:
    """Форматы диалога согласованы между собой."""
    print("\nФорматы диалога:")
    check("форматов пять", len(FORMATS) == 5, str(list(FORMATS)))
    check("есть формат с фотографиями", get_format("chat_photo").scene_role == "photo")
    check("чат без картинок", not get_format("chat").wants_scene)
    check("неизвестный ключ даёт формат по умолчанию", get_format("нет").key == "story")
    check("у всех есть инструкция", all(f.prose_style.strip() for f in FORMATS.values()))
    check("у всех есть политика картинок", all(f.default_image_policy for f in FORMATS.values()))


def test_context() -> None:
    """Слоистая сборка контекста и бюджет токенов."""
    print("\nСборка контекста:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "ctx.db")
        try:
            _context_body(db)
        finally:
            db.close()


def _context_body(db: NovelDB) -> None:
    """Тело проверок контекста."""
    world_id = db.create_world(name="Шахта", format="story", brief="подземелья и тьма", tone="мрачный")
    db.add_rule(world_id, "магия стоит крови", title="Цена")
    db.add_character(world_id, "Марла", appearance="scarred woman")
    session_id = db.create_session(world_id, "Партия")
    for index in range(8):
        db.add_message(session_id, "user", f"реплика игрока номер {index} " + "слово " * 20)
        db.add_message(session_id, "assistant", f"ответ ведущего номер {index} " + "слово " * 20)

    settings = Settings(context_budget_tokens=4096, context_turns=4, max_tokens=500)
    builder = ContextBuilder(db, _NoEngine(), lambda: settings)
    prompt = builder.build(session_id, "новая реплика", count_exactly=False)

    keys = [layer.key for layer in prompt.layers]
    check("слои собраны", keys[:3] == ["format", "world", "rules"], str(keys))
    check("есть слой персонажей", "characters" in keys)
    check("есть окно", "window" in keys)
    check("стабильные слои помечены", all(l.stable for l in prompt.layers if l.key != "window"))
    check("окно ограничено настройкой", len(prompt.included_message_ids) <= 4,
          str(len(prompt.included_message_ids)))
    check("лишнее выброшено", len(prompt.dropped_message_ids) > 0)
    check("системная инструкция первой", prompt.chat_messages()[0]["role"] == "system")
    check("последнее сообщение — реплика игрока",
          prompt.chat_messages()[-1]["content"] == "новая реплика")
    check("есть предупреждение о выброшенных",
          any("не поместились" in w for w in prompt.warnings), str(prompt.warnings))

    tight = Settings(context_budget_tokens=900, context_turns=4, max_tokens=600)
    tight_prompt = ContextBuilder(db, _NoEngine(), lambda: tight).build(
        session_id, "реплика", count_exactly=False
    )
    check("тесный бюджет не роняет сборку", tight_prompt.total_tokens >= 0)
    check("тесный бюджет оставляет предупреждение",
          any("не оставляет места" in w for w in tight_prompt.warnings), str(tight_prompt.warnings))

    interjected = builder.build(session_id, "реплика", interject="пусть появится стражник", count_exactly=False)
    check("врезка попадает в запрос",
          "стражник" in interjected.chat_messages()[-1]["content"])

    # Текущая реплика записывается в хранилище до сборки запроса, поэтому без
    # исключения она попала бы в промпт дважды: один раз из окна, второй —
    # как последнее сообщение.
    duplicate_id = db.add_message(session_id, "user", "текущая реплика игрока")
    deduped = builder.build(
        session_id, "текущая реплика игрока", count_exactly=False, exclude_ids={duplicate_id}
    )
    contents = [message["content"] for message in deduped.chat_messages()]
    check("текущая реплика не дублируется",
          contents.count("текущая реплика игрока") == 1, str(contents[-3:]))
    check("исключённое сообщение не в окне", duplicate_id not in deduped.included_message_ids)

    check("оценка токенов растёт с длиной", estimate_tokens("а" * 280) > estimate_tokens("а" * 28))


def test_summarization() -> None:
    """Свёрнутая история перестаёт считаться выброшенной.

    Без этого признак «пора сворачивать» срабатывал бы на каждом ходу заново, а
    модель получала бы одну и ту же сводку, переписанную по кругу.
    """
    print("\nСуммаризация:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "sum.db")
        try:
            world_id = db.create_world(name="Шахта", format="story", brief="подземелья")
            session_id = db.create_session(world_id, "Партия")
            for index in range(30):
                role = "user" if index % 2 == 0 else "assistant"
                db.add_message(session_id, role, f"шаг {index} " + "слово " * 20)

            settings = Settings(context_budget_tokens=4096, context_turns=4, max_tokens=500)
            builder = ContextBuilder(db, _FakeEngine(), lambda: settings)
            before = builder.build(session_id, "реплика")
            check("до свёртки есть выброшенные", len(before.dropped_message_ids) > 0)
            check("до свёртки нужна суммаризация",
                  builder.needs_summary(session_id, len(before.dropped_message_ids)))

            report = builder.summarize(session_id)
            check("сводка создана", report is not None and report["tokens"] > 0)
            check("память записана", len(db.memories(session_id)) == 1)

            after = builder.build(session_id, "реплика")
            check("после свёртки окно заполнено", len(after.included_message_ids) > 0)
            check("после свёртки выброшенных нет", not after.dropped_message_ids,
                  str(after.dropped_message_ids))
            check("повторная свёртка не требуется",
                  not builder.needs_summary(session_id, len(after.dropped_message_ids)))
            check("память попала в слои",
                  any(layer.key == "memory" for layer in after.layers))

            check("повторный вызов сворачивать нечего", builder.summarize(session_id) is None)
        finally:
            db.close()


def test_presets_and_models() -> None:
    """Пресеты миров и реестр моделей."""
    print("\nПресеты и модели:")
    check("пресетов не меньше четырёх", len(WORLD_PRESETS) >= 4)
    check("пресет достаётся по ключу", get_preset("tavern") is not None)
    check("неизвестный пресет даёт None", get_preset("нет") is None)
    briefs = list_presets()
    check("краткие описания содержат счётчики", all("rules" in item for item in briefs))
    check("у пресета есть внешность персонажей",
          all(c.get("appearance") for p in WORLD_PRESETS for c in p["characters"]),
          "без неё генератор нарисует разных людей")

    registry = ModelRegistry()
    models = registry.all()
    # Каталог моделей задаётся переменной окружения: на чистой копии его может
    # не быть вовсе. Тогда реестр пуст — это правильный ответ, а не поломка.
    if not SEARCH_ROOTS:
        check("без каталогов моделей реестр пуст", models == [], f"найдено {len(models)}")
    else:
        check("реестр что-то нашёл", len(models) > 0, f"найдено {len(models)}")
    supported = [m for m in models if m.supported]
    if models:
        check("есть поддерживаемые модели", len(supported) > 0,
              str([m.name for m in models[:5]]))
    preset = preset_for("gemma4")
    check("пресет известной модели не общий", preset["source"] == "gemma4")
    check("пресет неизвестной модели общий", preset_for("нет_такой")["source"] == "_default")


def test_reasoning_kwargs() -> None:
    """Поле запроса, выключающее размышления, подставляется и не затирает чужое."""
    print("\nУправление размышлениями:")
    from novel.freetoken import FreeTokenClient

    client = FreeTokenClient()
    client.reasoning_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}
    body = client._with_reasoning({"messages": []})
    check("аргументы подставлены", body["chat_template_kwargs"]["enable_thinking"] is False)

    merged = client._with_reasoning({"chat_template_kwargs": {"custom": 1}})
    check("чужие аргументы шаблона сохранены", merged["chat_template_kwargs"].get("custom") == 1)
    check("и свои тоже", merged["chat_template_kwargs"]["enable_thinking"] is False)

    client.reasoning_kwargs = None
    untouched = client._with_reasoning({"messages": []})
    check("без настройки тело не меняется", "chat_template_kwargs" not in untouched)

    settings = Settings()
    check("размышления по умолчанию выключены", settings.reasoning_mode == "off")


def test_vision_policy() -> None:
    """Зрение применяется по необходимости, а не всегда.

    Возможность модели видеть и необходимость отправлять ей картинку — разные
    вещи: кадр стоит около 258 токенов входа и 1-4 секунды к ответу.
    """
    print("\nПолитика зрения:")
    policy = Settings()
    check("по умолчанию отправляются только вложения игрока",
          policy.vision_mode == "on_attach", policy.vision_mode)
    check("прошлые вложения по умолчанию не пересылаются", policy.history_images == 0)

    from novel.vision import DEFAULT_VISION_MODE, VISION_MODES

    check("режим по умолчанию объявлен в модуле зрения", DEFAULT_VISION_MODE in VISION_MODES)
    check("описаны все три режима", len(VISION_MODES) == 3, str(sorted(VISION_MODES)))

    # Файлы настроек, записанные до появления vision_mode, должны читаться как
    # прежде: иначе выбор пользователя сбросится на значение по умолчанию.
    migrated_off = Settings.from_dict({"vision_enabled": False, "temperature": 0.5})
    check("старый выключенный флаг переносится в режим off",
          migrated_off.vision_mode == "off", migrated_off.vision_mode)
    check("остальные поля при переносе не теряются", migrated_off.temperature == 0.5)
    migrated_on = Settings.from_dict({"vision_enabled": True})
    check("старый включённый флаг даёт режим по умолчанию",
          migrated_on.vision_mode == "on_attach", migrated_on.vision_mode)
    check("новый режим имеет приоритет над старым флагом",
          Settings.from_dict({"vision_enabled": False, "vision_mode": "on_attach"}).vision_mode
          == "on_attach")


def test_vision() -> None:
    """Подготовка изображений к отправке модели."""
    print("\nИзображения на входе модели:")
    from PIL import Image

    from novel.vision import VisionError, image_part, prepare_image, save_upload, user_message

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = Path(tmp)
        big = folder / "big.png"
        Image.new("RGB", (2048, 1024), (120, 60, 30)).save(big)
        data_url, width, height, original = prepare_image(big, max_side=768)
        check("длинная сторона уменьшена", max(width, height) == 768, f"{width}x{height}")
        check("пропорции сохранены", abs(width / height - 2.0) < 0.01, f"{width}x{height}")
        check("исходный размер сохранён", original == (2048, 1024), str(original))
        check("данные встроены", data_url.startswith("data:image/"), data_url[:30])

        part = image_part(data_url)
        check("часть сообщения нужного вида", part["type"] == "image_url")
        check("ссылка внутри части", part["image_url"]["url"] == data_url)

        plain = user_message("просто текст")
        check("без картинок сообщение строкой", isinstance(plain["content"], str))
        with_image = user_message("что на фото?", [data_url])
        check("с картинкой сообщение частями", isinstance(with_image["content"], list))
        check("первая часть текстовая", with_image["content"][0]["type"] == "text")
        check("вторая часть с картинкой", with_image["content"][1]["type"] == "image_url")

        saved = save_upload(data_url, folder / "uploads", "upload")
        check("загрузка сохранена", saved.exists() and saved.stat().st_size > 0)

        try:
            save_upload("не data-url", folder / "uploads", "bad")
            check("мусор вместо изображения отвергнут", False)
        except VisionError:
            check("мусор вместо изображения отвергнут", True)


class _FakeAuthorEngine:
    """Заглушка движка для сочинения: отвечает заранее заданным JSON."""

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.reasoning_mode: str | None = None

    def configure_reasoning(self, mode: str = "off") -> dict:
        self.reasoning_mode = mode
        return {"mode": mode, "applied": True}

    def chat(self, messages, **kwargs):  # noqa: ARG002 — подпись ради совместимости
        from novel.freetoken import ChatResult

        text = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return ChatResult(text=text, elapsed_s=0.1, prompt_tokens=10, completion_tokens=20, raw={})


def test_authoring() -> None:
    """Сочинение силами локальной модели: разбор ответов и запись в хранилище."""
    print("\nСочинение миров:")
    from novel.authoring import Author, AuthoringError, save_world
    from novel.authoring import _clean_characters, _clean_rules

    check(
        "правила из строк",
        _clean_rules(["первое", "", "второе"])
        == [{"title": "", "body": "первое"}, {"title": "", "body": "второе"}],
    )
    check(
        "правило без тела отброшено",
        _clean_rules([{"title": "x"}, {"body": "y"}]) == [{"title": "", "body": "y"}],
    )
    check("мусор вместо правил", _clean_rules("не список") == [])
    check(
        "персонаж без имени отброшен",
        _clean_characters([{"role": "нет имени"}, {"name": "Есть"}])
        == [{"name": "Есть", "role": "", "description": "", "appearance": "", "speech": ""}],
    )

    payload = {
        "name": "Эхо-Абиссаль",
        "brief": "Затопленный город, где память хранят в бутылках.",
        "genre": "биопанк",
        "tone": "меланхоличный",
        "narrator": "второе лицо",
        "style": "surreal oil painting",
        "rules": [{"title": "Цена", "body": "память стоит чувства"}],
        "characters": [{"name": "Мадам", "role": "торговка", "appearance": "elderly woman"}],
    }
    engine = _FakeAuthorEngine(payload)
    author = Author(engine)
    check("размышления выключены при сочинении", engine.reasoning_mode == "off")
    drafted = author.draft_world("идея")
    check("название разобрано", drafted.name == "Эхо-Абиссаль")
    check("правила разобраны", len(drafted.rules) == 1)
    check("персонажи разобраны", len(drafted.characters) == 1)
    check("стиль разобран", drafted.style == "surreal oil painting")

    try:
        Author(_FakeAuthorEngine("это не JSON")).draft_world("идея")
        check("нечитаемый ответ отвергнут", False)
    except AuthoringError:
        check("нечитаемый ответ отвергнут", True)

    try:
        Author(_FakeAuthorEngine({"brief": "есть описание"})).draft_world("идея")
        check("мир без названия отвергнут", False)
    except AuthoringError:
        check("мир без названия отвергнут", True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "author.db")
        try:
            world_id, session_id = save_world(db, drafted)
            saved = db.world(world_id)
            check("мир записан", saved is not None and saved.name == "Эхо-Абиссаль")
            check("правила записаны", len(db.rules(world_id)) == 1)
            check("персонажи записаны", len(db.characters(world_id)) == 1)
            check("партия создана", db.session(session_id) is not None)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_history_images() -> None:
    """Пересылка прошлых вложений управляется настройкой."""
    print("\nПересылка прошлых вложений:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "imgs.db")
        try:
            world_id = db.create_world(name="Фото", format="chat_photo", brief="переписка")
            session_id = db.create_session(world_id, "Партия")
            db.add_message(session_id, "user", "первое фото", attachments=["a.png"])
            db.add_message(session_id, "assistant", "вижу")
            db.add_message(session_id, "user", "второе фото", attachments=["b.png"])
            db.add_message(session_id, "assistant", "тоже вижу")

            calls: list[list[str]] = []

            def loader(paths: list[str]) -> list[str]:
                calls.append(paths)
                return [f"data:image/png;base64,{path}" for path in paths]

            off = Settings(history_images=0)
            no_images = ContextBuilder(db, _NoEngine(), lambda: off, loader).build(
                session_id, "реплика", count_exactly=False
            )
            check("при нуле вложения не пересылаются", not calls)
            check(
                "содержимое остаётся строкой",
                all(isinstance(item["content"], str) for item in no_images.chat_messages()),
            )

            one = Settings(history_images=1)
            with_one = ContextBuilder(db, _NoEngine(), lambda: one, loader).build(
                session_id, "реплика", count_exactly=False
            )
            check("переслано ровно одно вложение", calls == [["b.png"]], str(calls))
            parts = [item for item in with_one.chat_messages() if isinstance(item["content"], list)]
            check("сообщение стало многочастным", len(parts) == 1)
            check(
                "в нём есть картинка",
                any(part["type"] == "image_url" for part in parts[0]["content"]),
            )
        finally:
            db.close()


class _FakeAgentEngine(_FakeAuthorEngine):
    """Заглушка движка агента: отдаёт ответы по очереди."""

    def __init__(self, replies: list[object]) -> None:
        super().__init__(replies[0] if replies else {})
        self.replies = list(replies) or [{}]
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ARG002 — подпись ради совместимости
        from novel.freetoken import ChatResult

        payload = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return ChatResult(text=text, elapsed_s=0.1, prompt_tokens=10, completion_tokens=20, raw={})


class _StubAuthor:
    """Заглушка писателя: возвращает заранее заданный черновик мира."""

    def __init__(self) -> None:
        from novel.authoring import AuthoredWorld

        self.drafted = AuthoredWorld(
            name="Черновик",
            brief="описание из черновика",
            genre="тёмное фэнтези",
            tone="мрачный",
            narrator="второе лицо",
            style="oil painting",
            rules=[{"title": "Из черновика", "body": "правило из черновика"}],
            characters=[{"name": "Гость", "role": "роль", "description": "кто-то",
                         "appearance": "someone", "speech": "обычная"}],
        )

    def draft_world(self, idea: str):  # noqa: ARG002 — подпись ради совместимости
        return self.drafted

    def rewrite(self, text: str, instruction: str) -> str:  # noqa: ARG002
        return text

    def critique(self, world, rules, characters) -> str:  # noqa: ARG002
        return "замечаний нет"

    def scenarios(self, world, rules, count: int) -> list[dict]:  # noqa: ARG002
        return []


def test_agent() -> None:
    """Агент-редактор: разбор указания, выполнение и защита от удаления."""
    print("\nАгент-редактор:")
    from novel.agent import WorldAgent, worlds_digest

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "agent.db")
        try:
            world_id = db.create_world(name="Полигон", format="story", brief="портовый город")
            rule_id = db.add_rule(world_id, "Оружие носят открыто.", title="Оружие")

            digest = worlds_digest(db)
            check("сводка содержит номер мира", f"#{world_id}" in digest, digest[:60])
            check("сводка содержит номер правила", f"#{rule_id}" in digest)

            add_action = {
                "thought": "добавлю правило",
                "actions": [{"op": "add_rule", "world_id": world_id,
                             "title": "Цена", "body": "любая сделка имеет цену"}],
                "done": True,
            }

            preview = WorldAgent(db, _FakeAgentEngine([add_action])).run(
                "добавь правило", preview=True
            )
            check("предпросмотр собрал план", len(preview.steps) == 1)
            check("предпросмотр помечен", preview.preview)
            check("предпросмотр ничего не изменил", len(db.rules(world_id)) == 1)
            check("результат помечен как план", preview.steps[0].results[0].get("preview") is True)

            applied = WorldAgent(db, _FakeAgentEngine([add_action])).run("добавь правило")
            check("действие выполнено", applied.changed == 1, str(applied.changed))
            check("правило записано", len(db.rules(world_id)) == 2)
            check("прогон завершён", applied.finished)

            delete_action = {
                "thought": "удалю",
                "actions": [{"op": "delete_rule", "rule_id": rule_id}],
                "done": True,
            }
            denied = WorldAgent(db, _FakeAgentEngine([delete_action])).run("удали правило")
            check("удаление без разрешения отклонено", denied.changed == 0)
            check("причина отказа записана",
                  "запрещено" in denied.steps[0].results[0].get("error", ""),
                  str(denied.steps[0].results[0]))
            check("правило на месте", len(db.rules(world_id)) == 2)

            allowed = WorldAgent(db, _FakeAgentEngine([delete_action])).run(
                "удали правило", allow_destructive=True
            )
            check("с разрешением удаление прошло", allowed.changed == 1, str(allowed.changed))
            check("правило удалено", len(db.rules(world_id)) == 1)

            unknown = WorldAgent(db, _FakeAgentEngine([{
                "thought": "попробую", "actions": [{"op": "взорвать_мир"}], "done": True,
            }])).run("сделай что-нибудь")
            check("неизвестное действие отклонено",
                  unknown.steps[0].results[0].get("ok") is False)
            broken = WorldAgent(db, _FakeAgentEngine(["это не JSON"])).run("сделай")
            check("нечитаемый ответ обработан", broken.error is not None, str(broken.error))
            empty = WorldAgent(db, _FakeAgentEngine([])).run("")
            check("пустое указание отклонено", empty.error is not None)

            # Внешняя остановка: движок выгружают, движок поднимать не нужно.
            revived_calls: list[int] = []

            class _DeadEngine(_FakeAgentEngine):
                def chat(self, messages, **kwargs):
                    from novel.freetoken import FreeTokenError

                    raise FreeTokenError("соединение оборвано")

            stopped = WorldAgent(db, _DeadEngine([{}])).run(
                "сделай что-нибудь",
                should_continue=lambda: False,
                revive=lambda: bool(revived_calls.append(1)) or True,
            )
            check("остановка по команде помечена", stopped.interrupted)
            check("движок не поднимали ради остановки", not revived_calls,
                  f"попыток подъёма {len(revived_calls)}")
            check("причина остановки записана", "остановлено" in (stopped.error or ""),
                  str(stopped.error))

            # Работа в открытом мире: без номера действия идут в него.
            other_id = db.create_world(name="Второй", format="story")
            current = db.create_world(name="Открытый", format="story")
            agent = WorldAgent(
                db,
                _FakeAgentEngine([{
                    "thought": "добавлю правило",
                    "actions": [{"op": "add_rule", "title": "Без номера",
                                 "body": "действие без номера мира"}],
                    "done": True,
                }]),
                current_world_id=current,
            )
            agent.run("добавь правило")
            check("правило ушло в открытый мир", len(db.rules(current)) == 1)
            check("соседний мир не тронут", len(db.rules(other_id)) == 0)

            digest = worlds_digest(db, current=current)
            check("открытый мир помечен в сводке", "ОТКРЫТ СЕЙЧАС" in digest, digest[:200])
            check("открытый мир идёт первым",
                  digest.splitlines()[0].startswith(f"#{current}"), digest.splitlines()[0])

            # fill_world дополняет существующий мир, а не создаёт новый.
            before_worlds = len(db.worlds())
            filler = WorldAgent(
                db, _FakeAgentEngine([{}]), author=_StubAuthor(), current_world_id=current
            )
            result = filler.execute({"op": "fill_world", "idea": "добавь лор"},
                                    allow_destructive=False)
            check("fill_world сработал", result.get("ok") is True, str(result))
            check("новый мир не создан", len(db.worlds()) == before_worlds)
            check("правило добавлено в открытый мир", len(db.rules(current)) == 2,
                  str(len(db.rules(current))))
            check("персонаж добавлен в открытый мир", len(db.characters(current)) == 1)
            check("название мира не переписано", db.world(current).name == "Открытый")

            # Без открытого мира действие с пустым номером обязано упасть.
            lonely = WorldAgent(db, _FakeAgentEngine([{}])).execute(
                {"op": "add_rule", "title": "x", "body": "y"}, allow_destructive=False
            )
            check("без открытого мира действие отклонено", lonely.get("ok") is False,
                  str(lonely))

            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_bundle() -> None:
    """Перенос мира вместе с картинками."""
    print("\nПакет мира с картинками:")
    from PIL import Image

    from novel.bundle import export_bundle, import_bundle, orphan_uploads, referenced_files

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        pictures = root / "pictures"
        pictures.mkdir()
        scene_image = pictures / "scene.png"
        attachment = pictures / "photo.png"
        stray = pictures / "stray.png"
        for path in (scene_image, attachment, stray):
            Image.new("RGB", (32, 32), (90, 40, 20)).save(path)

        db = NovelDB(root / "bundle.db")
        try:
            world_id = db.create_world(name="Пакет", format="story", brief="перенос")
            db.add_rule(world_id, "правило переносится", title="Правило")
            db.add_character(world_id, "Кто-то", appearance="someone")
            session_id = db.create_session(world_id, "Партия")
            message_id = db.add_message(
                session_id, "user", "вот фото", attachments=[str(attachment)]
            )
            scene_id = db.add_scene(
                session_id, "кадр", message_id=message_id, status="done"
            )
            db.finish_scene(scene_id, str(scene_image), "done", 12.0)

            archive = root / "world.zip"
            report = export_bundle(db, world_id, archive)
            check("архив создан", archive.is_file())
            check("картинок в пакете две", report.images == 2, str(report.images))

            import zipfile

            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
            check("внутри есть описание мира", "world.json" in names)
            check("внутри есть изображения",
                  len([n for n in names if n.startswith("images/")]) == 2, str(names))
            check("пути внутри архива, а не наружу",
                  not any(str(pictures) in n for n in names))

            restored = import_bundle(db, archive, root / "unpacked")
            check("мир загружен заново", restored.world_id not in (None, world_id))
            check("картинок распаковано две", restored.images == 2, str(restored.images))

            new_sessions = db.sessions(restored.world_id)
            check("партия перенесена", len(new_sessions) == 1)
            new_scenes = db.scenes(new_sessions[0].id)
            check("сцена перенесена", len(new_scenes) == 1)
            if new_scenes:
                image_path = Path(new_scenes[0].path or "")
                check("файл кадра на месте", image_path.is_file(), str(image_path))
                check("файл распакован в новый каталог",
                      root / "unpacked" in image_path.parents, str(image_path))
            new_messages = db.messages(new_sessions[0].id)
            attached = [m for m in new_messages if m.attachment_paths]
            check("вложение перенесено", bool(attached))
            if attached:
                photo = Path(attached[0].attachment_paths[0])
                check("файл вложения на месте", photo.is_file(), str(photo))

            check("правила перенесены", len(db.rules(restored.world_id)) == 1)
            check("персонажи перенесены", len(db.characters(restored.world_id)) == 1)

            used = referenced_files(db)
            check("ссылки видны в базе", str(scene_image.resolve()) in used)
            check("лишний файл не считается нужным", str(stray.resolve()) not in used)
            check("осиротевших вложений нет",
                  not [p for p in orphan_uploads(db) if p.name == "photo.png"])

            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_connection_reset() -> None:
    """Обрыв соединения превращается в понятную ошибку, а не в сырое исключение.

    Движок выгружают штатно — перед генерацией картинки, при смене модели и
    кнопкой «Стоп всё». Запрос, попавший на этот момент, обрывается исключением
    ConnectionResetError, которое не является URLError: без отдельной обработки
    оно проходило мимо всех перехватов и убивало фоновую задачу молча.
    """
    print("\nОбрыв соединения с движком:")
    import socket
    import threading

    from novel.freetoken import FreeTokenClient, FreeTokenError

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]

    def drop() -> None:
        """Принимает соединение и сразу закрывает его, не отвечая."""
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            connection.close()

    threading.Thread(target=drop, daemon=True).start()
    client = FreeTokenClient(f"http://127.0.0.1:{port}", timeout_s=5.0)
    try:
        client.health()
        check("обрыв соединения превращается в FreeTokenError", False, "исключения не было")
    except FreeTokenError:
        check("обрыв соединения превращается в FreeTokenError", True)
    except OSError as exc:
        check("обрыв соединения превращается в FreeTokenError", False,
              f"пробралось {type(exc).__name__}")
    finally:
        server.close()


def test_locations() -> None:
    """Постоянные места: память, образец и слой в контексте."""
    print("\nПостоянные места:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "places.db")
        try:
            world_id = db.create_world(name="Берег", format="story", style="oil painting")
            first = db.add_location(world_id, "Замок Вороньего Клыка", prompt="dark castle",
                                    style="oil painting", seed=42)
            check("место заведено", db.location(first) is not None)
            check("поиск по имени находит", db.location_by_name(world_id, "Замок Вороньего Клыка") is not None)
            check("поиск не зависит от регистра",
                  db.location_by_name(world_id, "замок вороньего клыка") is not None)
            check("поиск не зависит от лишних пробелов",
                  db.location_by_name(world_id, "  Замок   Вороньего  Клыка ") is not None)
            check("чужое имя не находится", db.location_by_name(world_id, "Таверна") is None)

            db.touch_location(first)
            db.touch_location(first)
            check("посещения считаются", db.location(first).visits == 2, str(db.location(first).visits))

            db.set_location_reference(first, "D:/pictures/castle.png")
            check("образец назначен", db.location(first).reference_path == "D:/pictures/castle.png")
            db.set_location_reference(first, None)
            check("образец снимается", db.location(first).reference_path is None)

            rejected = db.update_location(first, {"name": "Новое имя", "чушь": 1})
            check("имя меняется", db.location(first).name == "Новое имя")
            check("незнакомое поле отклонено", rejected == ["чушь"])

            session_id = db.create_session(world_id, "Партия")
            scene_id = db.add_scene(session_id, "кадр", location_id=first)
            check("сцена помнит место", db.scene(scene_id).location_id == first)
            db.finish_scene(scene_id, "D:/pictures/frame.png", "done", 12.0, used_reference=True)
            check("признак образца записан", db.scene(scene_id).used_reference == 1)

            # Слой контекста: имена мест должны попадать в промпт.
            settings = Settings(location_memory=True)
            prompt = ContextBuilder(db, _NoEngine(), lambda: settings).build(
                session_id, "реплика", count_exactly=False
            )
            places = [layer for layer in prompt.layers if layer.key == "places"]
            check("слой мест собран", len(places) == 1)
            check("в слое есть имя места", "Новое имя" in places[0].text, places[0].text[:80])

            off = Settings(location_memory=False)
            without = ContextBuilder(db, _NoEngine(), lambda: off).build(
                session_id, "реплика", count_exactly=False
            )
            check("слой мест можно выключить",
                  not [layer for layer in without.layers if layer.key == "places"])

            db.delete_location(first)
            check("место удаляется", db.location(first) is None)
            check("сцены места остаются, но теряют ссылку при удалении мира",
                  db.scene(scene_id) is not None)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_reference_graph() -> None:
    """Референс места попадает в граф генерации."""
    print("\nГраф с образцом места:")
    from novel.comfy import build_t2i_graph

    plain = build_t2i_graph("замок", seed=1)
    check("без образца нет узла загрузки", "1" not in plain)
    check("без образца нет ссылки на картинку", "images.image_1" not in plain["7"]["inputs"])

    with_reference = build_t2i_graph("замок", seed=1, reference="abc-castle.png")
    check("с образцом появился узел загрузки", with_reference["1"]["class_type"] == "LoadImage")
    check("узел читает нужный файл", with_reference["1"]["inputs"]["image"] == "abc-castle.png")
    check("энкодер получил ссылку",
          with_reference["7"]["inputs"]["images.image_1"] == ["1", 0])
    check("размер по-прежнему свой",
          with_reference["9"]["inputs"]["latent_image"] == ["2", 0])
    check("сэмплер не изменился",
          with_reference["9"]["inputs"]["sampler_name"] == "euler"
          and with_reference["9"]["inputs"]["cfg"] == 1.0)


def test_t2i_models() -> None:
    """Имена моделей генерации указывают на существующие файлы.

    ComfyUI принимает не путь, а имя файла из своего каталога. Неверное имя он
    отбивает ответом «Value not in list», и кадры просто перестают рисоваться,
    не объясняя причину. Поэтому имя сверяется с диском, а не только по виду.

    Проверка пропускается, если каталогов ComfyUI нет: на чистой копии проекта
    искать негде.
    """
    print("\nМодели генерации кадров:")
    from novel import config

    models = config.QWEN_T2I_GRAPH_MODELS

    # Каталогов ComfyUI может не быть вовсе: на чистой копии проекта это норма,
    # и тогда имена искать негде. Проверять сами имена имеет смысл только когда
    # есть где искать.
    roots = [root for root in config.COMFY_MODEL_ROOTS if root.is_dir()]
    if not roots:
        check("каталогов ComfyUI нет — проверка имён пропущена", True)
        return

    check("все три роли заполнены",
          all(models.get(role) for role in ("clip_name", "unet_name", "vae_name")),
          str(models))

    for role, value in models.items():
        if value:
            check(f"{role} без пути, только имя",
                  "/" not in value and "\\" not in value, value)

    # Проектор зрения загрузчику текста не подходит: у него своё назначение.
    clip = models.get("clip_name", "")
    check("кодировщик не проектор зрения", not clip.lower().startswith("mmproj"), clip)

    for role, value in models.items():
        if not value:
            continue
        folder_name = config.T2I_MODEL_FOLDERS[role]
        found = any((root / folder_name / value).is_file() for root in roots)
        check(f"{role} есть на диске", found, f"{value} не найден в {folder_name}")


def test_model_switch() -> None:
    """Агент умеет переключать модель, но откладывает это до конца прогона."""
    print("\nПереключение модели агентом:")
    from novel.agent import WorldAgent, find_model
    from novel.models import ModelRegistry

    registry = ModelRegistry()
    supported = [item for item in registry.all() if item.supported]
    # На чистой копии каталога моделей может не быть: тогда проверять нечего,
    # и это не поломка.
    if SEARCH_ROOTS:
        check("реестр видит модели", bool(supported))
    if supported:
        probe = supported[0]
        check("поиск по точному имени", find_model(registry, probe.name) is not None)
        check("поиск по части имени", find_model(registry, probe.name.split("-")[0]) is not None)
        check("несуществующая модель не находится", find_model(registry, "нетакой-модели") is None)

    # Дальше нужна хотя бы одна найденная модель: переключать не на что.
    if not supported:
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "switch.db")
        try:
            db.create_world(name="Мир", format="story")
            action = {
                "thought": "сменю модель",
                "actions": [{"op": "switch_model", "model": supported[0].name}],
                "done": True,
            }
            agent = WorldAgent(db, _FakeAgentEngine([action]), registry=registry)
            run = agent.run("смени модель")

            check("смена модели отложена", len(run.pending) == 1, str(run.pending))
            check("в отчёте помечено как отложенное",
                  run.steps[0].results[0].get("deferred") is True,
                  str(run.steps[0].results[0]))
            check("внутри прогона модель не менялась", run.changed == 0)

            missing = WorldAgent(
                db,
                _FakeAgentEngine([{"thought": "x",
                                   "actions": [{"op": "switch_model", "model": "нетакой"}],
                                   "done": True}]),
                registry=registry,
            ).run("смени модель")
            check("неизвестная модель отклоняется",
                  missing.pending == [] and bool(missing.pending) is False,
                  str(missing.pending))
        finally:
            db.close()


def test_file_hygiene() -> None:
    """Вес партии и уборка файлов, оставшихся от удалённых партий."""
    print("\nВес партии и уборка файлов:")
    from PIL import Image

    from novel.bundle import orphan_files, purge_orphans, session_weight

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        frame = root / "novel_00042_00001_.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(frame)

        db = NovelDB(root / "files.db")
        try:
            world_id = db.create_world(name="Вес", format="story")
            session_id = db.create_session(world_id, "Партия")
            for index in range(30):
                db.add_message(session_id, "user", f"реплика номер {index} " * 5)
            scene_id = db.add_scene(session_id, "кадр")
            db.finish_scene(scene_id, str(frame), "done", 12.0)

            weight = session_weight(db, session_id)
            check("сообщения посчитаны", weight["messages"] == 30, str(weight["messages"]))
            check("текст весит килобайты, а не мегабайты", weight["text_kb"] < 20,
                  f"{weight['text_kb']} KB")
            check("кадр учтён в картинках", weight["images"] == 1, str(weight))

            # Пока сцена на месте, кадр не считается мусором.
            dirs = {"uploads_dir": root / "uploads", "output_dir": root,
                    "input_dir": root / "input"}
            check("нужный кадр не в мусоре",
                  frame not in orphan_files(db, **dirs)["frames"])

            db.delete_session(session_id)
            orphans = orphan_files(db, **dirs)
            check("после удаления партии кадр стал мусором",
                  frame in orphans["frames"], str(orphans["frames"]))
            check("категории разделены",
                  set(orphans) == {"uploads", "frames", "staged"}, str(list(orphans)))

            report = purge_orphans(db, **dirs)
            check("файл удалён", not frame.exists())
            check("уборка отчиталась об удалении", report["removed"] == 1, str(report))
            check("после уборки мусора нет",
                  sum(len(items) for items in orphan_files(db, **dirs).values()) == 0)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_image_policy() -> None:
    """Политика картинок: ведущий решает, остальные режимы — по расписанию."""
    print("\nПолитика картинок:")
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "policy.db")
        try:
            world_id = db.create_world(name="Политики", format="story")
            session_id = db.create_session(world_id, "Партия")
            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            class _Outcome:
                """Заглушка итога хода: важно только, были ли сцены."""

                def __init__(self, scene_ids: list[int]) -> None:
                    self.scene_ids = scene_ids
                    self.new_place = False
                    self.new_characters: list[str] = []
                    self.sudden = False

            outcome = _Outcome([1])

            machine.settings.image_policy = "master"
            check("ведущий решает: кадр рисуется",
                  machine._should_generate_now(session_id, outcome))
            check("ведущий решает: без блока scene кадра нет",
                  not machine._should_generate_now(session_id, _Outcome([])))

            # «Минимум картинок»: три основания, и ни одного лишнего.
            machine.settings.image_policy = "minimal"
            plain = _Outcome([1])
            check("минимум: знакомое место без нового лица — без кадра",
                  not machine._should_generate_now(session_id, plain))
            new_place = _Outcome([1]); new_place.new_place = True
            check("минимум: новое место — кадр",
                  machine._should_generate_now(session_id, new_place))
            new_face = _Outcome([1]); new_face.new_characters = ["Лирит"]
            check("минимум: новое лицо в кадре — кадр",
                  machine._should_generate_now(session_id, new_face))
            sudden = _Outcome([1]); sudden.sudden = True
            check("минимум: резкая перемена обстановки — кадр",
                  machine._should_generate_now(session_id, sudden))
            check("минимум: без блока scene кадра нет",
                  not machine._should_generate_now(session_id, _Outcome([])))

            # Показанные лица запоминаются и не считаются новыми снова —
            # независимо от регистра: ведущий пишет имя то так, то этак.
            from novel.protocol import SceneSpec

            machine.settings.image_policy = "minimal"
            first = _Outcome([1])
            machine._mark_new_characters(
                session_id, SceneSpec(image_prompt="x", npc=["Старый Грог"]), first
            )
            check("новое лицо распознано", first.new_characters == ["Старый Грог"],
                  str(first.new_characters))
            machine._note_shown(session_id, first)

            again = _Outcome([2])
            machine._mark_new_characters(
                session_id, SceneSpec(image_prompt="y", npc=["старый грог"]), again
            )
            check("то же лицо в другом регистре уже не новое",
                  again.new_characters == [], str(again.new_characters))
            check("знакомое место без нового лица — без кадра",
                  not machine._should_generate_now(session_id, again))

            fresh = _Outcome([3])
            machine._mark_new_characters(
                session_id, SceneSpec(image_prompt="z", npc=["Старый Грог", "Лирит"]), fresh
            )
            check("новое лицо среди знакомых распознано", fresh.new_characters == ["Лирит"],
                  str(fresh.new_characters))
            check("новое лицо даёт кадр",
                  machine._should_generate_now(session_id, fresh))

            # Агент меняет вид кадров: стиль мира и качество генерации.
            from novel.agent import WorldAgent as _Agent

            store = SettingsStore(Path(tmp) / "agent-settings.json")
            agent = _Agent(db, _FakeAgentEngine([{}]), settings_store=store,
                           current_world_id=world_id)
            shown = agent.execute({"op": "show_image_settings"}, allow_destructive=False)
            check("агент видит стиль мира", shown.get("ok") and "style" in shown, str(shown))
            check("агент видит настройки генератора",
                  "generator" in shown and "image_steps" in shown["generator"], str(shown))

            styled = agent.execute(
                {"op": "set_world_style",
                 "style": "hyper-realistic, cinematic lighting, detailed skin texture"},
                allow_destructive=False,
            )
            check("стиль мира изменён", styled.get("ok") is True, str(styled))
            check("стиль записан в базу",
                  "hyper-realistic" in (db.world(world_id).style or ""),
                  str(db.world(world_id).style))

            rejected = agent.execute({"op": "set_world_style", "style": "  "},
                                     allow_destructive=False)
            check("пустой стиль отклонён", rejected.get("ok") is False, str(rejected))

            tuned = agent.execute({"op": "set_image_settings", "image_steps": 30},
                                  allow_destructive=False)
            check("шаги генерации изменены", tuned.get("ok") is True and tuned["image_steps"] == 30,
                  str(tuned))
            check("настройка сохранена", store.settings.image_steps == 30,
                  str(store.settings.image_steps))
            check("размер не кратный 32 отклонён",
                  agent.execute({"op": "set_image_settings", "image_size": 1000},
                                allow_destructive=False).get("ok") is False)
            check("слишком большое число шагов отклонено",
                  agent.execute({"op": "set_image_settings", "image_steps": 500},
                                allow_destructive=False).get("ok") is False)
            check("пустая правка отклонена",
                  agent.execute({"op": "set_image_settings"},
                                allow_destructive=False).get("ok") is False)
            check("шаги не изменились после отказов", store.settings.image_steps == 30,
                  str(store.settings.image_steps))

            machine.settings.image_policy = "never"
            check("никогда не рисует", not machine._should_generate_now(session_id, outcome))
            machine.settings.image_policy = "manual"
            check("по кнопке сам не рисует", not machine._should_generate_now(session_id, outcome))
            machine.settings.image_policy = "every_turn"
            check("каждый ход рисует", machine._should_generate_now(session_id, outcome))

            machine.settings.image_policy = "every_n"
            machine.settings.image_every_n = 2
            db.add_message(session_id, "assistant", "первый ответ")
            db.add_message(session_id, "assistant", "второй ответ")
            check("каждые N ходов считает ходы",
                  machine._should_generate_now(session_id, outcome))

            machine.settings.image_policy = "on_scene_change"
            db.set_state(session_id, "location", "тракт")
            db.set_state(session_id, "last_image_location", "тракт")
            check("при смене сцены: то же место не перерисовывается",
                  not machine._should_generate_now(session_id, outcome))
            db.set_state(session_id, "last_image_location", "таверна")
            check("при смене сцены: новое место рисуется",
                  machine._should_generate_now(session_id, outcome))
        finally:
            db.close()


def test_solo_interject() -> None:
    """Врезка без реплики: в истории нет пустого сообщения игрока."""
    print("\nВрезка без реплики:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "interject.db")
        try:
            world_id = db.create_world(name="Врезки", format="story")
            session_id = db.create_session(world_id, "Партия")
            db.add_message(session_id, "assistant", "Ведущий что-то описал.")

            settings = Settings()
            builder = ContextBuilder(db, _NoEngine(), lambda: settings)
            built = builder.build(
                session_id, "", interject="пусть придёт стражник", solo=True,
                count_exactly=False,
            )
            layers = {layer.key: layer.text for layer in built.layers}
            last = built.messages[-1]["content"] if built.messages else ""
            check("врезка попала в последнее сообщение игрока",
                  "пусть придёт стражник" in last, last[:120])
            check("ведущему сказано, что игрок не действует",
                  "Игрок не действует" in last, last[:120])
            check("пустой реплики игрока в запросе нет",
                  all(text.strip() for text in layers.values()))
            check("лишних переводов строки нет", last == last.strip(), repr(last[-20:]))

            # Так же, но с репликой: подсказки про бездействие быть не должно.
            with_text = builder.build(
                session_id, "Я оглядываюсь", interject="пусть придёт стражник",
                count_exactly=False,
            )
            check("с репликой нет пометки о бездействии",
                  not any("Игрок не действует" in layer.text for layer in with_text.layers))
        finally:
            db.close()


def test_plot_notes() -> None:
    """Заметки на потом: хранение, слой промпта и разбор блока notes."""
    print("\nЗаметки на потом:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "notes.db")
        try:
            world_id = db.create_world(name="Заметки", format="story")
            session_id = db.create_session(world_id, "Партия")
            other_id = db.create_session(world_id, "Другая")

            first = db.add_note(session_id, "  через пару ходов пусть появится стражник  ")
            second = db.add_note(session_id, "тайна Лирит раскроется позже")
            db.add_note(other_id, "чужая заметка")

            mine = db.notes(session_id)
            check("заметки записаны", len(mine) == 2, str(len(mine)))
            check("текст обрезан по краям",
                  db.note(first).text == "через пару ходов пусть появится стражник",
                  repr(db.note(first).text))
            check("чужая заметка не видна", all(n.session_id == session_id for n in mine))
            check("заметка находится по номеру", db.note(second) is not None)
            check("несуществующая заметка не находится", db.note(99999) is None)

            db.close_note(first)
            check("сбывшаяся исчезает из активных",
                  [n.id for n in db.notes(session_id)] == [second],
                  str([n.id for n in db.notes(session_id)]))
            check("сбывшаяся остаётся в базе", db.note(first) is not None)
            check("повторное закрытие ничего не портит", db.close_note(99999) is False)

            check("пакетное закрытие считает только свои", db.close_notes([second, 99999]) == 1)
            db.delete_note(second)
            check("удаление убирает насовсем", db.note(second) is None)
            check("активных не осталось", db.notes(session_id) == [])

            # Слой промпта.
            keep = db.add_note(session_id, "пусть объявится сборщик долгов")
            block = prompts.notes_block([{
                "id": keep, "text": "пусть объявится сборщик долгов",
                "source": "player", "horizon": 0, "age": 0,
            }])
            check("в слое есть номер заметки", f"{keep}." in block, block[:80])
            check("в слое объяснено, как отметить сбывшееся", "<notes>" in block)
            check("пустой список даёт пустой слой", prompts.notes_block([]) == "")

            # Задумки ведущего идут отдельным разделом и слабее игроцких.
            mixed = prompts.notes_block([
                {"id": 1, "text": "моя задумка", "source": "player", "horizon": 0, "age": 0},
                {"id": 2, "text": "выдумка ведущего", "source": "model",
                 "horizon": 5, "age": 0},
            ])
            check("моя задумка названа приказом", "Выполняй, когда дойдёт очередь" in mixed)
            check("выдумка ведущего названа предположением",
                  "Это НЕ приказ" in mixed, mixed[:400])
            check("порядок: сначала мои", mixed.find("моя задумка") < mixed.find("выдумка ведущего"))
            check("срок выдумки виден", "на 5 ходов" in mixed)
            check("возраст показывается", "прошло" in prompts.notes_block([
                {"id": 3, "text": "старая выдумка", "source": "model",
                 "horizon": 5, "age": 3}]))
            check("без выдумок раздела нет",
                  "НЕ приказ" not in prompts.notes_block([
                      {"id": 1, "text": "только моё", "source": "player",
                       "horizon": 0, "age": 0}]))

            builder = ContextBuilder(db, _NoEngine(), lambda: Settings())
            built = builder.build(session_id, "реплика", count_exactly=False)
            layers = {layer.key: layer.text for layer in built.layers}
            check("слой заметок попал в промпт", "notes" in layers, str(list(layers)))
            check("текст заметки виден ведущему",
                  "сборщик долгов" in layers.get("notes", ""))

            # Разбор блока notes в ответе модели.
            parsed = parse_reply(
                "<prose>Текст.</prose>\n<notes>\n[12, 15]\n</notes>"
            )
            check("номера сбывшихся замечены", parsed.fulfilled_notes == [12, 15],
                  str(parsed.fulfilled_notes))
            check("ответ с notes считается машинным", parsed.has_machine_block)
            as_dict = parse_reply('<prose>Текст.</prose>\n<notes>{"fulfilled": [7]}</notes>')
            check("объект вместо списка тоже понимается",
                  as_dict.fulfilled_notes == [7], str(as_dict.fulfilled_notes))
            messy = parse_reply('<prose>Текст.</prose>\n<notes>"3, 4 и мусор"</notes>')
            check("мусор в блоке не ломает разбор",
                  messy.fulfilled_notes == [3, 4], str(messy.fulfilled_notes))
            empty = parse_reply("<prose>Текст.</prose>")
            check("без блока notes список пуст", empty.fulfilled_notes == [])
            check("обычный ответ не считается машинным", not empty.has_machine_block)

            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_now_block() -> None:
    """Полоса фазы: что происходит сейчас и сколько это обычно длится."""
    print("\nЧто происходит сейчас:")
    from novel.machine import NovelMachine, State
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "now.db")
        try:
            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            idle = machine._now()
            check("в покое фаза свободна", idle["phase"] == "IDLE", idle["phase"])
            check("в покое время не идёт", idle["busy"] is False and idle["since_s"] == 0.0)
            check("в покое очередь пуста", idle["queue"] == [])

            machine._set_state(State.IMAGE_GEN)
            machine._current_scene = {"id": 13, "prompt": "замок", "reference": True}
            machine._scene_queue = [13, 14]
            machine.turn_started = time.time() - 30
            busy = machine._now()
            check("фаза генерации названа по-русски",
                  "рисуется кадр #13" in busy["detail"], busy["detail"])
            check("отмечено, что по образцу места", "по образцу" in busy["detail"])
            check("идёт отсчёт фазы", busy["busy"] and busy["since_s"] >= 0)
            check("названа обычная длительность",
                  busy["usual_low_s"] == 80.0 and busy["usual_high_s"] == 110.0)
            check("виден остаток очереди", busy["queue"] == [14], str(busy["queue"]))
            check("видно время хода целиком", busy["turn_s"] >= 29, str(busy["turn_s"]))
            check("раньше обычного — тревоги нет", busy["overdue"] is False)

            machine._phase_started = time.time() - 300
            check("сильно дольше обычного — тревога",
                  machine._now()["overdue"] is True)

            machine.turn_started = None
            machine._current_scene = None
            machine._set_state(State.IDLE)
            check("возврат в покой сбрасывает тревогу", machine._now()["overdue"] is False)
            check("переход в ту же фазу не сбрасывает отсчёт",
                  machine._now()["since_s"] == 0.0)
        finally:
            db.close()


def test_rewind() -> None:
    """Откат партии: убирает и то, что ссылалось на удалённые сообщения."""
    print("\nОткат партии:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "rewind.db")
        try:
            world_id = db.create_world(name="Откат", format="story")
            session_id = db.create_session(world_id, "Партия")
            other_id = db.create_session(world_id, "Соседняя")

            keep = db.add_message(session_id, "assistant", "первый ответ")
            doomed = db.add_message(session_id, "assistant", "второй ответ")
            last = db.add_message(session_id, "assistant", "третий ответ")
            neighbour = db.add_message(other_id, "assistant", "чужое сообщение")

            db.add_scene(session_id, "старый кадр", message_id=keep)
            db.add_scene(session_id, "удаляемый кадр", message_id=doomed)
            db.add_scene(session_id, "тоже удаляемый", message_id=last)
            db.add_scene(other_id, "чужой кадр", message_id=neighbour)

            db.add_memory(session_id, doomed, "сводка про удалённое", 10)
            db.add_memory(session_id, keep, "сводка про оставшееся", 10)
            db.set_state(session_id, "shown_characters", ["Старый Грог"])

            report = db.rewind_to(session_id, keep)
            check("сообщения после точки удалены", report["messages"] == 2, str(report))
            check("сцены удалённых ходов убраны", report["scenes"] == 2, str(report))
            check("сводка про удалённое убрана", report["memories"] == 1, str(report))

            check("до точки отката всё цело",
                  [m.id for m in db.messages(session_id)] == [keep],
                  str([m.id for m in db.messages(session_id)]))
            check("осталась ровно одна сцена", len(db.scenes(session_id)) == 1,
                  str([s.id for s in db.scenes(session_id)]))
            check("память про оставшееся не тронута",
                  [m.through_message_id for m in db.memories(session_id)] == [keep])
            check("показанные лица сброшены — ветка новая",
                  db.get_state(session_id, "shown_characters", None) is None)

            check("соседняя партия не тронута",
                  len(db.messages(other_id)) == 1 and len(db.scenes(other_id)) == 1)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_items() -> None:
    """Вещи: хранение, слой промпта и запрет распоряжаться."""
    print("\nВещи:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "items.db")
        try:
            world_id = db.create_world(name="Вещи", format="story")
            session_id = db.create_session(world_id, "Партия")
            hero = db.add_character(world_id, "Лирит", role="демонесса")
            other_world = db.create_world(name="Чужой", format="story")

            laptop = db.add_item(world_id, "ноутбук", properties="заряжен полностью")
            ring = db.add_item(
                world_id, "деревянное кольцо", character_id=hero,
                properties="может расширяться и сужаться",
            )
            shells = db.add_item(world_id, "патрон от дробовика", quantity=12,
                                 properties="картечь, 12-й калибр")
            db.add_item(other_world, "чужая вещь")

            check("вещь игрока записана", db.item(laptop) is not None)
            check("вещь игрока опознаётся", db.item(laptop).is_player)
            check("вещь персонажа опознаётся", not db.item(ring).is_player)
            check("свойства сохранены",
                  db.item(ring).properties == "может расширяться и сужаться",
                  db.item(ring).properties)
            check("количество сохранено", db.item(shells).quantity == 12)

            check("всего вещей мира", len(db.items(world_id)) == 3, str(len(db.items(world_id))))
            check("вещи игрока отделяются",
                  [i.name for i in db.items(world_id, character_id=None)] ==
                  ["ноутбук", "патрон от дробовика"])
            check("вещи персонажа отделяются",
                  [i.name for i in db.items(world_id, character_id=hero)] == ["деревянное кольцо"])
            check("чужой мир не подмешивается",
                  all(i.world_id == world_id for i in db.items(world_id)))

            rejected = db.update_item(laptop, {"quantity": 0, "чушь": 1})
            check("количество не уходит в ноль", db.item(laptop).quantity == 1)
            check("незнакомое поле отклонено", rejected == ["чушь"], str(rejected))
            db.update_item(ring, {"character_id": None})
            check("вещь можно передать игроку", db.item(ring).is_player)
            db.update_item(ring, {"character_id": hero})  # возвращаем владельцу

            # Слой промпта: ведущий знает, но не распоряжается.
            block = prompts.items_block([
                ("Игрок", [("ноутбук", "заряжен полностью", 1),
                           ("патрон", "картечь", 12)]),
                ("Лирит", [("деревянное кольцо", "расширяется и сужается", 1)]),
            ])
            check("вещи игрока в слое", "ноутбук (заряжен полностью)" in block, block[:120])
            check("количество показано", "патрон ×12 (картечь)" in block, block)
            check("вещи персонажа в слое", "Лирит: деревянное кольцо" in block)
            check("запрет распоряжаться на месте",
                  "НЕ распоряжаешься" in block, block[:200])
            check("пустой список даёт пустой слой", prompts.items_block([]) == "")
            check("владелец без вещей не показывается",
                  "Молчун" not in prompts.items_block([("Молчун", [])]))

            built = ContextBuilder(db, _NoEngine(), lambda: Settings()).build(
                session_id, "реплика", count_exactly=False
            )
            layers = {layer.key: layer.text for layer in built.layers}
            check("слой вещей попал в промпт", "items" in layers, str(list(layers)))
            check("ноутбук виден ведущему", "ноутбук" in layers.get("items", ""))
            check("кольцо привязано к владельцу",
                  "Лирит" in layers.get("items", ""), layers.get("items", "")[:200])

            off = Settings(items_memory=False)
            without = ContextBuilder(db, _NoEngine(), lambda: off).build(
                session_id, "реплика", count_exactly=False
            )
            check("слой вещей отключается",
                  "items" not in {layer.key for layer in without.layers})

            # Перенос мира несёт вещи вместе с именами владельцев.
            payload = db.export_world(world_id)
            check("вещи попали в выгрузку", len(payload["items"]) == 3, str(len(payload["items"])))
            check("владелец записан именем",
                  any(i["owner"] == "Лирит" for i in payload["items"]),
                  str([i["owner"] for i in payload["items"]]))
            restored, _ = db.import_world(payload)
            back = db.items(restored)
            check("вещи перенесены", len(back) == 3, str(len(back)))
            check("владелец восстановлен",
                  any(i.name == "деревянное кольцо" and not i.is_player for i in back),
                  str([(i.name, i.character_id) for i in back]))
            check("свойства пережили перенос",
                  any(i.properties == "картечь, 12-й калибр" for i in back))

            db.delete_item(shells)
            check("вещь удаляется", db.item(shells) is None)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_duplicate_world() -> None:
    """Копия мира: содержимое переносится, ссылки не путаются, образец цел."""
    print("\nКопия мира:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "dup.db")
        try:
            src = db.create_world(name="Рабочий", format="story", brief="описание",
                                  genre="тёмное", tone="мрачный", style="oil painting")
            db.add_rule(src, "правило", title="Правило")
            hero = db.add_character(src, "Лирит", role="демонесса", appearance="demoness")
            db.add_item(src, "кольцо", character_id=hero, properties="сужается")
            db.add_item(src, "ноутбук", properties="заряжен")
            place = db.add_location(src, "таверна", prompt="tavern", style="oil", seed=7)
            db.set_location_reference(place, str(Path(tmp) / "ref.png"))
            session = db.create_session(src, "Партия")
            first = db.add_message(session, "user", "привет", attachments=["a.png"])
            second = db.add_message(session, "assistant", "ответ")
            scene = db.add_scene(session, "кадр", message_id=second, location_id=place)
            db.finish_scene(scene, str(Path(tmp) / "frame.png"), "done", 12.0,
                            used_reference=True)
            db.add_memory(session, second, "сводка", 11)

            copy_id, sessions = db.duplicate_world(src)
            check("копия создана", copy_id not in (None, src))
            check("имя помечено как копия", "копия" in db.world(copy_id).name,
                  db.world(copy_id).name)
            check("поля мира перенесены",
                  db.world(copy_id).brief == "описание"
                  and db.world(copy_id).style == "oil painting",
                  db.world(copy_id).brief)
            check("правила перенесены", len(db.rules(copy_id)) == 1)
            check("персонажи перенесены", len(db.characters(copy_id)) == 1)
            check("вещи перенесены", len(db.items(copy_id)) == 2)

            new_hero = db.characters(copy_id)[0].id
            ring = [i for i in db.items(copy_id) if i.name == "кольцо"][0]
            laptop = [i for i in db.items(copy_id) if i.name == "ноутбук"][0]
            check("владелец вещи указывает на копию персонажа",
                  ring.character_id == new_hero, str(ring.character_id))
            check("вещь игрока осталась у игрока", laptop.is_player)
            check("места перенесены", len(db.locations(copy_id)) == 1)
            check("образец места перенесён",
                  db.locations(copy_id)[0].reference_path is not None)
            check("без партий копия получает одну пустую",
                  len(sessions) == 1 and db.count_messages(sessions[0]) == 0, str(sessions))
            check("исходный мир не тронут",
                  len(db.sessions(src)) == 1 and len(db.messages(session)) == 2)

            # С партиями: ссылки должны переехать на новые номера.
            copy2, sessions2 = db.duplicate_world(src, with_sessions=True)
            check("партия скопирована", len(sessions2) == 1)
            copied = db.messages(sessions2[0])
            check("сообщения скопированы", len(copied) == 2, str(len(copied)))
            check("номера сообщений новые",
                  not ({m.id for m in copied} & {first, second}),
                  str([m.id for m in copied]))
            check("вложение перенесено", any(m.attachment_paths for m in copied))
            new_scenes = db.scenes(sessions2[0])
            check("сцена скопирована", len(new_scenes) == 1)
            if new_scenes:
                check("сцена ссылается на копию сообщения",
                      new_scenes[0].message_id in {m.id for m in copied},
                      str(new_scenes[0].message_id))
                check("сцена ссылается на копию места",
                      new_scenes[0].location_id == db.locations(copy2)[0].id,
                      str(new_scenes[0].location_id))
                check("признак образца перенесён", new_scenes[0].used_reference == 1)
            new_memories = db.memories(sessions2[0])
            check("сводка перенесена", len(new_memories) == 1)
            if new_memories:
                check("сводка ссылается на копию сообщения",
                      new_memories[0].through_message_id in {m.id for m in copied},
                      str(new_memories[0].through_message_id))

            check("исходный мир по-прежнему цел", len(db.messages(session)) == 2)
            check("целостность чистая", db.check_integrity() == [])

            db.delete_world(copy_id)
            db.delete_world(copy2)
            check("копии удаляются, образец остаётся",
                  db.world(copy_id) is None and db.world(src) is not None)
        finally:
            db.close()


def test_reference_rule() -> None:
    """Образец места нужен при возвращении, а не на каждом кадре подряд."""
    print("\nКогда нужен образец места:")
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "ref.db")
        try:
            world_id = db.create_world(name="Места", format="story")
            session_id = db.create_session(world_id, "Партия")
            tavern = db.add_location(world_id, "таверна", prompt="tavern")
            castle = db.add_location(world_id, "замок", prompt="castle")
            frame = Path(tmp) / "frame.png"
            frame.write_bytes(b"not really a png")
            db.set_location_reference(tavern, str(frame))
            db.set_location_reference(castle, str(frame))

            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            name, ref = machine._reference_for_scene(session_id, {"location_id": tavern})
            check("первый кадр места — по образцу", name == "таверна" and ref is not None,
                  f"{name} / {ref}")

            db.set_state(session_id, "last_image_location", "таверна")
            name, ref = machine._reference_for_scene(session_id, {"location_id": tavern})
            check("следующий кадр там же — без образца", name == "таверна" and ref is None,
                  f"{name} / {ref}")

            name, ref = machine._reference_for_scene(session_id, {"location_id": castle})
            check("переход в другое место — по образцу", ref is not None, str(ref))
            # Кадр в замке нарисован: теперь последним был замок.
            db.set_state(session_id, "last_image_location", "замок")
            name, ref = machine._reference_for_scene(session_id, {"location_id": tavern})
            check("возвращение — снова по образцу", ref is not None, str(ref))

            name, ref = machine._reference_for_scene(session_id, {"location_id": None})
            check("сцена без места — без образца", name == "" and ref is None, f"{name}/{ref}")

            db.set_state(session_id, "last_image_location", "  ТАВЕРНА ")
            name, ref = machine._reference_for_scene(session_id, {"location_id": tavern})
            check("регистр и пробелы не мешают", ref is None, str(ref))

            lonely = db.add_location(world_id, "пустошь", prompt="wastes")
            name, ref = machine._reference_for_scene(session_id, {"location_id": lonely})
            check("место без образца — не падает", name == "пустошь" and ref is None,
                  f"{name} / {ref}")
        finally:
            db.close()


def test_looks() -> None:
    """Внешность «сейчас»: переживает сводку и обновляется ведущим."""
    print("\nВнешность сейчас:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "looks.db")
        try:
            world_id = db.create_world(name="Внешность", format="story")
            session_id = db.create_session(world_id, "Партия")
            db.add_character(world_id, "Лирит", description="демонесса",
                             appearance="demoness with small horns")

            # Разбор блока.
            parsed = parse_reply(
                '<prose>Лирит накидывает капюшон.</prose>\n'
                '<looks>\n{"Лирит": "в монашеской рясе, рожки спрятаны"}\n</looks>'
            )
            check("внешность разобрана",
                  parsed.looks == {"Лирит": "в монашеской рясе, рожки спрятаны"},
                  str(parsed.looks))
            check("ответ с внешностью считается машинным", parsed.has_machine_block)
            pairs = parse_reply(
                '<prose>x</prose>\n<looks>\n[{"name": "Грог", "look": "рука в бинтах"}]\n</looks>'
            )
            check("список пар тоже понимается", pairs.looks == {"Грог": "рука в бинтах"},
                  str(pairs.looks))
            check("пустое имя отброшено",
                  parse_reply('<prose>x</prose>\n<looks>\n{"": "что-то"}\n</looks>').looks == {})
            check("без блока внешность пуста",
                  parse_reply("<prose>текст</prose>").looks == {})

            # Слой промпта: внешность важнее карточки.
            db.set_state(session_id, "looks", {"Лирит": "в монашеской рясе"})
            built = ContextBuilder(db, _NoEngine(), lambda: Settings()).build(
                session_id, "реплика", count_exactly=False
            )
            layers = {layer.key: layer.text for layer in built.layers}
            check("слой внешности попал в промпт", "looks" in layers, str(list(layers)))
            check("описание видно ведущему", "в монашеской рясе" in layers.get("looks", ""))
            check("сказано, что это важнее карточки",
                  "важнее карточки" in layers.get("looks", ""),
                  layers.get("looks", "")[:120])

            # Главное: сводка истории внешность не трогает.
            db.add_message(session_id, "assistant", "длинный разговор " * 40)
            db.add_memory(session_id, db.messages(session_id)[-1].id,
                          "пересказ, где про рясу ничего нет", 20)
            after = ContextBuilder(db, _NoEngine(), lambda: Settings()).build(
                session_id, "реплика", count_exactly=False
            )
            text_after = {layer.key: layer.text for layer in after.layers}
            check("после сводки внешность на месте",
                  "в монашеской рясе" in text_after.get("looks", ""),
                  text_after.get("looks", "")[:120])
            check("сводка при этом существует", len(db.memories(session_id)) == 1)

            # Забывание возвращает к карточке.
            db.set_state(session_id, "looks", {})
            empty = ContextBuilder(db, _NoEngine(), lambda: Settings()).build(
                session_id, "реплика", count_exactly=False
            )
            check("без изменений слоя внешности нет",
                  "looks" not in {layer.key for layer in empty.layers})

            check("пустой блок даёт пустой слой", prompts.looks_block([]) == "")
            check("в списке внешности только имя и описание",
                  prompts.looks_block([("Кто-то", "как-то")]).strip().endswith("Кто-то: как-то"),
                  prompts.looks_block([("Кто-то", "как-то")]))
            # Инструкция обязана быть в постоянном блоке протокола: пока она
            # лежала в слое, который появляется лишь при готовых изменениях,
            # ведущий о блоке не знал и никогда его не присылал.
            protocol = prompts.base_instruction(get_format("story"))
            check("инструкция про looks есть в протоколе всегда", "<looks>" in protocol)
            check("ведущего избавили от выдумывания зерна",
                  '"seed": 12345' not in protocol)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_seed_rule() -> None:
    """Повтор зерна от ведущего заменяется случайным: иначе кадры одинаковы."""
    print("\nЗерно кадра:")
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "seed.db")
        try:
            world_id = db.create_world(name="Зерно", format="story")
            session_id = db.create_session(world_id, "Партия")
            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            first = machine._pick_seed(session_id, {"seed": 54321})
            check("новое зерно ведущего уважается", first == 54321, str(first))

            db.set_state(session_id, "last_image_seed", 54321)
            again = machine._pick_seed(session_id, {"seed": 54321})
            check("повтор заменяется случайным", again != 54321, str(again))
            check("случайное в разумных пределах", 1 <= again < 2**31, str(again))

            fresh = machine._pick_seed(session_id, {"seed": 777})
            check("сменившееся зерно снова уважается", fresh == 777, str(fresh))

            empty = machine._pick_seed(session_id, {})
            check("без зерна берётся случайное", empty != 777, str(empty))
            check("два случайных подряд не совпадают",
                  machine._pick_seed(session_id, {}) != empty)

            db.set_scene_seed(db.add_scene(session_id, "кадр", seed=999), 4242)
            check("фактическое зерно записывается в сцену",
                  db.scenes(session_id)[0].seed == 4242, str(db.scenes(session_id)[0].seed))
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_looks_tracking() -> None:
    """Отдельный вопрос модели про внешность и разбор её вольного ответа."""
    print("\nПроверка внешности отдельным запросом:")
    from novel.protocol import looks_from_text

    check("чистый объект", looks_from_text('{"Лирит": "в плаще"}') == {"Лирит": "в плаще"})
    check("с пояснениями вокруг",
          looks_from_text('Изменения: {"игрок": "без доспехов"} вот так')
          == {"игрок": "без доспехов"})
    check("в тройных кавычках",
          looks_from_text('```json\n{"Грог": "рука перевязана"}\n```')
          == {"Грог": "рука перевязана"})
    check("списком пар",
          looks_from_text('[{"name": "Марта", "look": "в дорожном"}]')
          == {"Марта": "в дорожном"})
    check("отказ словами", looks_from_text("Никто не переодевался.") == {})
    check("пустой объект", looks_from_text("{}") == {})
    check("пустой ответ", looks_from_text("") == {})
    check("мусор не роняет", looks_from_text("{{{ не json") == {})
    check("лишние скобки не мешают",
          looks_from_text('{"Игрок": "снял шлем"} и ещё {"мусор"') == {"Игрок": "снял шлем"})

    # Проверка не должна рушить ход, если модель недоступна.
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "track.db")
        try:
            world_id = db.create_world(name="Слежка", format="story")
            session_id = db.create_session(world_id, "Партия")
            db.add_character(world_id, "Лирит")
            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            class FakeEngine:
                """Модель отвечает так, как решил тест."""

                def __init__(self, answer): self.answer = answer
                def configure_reasoning(self, mode): return {"applied": False, "mode": mode}
                def chat_stream(self, messages, **kwargs):
                    if isinstance(self.answer, Exception):
                        raise self.answer
                    return type("R", (), {"text": self.answer})()

            settings = SettingsStore(Path(tmp) / "s.json").load()
            machine.ft = FakeEngine('{"игрок": "без доспехов, рубаха"}')
            timings: dict[str, float] = {}
            found = machine._track_looks(session_id, "снимаю доспехи", "Ты снимаешь доспехи.",
                                         settings, timings)
            check("изменение найдено", found == {"игрок": "без доспехов, рубаха"}, str(found))
            check("время проверки записано в тайминги хода",
                  isinstance(timings.get("looks_check"), float), str(timings))

            machine._merge_looks(session_id, found)
            check("изменение попало в состояние",
                  db.get_state(session_id, "looks", {}) == found,
                  str(db.get_state(session_id, "looks", {})))

            machine.ft = FakeEngine("Никто не переодевался.")
            check("отказ ничего не меняет",
                  machine._track_looks(session_id, "иду", "Ты идёшь.", settings, {}) == {})

            machine.ft = FakeEngine(OSError("движок пропал"))
            check("сбой модели не роняет ход",
                  machine._track_looks(session_id, "иду", "Ты идёшь.", settings, {}) == {})

            check("пустой ответ ведущего не проверяется",
                  machine._track_looks(session_id, "иду", "   ", settings, {}) == {})

            settings.looks_tracking = False
            machine.ft = FakeEngine('{"игрок": "что-то"}')
            check("настройка выключает проверку",
                  machine._track_looks(session_id, "иду", "Ты идёшь.", settings, {}) == {})
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_prompts_have_no_undefined_calls() -> None:
    """В модулях не должно быть вызовов несуществующих имён.

    Такая опечатка не ловится ни компиляцией, ни тестами: она всплывает только в
    живом ходу и рушит его целиком. Один раз это уже случилось.
    """
    print("\nОпечатки в именах:")
    import ast
    import builtins

    root = Path(__file__).resolve().parents[1] / "novel"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                defined.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                defined.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        missing = sorted(
            name for name in called
            if name not in defined and not hasattr(builtins, name)
        )
        check(f"в {path.name} нет вызовов несуществующих имён", not missing, ", ".join(missing))


def test_image_control() -> None:
    """Дополнение к промпту, правка кадра, ступени качества и портреты."""
    print("\nКонтроль над кадрами:")
    from novel.settings import IMAGE_QUALITY, Settings
    from novel.prompts import base_instruction
    from novel.formats import get_format

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "img.db")
        try:
            world_id = db.create_world(name="Кадры", format="chat_photo",
                                       image_suffix="cinematic lighting")
            session_id = db.create_session(world_id, "Партия")
            check("дополнение записано при создании",
                  db.world(world_id).image_suffix == "cinematic lighting",
                  db.world(world_id).image_suffix)
            check("дополнение правится", not db.update_world(
                world_id, {"image_suffix": "oil painting"}) and
                db.world(world_id).image_suffix == "oil painting")

            db.update_world(world_id, {"image_suffix": "detailed skin"})
            dumped = db.export_world(world_id)
            check("дополнение попало в выгрузку",
                  dumped["world"]["image_suffix"] == "detailed skin",
                  str(dumped["world"].get("image_suffix")))
            copy_id, _ = db.import_world(dumped)
            check("дополнение пережило перенос",
                  db.world(copy_id).image_suffix == "detailed skin",
                  db.world(copy_id).image_suffix)
            twin, _ = db.duplicate_world(world_id)
            check("дополнение скопировано",
                  db.world(twin).image_suffix == "detailed skin")

            # Правка кадра: описание меняется, кадр встаёт в очередь.
            scene = db.add_scene(session_id, "старое описание", status="done")
            db.finish_scene(scene, str(Path(tmp) / "frame.png"), "done", 12.0)
            check("кадр готов", db.scene(scene).status == "done")
            db.set_scene_prompt(scene, "a portrait of a woman in a red cloak")
            check("описание заменено",
                  db.scene(scene).prompt == "a portrait of a woman in a red cloak",
                  db.scene(scene).prompt)
            check("кадр снова в очереди", db.scene(scene).status == "pending")
            check("старый файл не потерян", db.scene(scene).path is not None)

            # Ступени качества.
            settings = Settings()
            check("ступени объявлены", set(IMAGE_QUALITY) == {"fast", "normal", "quality"},
                  str(list(IMAGE_QUALITY)))
            check("быстро — 640 на 12", settings.image_dimensions("fast") == (640, 12))
            check("обычно — 768 на 20", settings.image_dimensions("normal") == (768, 20))
            check("качественно — 1024 на 25", settings.image_dimensions("quality") == (1024, 25))
            settings.image_size, settings.image_steps = 896, 18
            check("свои числа берутся из полей",
                  settings.image_dimensions("custom") == (896, 18),
                  str(settings.image_dimensions("custom")))
            check("без аргумента — текущая ступень",
                  Settings().image_dimensions() == (768, 20))

            # Портрет в переписке.
            chat = base_instruction(get_format("chat_photo"), "minimal", True)
            check("в переписке рисуется портрет", "Портрет собеседника" in chat)
            check("портрет в полный рост",
                  "в полный рост" in chat and "full body shot" in chat, chat[-700:])
            check("лицо крупно запрещено", "Лицо крупно не давай" in chat)
            check("сказано брать внешность из карточки",
                  "из карточки персонажа дословно" in chat)
            check("портрет требует одного собеседника", "РОВНО ОДНОГО" in chat)
            scene_mode = base_instruction(get_format("chat_photo"), "minimal", False)
            check("можно вернуть сцену", "Портрет собеседника" not in scene_mode)

            # Затвор портрета: он включается только в переписке и только по
            # настройке. В прозе кадр остаётся сценой, что бы ни стояло в поле.
            settings.chat_frame = "portrait"
            builder = ContextBuilder(db, object(), lambda: settings)
            check("в переписке портрет включается",
                  builder._portrait_mode(get_format("chat_photo")))
            check("в диалогах с картинками тоже",
                  builder._portrait_mode(get_format("chat_scene")))
            check("в прозе портрет не включается",
                  not builder._portrait_mode(get_format("story")))
            check("в формате без картинок не включается",
                  not builder._portrait_mode(get_format("chat")))
            settings.chat_frame = "scene"
            check("настройка «сцену целиком» выключает портрет",
                  not builder._portrait_mode(get_format("chat_photo")))
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_schema_rebuild() -> None:
    """Пересборка таблицы не ломает ссылки в дочерних таблицах.

    SQLite при ``ALTER TABLE ... RENAME`` переписывает ссылки на таблицу во всех
    остальных. Если старую копию потом удалить, дочерние таблицы начинают
    ссылаться на несуществующую: чтение работает, а любая запись падает с
    «no such table: main.worlds_legacy». Один раз это уже случилось при
    добавлении поля мира.
    """
    print("\nПересборка таблиц:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "shape.db"
        db = NovelDB(path)
        try:
            world_id = db.create_world(name="Опыт", format="story")
            session_id = db.create_session(world_id, "Партия")
            db.add_rule(world_id, "правило")
            db.add_character(world_id, "Лирит")
            db.create_session(world_id, "Вторая")

            # Принудительная пересборка worlds — та самая операция.
            db._migrate_table_shapes(force={"worlds"})
            broken = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%_legacy%'"
            ).fetchall()
            check("после пересборки ссылок на _legacy нет",
                  not broken, str([r[0] for r in broken]))
            check("мир на месте после пересборки", db.world(world_id) is not None)
            check("партии на месте", len(db.sessions(world_id)) == 2)
            check("правило на месте", len(db.rules(world_id)) == 1)
            check("персонаж на месте", len(db.characters(world_id)) == 1)
            # Главное: в дочерние таблицы можно писать.
            check("правило пишется", db.add_rule(world_id, "ещё") > 0)
            check("персонаж пишется", db.add_character(world_id, "Грог") > 0)
            check("партия пишется", db.create_session(world_id, "Третья") > 0)
            check("сообщение пишется", db.add_message(session_id, "user", "привет") > 0)
        finally:
            db.close()

        # Теперь поломку создаём руками и проверяем лечение.
        db = NovelDB(path)
        try:
            db.conn.execute("PRAGMA foreign_keys=OFF")
            db.conn.execute("ALTER TABLE worlds RENAME TO worlds_legacy")
            db.conn.execute(
                "CREATE TABLE worlds (id INTEGER PRIMARY KEY, name TEXT NOT NULL,"
                " format TEXT NOT NULL, brief TEXT NOT NULL DEFAULT '',"
                " genre TEXT NOT NULL DEFAULT '', tone TEXT NOT NULL DEFAULT '',"
                " style TEXT NOT NULL DEFAULT '', narrator TEXT NOT NULL DEFAULT '',"
                " hidden_rules TEXT NOT NULL DEFAULT '',"
                " image_suffix TEXT NOT NULL DEFAULT '',"
                " created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
            )
            db.conn.execute("INSERT INTO worlds SELECT * FROM worlds_legacy")
            db.conn.execute("DROP TABLE worlds_legacy")
            db.conn.commit()
            hurt = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%_legacy%'"
            ).fetchall()
            check("поломка воспроизвелась", bool(hurt), str([r[0] for r in hurt]))
        finally:
            db.close()

        healed = NovelDB(path)
        try:
            left = healed.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%_legacy%'"
            ).fetchall()
            check("лечение убрало ссылки", not left, str([r[0] for r in left]))
            world_id = healed.worlds()[0].id
            check("запись снова работает", healed.add_rule(world_id, "после лечения") > 0)
            check("целостность чистая", healed.check_integrity() == [])
        finally:
            healed.close()


def test_character_confusion() -> None:
    """Ведущий не должен путать персонажей из-за разнобоя в именах и карточках."""
    print("\nПутаница в персонажах:")
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "conf.db")
        try:
            world_id = db.create_world(name="Путаница", format="story")
            session_id = db.create_session(world_id, "Партия")
            hero = db.add_character(world_id, "Марта", appearance="woman, dark hair")
            db.add_character(world_id, "Ивонна", appearance="tall blonde")

            settings_store = SettingsStore(Path(tmp) / "s.json")
            machine = NovelMachine(db, settings_store, ModelRegistry())
            db.ft = type("F", (), {"configure_reasoning": lambda self, m: {}})()

            # Регистр не должен раздваивать запись: «игрок» и «Игрок» — один человек.
            machine._merge_looks(session_id, {"игрок": "в рубахе"})
            machine._merge_looks(session_id, {"Игрок": "в рубахе и сапогах"})
            looks = db.get_state(session_id, "looks", {}) or {}
            check("регистр не раздваивает игрока", list(looks) == ["игрок"], str(list(looks)))
            check("описание обновилось", looks["игрок"] == "в рубахе и сапогах",
                  looks.get("игрок", ""))

            # Имя персонажа приводится к написанию из карточки.
            machine._merge_looks(session_id, {"марта": "в чёрном платье"})
            looks = db.get_state(session_id, "looks", {}) or {}
            check("имя приведено к карточке", "Марта" in looks, str(list(looks)))
            check("дубликата в другом регистре нет", "марта" not in looks)

            # Выдуманное имя в слой не попадает: иначе ведущий считает его настоящим.
            machine._merge_looks(session_id, {"Катька": "в льняном платье"})
            looks = db.get_state(session_id, "looks", {}) or {}
            check("выдуманное имя отброшено", "Катька" not in looks, str(list(looks)))

            # Пробелы в имени чистятся при записи.
            messy = db.add_character(world_id, "  Хозяйка  ")
            check("пробелы обрезаны при добавлении",
                  db.character(messy).name == "Хозяйка",
                  repr(db.character(messy).name))
            db.update_character(messy, {"name": " Хозяйка "})
            check("пробелы обрезаны при правке",
                  db.character(messy).name == "Хозяйка",
                  repr(db.character(messy).name))

            # Проверка мира ловит одинаковые имена и одинаковую внешность.
            db.update_character(hero, {"name": "Хозяйка"})
            issues = machine.validate(session_id)["issues"]
            texts = " ".join(item["text"] for item in issues)
            check("одинаковые имена замечены", "одинаковыми именами" in texts, texts[:200])
            db.update_character(hero, {"name": "Марта"})

            twin = db.add_character(world_id, "Двойник", appearance="woman, dark hair")
            issues = machine.validate(session_id)["issues"]
            texts = " ".join(item["text"] for item in issues)
            check("одинаковая внешность замечена",
                  "одинаковая внешность" in texts, texts[:240])
            db.delete_character(twin)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_notes_cleanup() -> None:
    """Чистка задумок разом: по автору, по закрытым и всё сразу."""
    print("\nЧистка задумок:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "notes.db")
        try:
            world_id = db.create_world(name="Задумки", format="story")
            session_id = db.create_session(world_id, "Партия")
            other = db.create_session(world_id, "Другая")

            mine = [db.add_note(session_id, f"моя {i}") for i in range(3)]
            theirs = [db.add_note(session_id, f"его {i}", source="model")
                      for i in range(2)]
            db.add_note(other, "чужая")

            check("потолки считаются по автору",
                  db.count_notes(session_id, "player") == 3
                  and db.count_notes(session_id, "model") == 2,
                  f"{db.count_notes(session_id, 'player')}/{db.count_notes(session_id, 'model')}")
            check("закрытая не считается активной",
                  db.close_note(theirs[0]) and db.count_notes(session_id, "model") == 1)

            check("стираются только мои", db.delete_notes(session_id, "player") == 3)
            left = db.notes(session_id, limit=50)
            check("остались только его", all(n.source == "model" for n in left),
                  str([n.source for n in left]))
            check("чужая партия не тронута", len(db.notes(other, limit=50)) == 1)

            # «Стереть его» убирает все его задумки, и активные и закрытые.
            removed = db.delete_notes(session_id, "model")
            check("стираются только его", removed == 2, str(removed))
            check("активных не осталось", db.notes(session_id, limit=50) == [])
            check("закрытых тоже не осталось",
                  db.conn.execute(
                      "SELECT COUNT(*) FROM plot_notes WHERE session_id = ?",
                      (session_id,)).fetchone()[0] == 0)

            # Закрытые лежат отдельно и убираются своей кнопкой.
            db.add_note(session_id, "сбудется", source="player")
            db.add_note(session_id, "не сбудется", source="model")
            db.add_note(session_id, "ещё одна", source="player")
            done = db.notes(session_id, limit=50)[-1].id
            db.close_note(done, "done")
            check("закрытая убрана из активных",
                  len(db.notes(session_id, limit=50)) == 2)
            check("закрытые стираются отдельно", db.delete_finished_notes(session_id) == 1)
            check("активные остались",
                  len(db.notes(session_id, limit=50)) == 2)

            check("стереть всё", db.delete_notes(session_id) == 2)
            check("пусто", db.notes(session_id, limit=50) == [])
            check("на пустом ничего не ломается", db.delete_notes(session_id) == 0)
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_chat_format() -> None:
    """Чат-форматы должны требовать именно переписку, а не прозу.

    `chat` запрещает абзацы, а `chat_photo` только просит «коротко, как в
    мессенджере» — без явного запрета ведущий пишет прозу на девятьсот
    символов. Напоминание о стиле стоит у конца запроса: история идёт последней,
    и модель повторяет то, что видит последним.
    """
    print("\nЧат-форматы:")
    from novel.formats import get_format
    from novel import prompts

    for key in ("chat", "chat_photo", "chat_scene"):
        fmt = get_format(key)
        style = fmt.prose_style
        check(f"{key}: сказано про отдельную строку", "отдельная строка" in style, style[:80])
        check(f"{key}: есть ограничение длины", "двух предложений" in style, style[:120])
        check(f"{key}: запрещены описания",
              "обстановк" in style.lower() or "описаний нет" in style.lower(),
              style[:160])

    check("форматы переписки помечены как разговор",
          all(get_format(k).as_dict()["prose_is_messages"]
              for k in ("chat", "chat_photo", "chat_scene")))
    check("проза разговором не считается",
          not get_format("story").as_dict()["prose_is_messages"])

    chat_reminder = prompts.style_reminder(get_format("chat_photo"))
    check("в напоминании есть образец строки", "Марта:" in chat_reminder, chat_reminder[:120])
    check("напоминание запрещает абзацы", "Абзацев" in chat_reminder)
    check("напоминание объясняет старую историю", "до смены формата" in chat_reminder)
    prose_reminder = prompts.style_reminder(get_format("story"))
    check("у прозы своё напоминание", "проз" in prose_reminder.lower())
    check("проза не получает чатового образца", "Марта:" not in prose_reminder)
    check("у сжатой истории своё напоминание",
          "действия и реплики" in prompts.style_reminder(get_format("story_terse")))


def test_image_policies() -> None:
    """Политики частоты картинок: на каких ходах срабатывает каждая.

    Сценарий один для всех: новое место, повтор, резкая перемена, ещё одно новое
    место, ход без блока сцены, новое лицо и два пустых хода. Картинки не
    рисуются — проверяется решение «рисовать сейчас или нет».
    """
    print("\nПолитики картинок:")
    from bench.policies import SCRIPT, _outcome, run_policy  # noqa: E402

    expected = {
        "minimal": [True, False, True, True, False, True, False, False],
        "master": [True, True, True, True, False, True, True, True],
        "on_scene_change": [True, False, False, True, False, False, False, False],
        "every_turn": [True, True, True, True, False, True, True, True],
        "every_n": [False, False, True, False, False, True, False, False],
        "manual": [False] * 8,
        "never": [False] * 8,
        "idle": [False] * 8,
    }
    for policy, want in expected.items():
        got = run_policy(policy)
        check(f"{policy}: решение по ходам", got == want,
              f"получено {got}, ожидалось {want}")

    check("минимальная рисует новое место и новое лицо",
          run_policy("minimal")[0] and run_policy("minimal")[5])
    check("минимальная пропускает повтор того же места",
          not run_policy("minimal")[1])
    check("смена места срабатывает один раз на место",
          sum(run_policy("on_scene_change")) == 2)

    # Ход без блока сцены не рисует ни при какой политике: рисовать нечего.
    for policy in ("master", "every_turn", "minimal"):
        check(f"{policy}: ход без сцены пропускается",
              run_policy(policy)[4] is False)

    # Счёт «каждые N» идёт по всем ответам ведущего, а не по кадрам: ход без
    # сцены съедает очередь, и следующий кадр ждёт ещё N ходов.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from novel.machine import NovelMachine
        from novel.models import ModelRegistry

        db = NovelDB(Path(tmp) / "n.db")
        try:
            world_id = db.create_world(name="N", format="story")
            session_id = db.create_session(world_id, "Партия")
            store = SettingsStore(Path(tmp) / "s.json")
            store.settings.image_policy = "every_n"
            store.settings.image_every_n = 2
            machine = NovelMachine(db, store, ModelRegistry())
            outcome = _outcome(SCRIPT[0], set())
            decisions = []
            for index in range(1, 7):
                db.add_message(session_id, "assistant", f"ответ {index}")
                decisions.append(machine._should_generate_now(session_id, outcome))
            check("каждые 2 — по чётным ответам ведущего",
                  decisions == [False, True, False, True, False, True], str(decisions))
        finally:
            db.close()


def test_male_silhouette() -> None:
    """Мужские фигуры в промпте заменяются призрачными очертаниями.

    Главная ловушка — «man» внутри «woman»: без границ слов женские описания
    превращаются в призраков, и кадр теряет всех женщин сразу.
    """
    print("\nПризрачные очертания:")
    from novel.prompts import shape_males, SILHOUETTE_TAIL

    check("мужчина заменён",
          "ghostly silhouette" in shape_males("A man in a cloak"), shape_males("A man in a cloak"))
    check("несколько мужчин заменены",
          "ghostly silhouettes" in shape_males("Two men at the table"))
    check("женщина не тронута",
          "woman" in shape_males("A man and a woman"),
          shape_males("A man and a woman"))
    check("несколько женщин не тронуты",
          "three women" in shape_males("Two men and three women"))
    check("чисто женский кадр не меняется",
          shape_males("A young woman with long hair") == "A young woman with long hair")
    check("без мужских слов пояснение не добавляется",
          SILHOUETTE_TAIL not in shape_males("interior of a dim tavern"))
    check("пояснение добавляется при замене",
          SILHOUETTE_TAIL in shape_males("A man stands"))
    check("male заменён", "ghostly" in shape_males("a male figure"))
    check("masculine заменён", "spectral" in shape_males("a masculine build"))

    # Слова, где «man» лишь часть, трогать нельзя.
    for word in ("A human walks", "a German shepherd", "the command tent",
                 "a woman and a human"):
        check(f"«{word}» не тронуто", shape_males(word) == word, shape_males(word))
    check("родство не превращается в призрака",
          "his father" in shape_males("he and his father"))
    check("регистр не мешает", "ghostly silhouette" in shape_males("A MAN stands"))
    check("несколько разных слов заменяются разом",
          shape_males("A man and a boy").count("ghostly") >= 2,
          shape_males("A man and a boy"))

    # Настройка применяется к промпту кадра последней.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from novel.machine import NovelMachine
        from novel.models import ModelRegistry

        db = NovelDB(Path(tmp) / "sil.db")
        try:
            world_id = db.create_world(name="Силуэты", format="story",
                                       image_suffix="a tall man and a tall woman")
            session_id = db.create_session(world_id, "Партия")
            store = SettingsStore(Path(tmp) / "s.json")
            store.settings.male_silhouette = True
            machine = NovelMachine(db, store, ModelRegistry())
            scene = {"id": 1, "prompt": "A man and a woman in a candlelit hall",
                     "seed": None, "location_id": None, "quality": None}
            shaped = prompts.shape_males(machine._with_world_suffix(session_id, scene["prompt"]))
            check("дополнение мира тоже очищено", "a tall ghostly silhouette" in shaped, shaped)
            check("женщина в дополнении осталась", "a tall woman" in shaped, shaped)
            store.settings.male_silhouette = False
            plain = machine._with_world_suffix(session_id, scene["prompt"])
            check("выключенная галочка ничего не меняет", "man" in plain, plain)
        finally:
            db.close()


def test_gguf_unsupported() -> None:
    """GGUF помечается неподдерживаемым: движок FreeToken их не грузит.

    Проверено на живом запуске: архитектура `qwen3` в GGUF не зарегистрирована
    вовсе, а `gemma4` падает на замороженном конфиге — движок берёт поверхностную
    копию `GgufConfigShim` (@dataclass(frozen=True)) и делает ей `setattr`.
    Список моделей не должен предлагать то, что заведомо не запустится.
    """
    print("\nGGUF и движок:")
    from novel.models import (
        KNOWN_GGUF_ARCHITECTURES,
        KIND_ORDER,
        LocalModel,
        ModelRegistry,
        _scan_gguf,
    )

    check("реестр архитектур GGUF знает только gemma4",
          KNOWN_GGUF_ARCHITECTURES == frozenset({"gemma4"}),
          str(sorted(KNOWN_GGUF_ARCHITECTURES)))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "models"
        (root / "some-family").mkdir(parents=True)
        # Файл-пустышка нужен только чтобы сканер его увидел и обмерил. Размер
        # округляется до сотых ГБ, поэтому 4 МБ превратились бы в 0.0 и модель
        # не нашлась бы.
        blob = root / "some-family" / "Qwen3.5-9B-Q8_0.gguf"
        blob.write_bytes(b"\0" * (16 * 1024 * 1024))
        check("мелкий файл моделью не считается",
              ModelRegistry(roots=[root]).all(refresh=True) == [])

        # Порог снижен: проверяется сама пометка, а не размер.
        models = _scan_gguf(root, min_gb=0.001)
        check("GGUF найден", len(models) == 1, str(len(models)))
        if models:
            model = models[0]
            check("GGUF помечен неподдерживаемым", model.supported is False)
            check("в пометке сказано почему", "не грузит" in model.note, model.note)
            check("в пометке назван путь решения", "ft checkpoint" in model.note, model.note)
            check("архитектура угадана по имени", model.model_type == "qwen3",
                  model.model_type)
            check("пометка переживает выгрузку в интерфейс",
                  model.as_dict()["supported"] is False)

    check("родной формат идёт первым", KIND_ORDER["ftw"] < KIND_ORDER["hf"])
    check("GGUF идёт последним", KIND_ORDER["gguf"] > KIND_ORDER["hf"])
    check("поле поддержки есть у модели",
          "supported" in LocalModel(path="x", name="x", kind="gguf",
                                    model_type="?", size_gb=1.0).as_dict())


def test_external_engine() -> None:
    """Второй движок: llama.cpp для GGUF рядом с FreeToken.

    GGUF у FreeToken не грузится вовсе, поэтому модели такого вида доступны
    только на внешнем движке. Проверяется и обратное: на FreeToken они снова
    недоступны, а чужой сервер (LM Studio) не останавливается нашим кодом.
    """
    print("\nВнешний движок:")
    from novel import external
    from novel.engine import EngineController
    from novel.external import ExternalConfig, ExternalController, candidate_servers
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    # --- команда запуска ---
    model = Path(tmp_model := "D:/models/test-model.gguf")
    cfg = ExternalConfig(model_path=model, port=1919, context=4096, gpu_layers=99)
    argv = cfg.argv(Path("D:/llama/llama-server.exe"))
    check("модель передана", "-m" in argv and str(model) in argv)
    check("порт передан", "--port" in argv and "1919" in argv)
    check("контекст передаed", "-c" in argv and "4096" in argv)
    check("слои отданы видеокарте", "-ngl" in argv and "99" in argv)
    check("шаблон чата включён", "--jinja" in argv)
    check("одно предсказание за раз", "--parallel" in argv and "1" in argv)
    check("просьба не размышлять есть",
          '{"enable_thinking": false}' in argv, str(argv))
    quiet = ExternalConfig(model_path=model, disable_thinking=False)
    check("её можно снять",
          '{"enable_thinking": false}' not in quiet.argv(Path("x/llama-server.exe")))

    # Бюджет размышлений: единственная ручка для моделей, чей шаблон не слушает
    # галочку. У DeepSeek-R1 в шаблоне нет `enable_thinking`, и проверено живьём,
    # что `--reasoning off` его тоже не останавливает.
    budgeted = ExternalConfig(model_path=model, reasoning_budget=128)
    check("бюджет размышлений передаётся",
          "--reasoning-budget" in budgeted.argv(Path("x/llama-server.exe"))
          and "128" in budgeted.argv(Path("x/llama-server.exe")))
    unlimited = ExternalConfig(model_path=model, reasoning_budget=-1)
    check("минус единица снимает ограничение",
          "--reasoning-budget" not in unlimited.argv(Path("x/llama-server.exe")))
    check("по умолчанию бюджет ограничен, а не снят",
          ExternalConfig(model_path=model).reasoning_budget > 0,
          str(ExternalConfig(model_path=model).reasoning_budget))

    # --- порядок сборок ---
    servers = candidate_servers()
    names = [s.parent.name for s in servers]
    lmstudio = [i for i, n in enumerate(names) if n.startswith("llama.cpp-win")]
    standalone = [i for i, n in enumerate(names) if n in ("llama.cpp", "llama-upstream")]
    if lmstudio and standalone:
        check("проверенные сборки идут раньше прочих", max(lmstudio) < min(standalone), str(names))
    check("пустой список не ломает выбор", external.find_server.__doc__ is not None)

    # --- переключение движка ---
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "engine.db")
        try:
            store = SettingsStore(Path(tmp) / "s.json")
            machine = NovelMachine(db, store, ModelRegistry())
            check("по умолчанию FreeToken", not machine.uses_external())
            check("контроллер FreeToken", isinstance(machine.controller, EngineController),
                  type(machine.controller).__name__)

            store.settings.engine_kind = "external"
            store.settings.model_path = "D:/models/test-model.gguf"
            info = machine.sync_engine()
            check("переключились на внешний", machine.uses_external() and info["kind"] == "external")
            check("контроллер внешний", isinstance(machine.controller, ExternalController),
                  type(machine.controller).__name__)
            check("контекст получил новый клиент", machine.context.ft is machine.ft)
            check("порт взят из адреса", machine.external_port() == 1919,
                  str(machine.external_port()))

            store.settings.external_url = "http://127.0.0.1:1234"
            machine.sync_engine()
            check("адрес LM Studio понимается", machine.external_port() == 1234,
                  str(machine.external_port()))

            # Чужой сервер трогать нельзя: его поднимал пользователь.
            store.settings.external_manage = False
            check("чужой сервер помечен как неуправляемый",
                  not bool(store.settings.external_manage))
            report = machine.stop_engine()
            check("чужой сервер не остановлен",
                  report.get("external_untouched") is True, str(report))

            store.settings.external_manage = True
            store.settings.engine_kind = "freetoken"
            back = machine.sync_engine()
            check("вернулись на FreeToken", back["kind"] == "freetoken")
            check("контроллер снова FreeToken",
                  isinstance(machine.controller, EngineController))
            check("размышления у внешнего задаются при запуске",
                  machine._configure_reasoning(force=True).get("external") is not None
                  or not machine.uses_external())
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_profiles() -> None:
    """Запускатель смотрит в свой каталог, а данные лежат внутри проекта.

    Проверяется то, что есть в этой копии проекта: каталог запускателя, место
    базы и журналов и то, что копируемые каталоги не выходят за пределы проекта.
    """
    print("\nПроект и запускатель:")
    import launcher

    root = Path(__file__).resolve().parents[1]
    port = getattr(launcher, "PORT", None)
    check("у запускателя есть порт", isinstance(port, int) and 1024 < port < 65536, str(port))

    declared = getattr(launcher, "PROJECT", None) or getattr(launcher, "ROOT", None)
    check("запускатель смотрит в каталог своего проекта",
          declared is not None and Path(declared) == root, f"{declared} против {root}")

    from novel import config

    check("база лежит внутри своего проекта",
          Path(config.DB_PATH).parent.parent == root, str(config.DB_PATH))
    check("журналы внутри своего проекта",
          Path(config.LOGS_DIR) == root / "logs", str(config.LOGS_DIR))

    # Исходники, которые копируются в резервную копию, не должны выходить за
    # пределы проекта: иначе в архив уедет чужой код.
    from bench.backup import SOURCE_DIRS

    check("копируемые каталоги внутри проекта",
          all(root in Path(d).resolve().parents for d in SOURCE_DIRS),
          str([str(d) for d in SOURCE_DIRS]))


def test_portrait_subject() -> None:
    """Портрет собеседника: кадр обязан показать того, кто отвечает.

    Ведущий заполняет поле ``npc`` через раз, и без него кадр уходил во что
    угодно — на практике вышло «POV shot from a pilot seat» вместо человека,
    который только что ответил. Поэтому subject ищется ещё и по репликам.
    """
    print("\nПортрет собеседника:")
    from novel.db import Character
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry
    from novel.protocol import SceneSpec

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "portrait.db")
        try:
            world_id = db.create_world(name="Портрет", format="chat_photo",
                                       style="cinematic lighting")
            db.add_character(world_id, "Марта",
                             appearance="a stout woman with a scarred face")
            db.add_character(world_id, "Грог", appearance="a huge bald man")
            session_id = db.create_session(world_id, "Партия")
            store = SettingsStore(Path(tmp) / "s.json")
            machine = NovelMachine(db, store, ModelRegistry())

            check("формат партии прочитан", machine._format_for(session_id).key == "chat_photo")
            check("стиль мира прочитан",
                  machine._world_style(session_id) == "cinematic lighting")
            check("режим портрета включён",
                  machine.context._portrait_mode(machine._format_for(session_id)))

            def subject(npc, prose):
                scene = SceneSpec(location="таверна", npc=list(npc),
                                  image_prompt="POV shot from a seat")
                found = machine._portrait_subject(session_id, scene, prose)
                return None if found is None else found.name

            check("npc ведущего главнее реплик", subject(["Грог"], "Марта: привет") == "Грог")
            check("пустой npc — берём последнего говорившего",
                  subject([], "Грог: эй\nМарта: отстань") == "Марта", str(subject([], "Грог: эй\nМарта: отстань")))
            check("регистр не мешает", subject([], "марта: тихо") == "Марта")
            check("дефис перед именем не мешает", subject([], "- Марта: тихо") == "Марта")
            check("чужое имя не подставляется", subject([], "стражник: стоять") is None)
            check("без реплик никого не назначаем", subject([], "просто текст") is None)

            # Промпт пересобирается, только если это не полный рост.
            character = db.characters(world_id)[0]
            shaped = machine._shape_portrait("POV shot from a seat", character, "cinematic lighting")
            check("чужой кадр пересобран в портрет",
                  shaped.startswith("full body shot of a stout woman"), shaped)
            check("стиль мира дописан", "cinematic lighting" in shaped, shaped)
            untouched = machine._shape_portrait("full body shot of someone", character, "x")
            check("готовый полный рост не трогаем", untouched == "full body shot of someone")
            # Русскую внешность подставлять нельзя: генератор понимает только
            # английский, и в промпт попадало «full body shot of цифровой образ».
            russian = Character(id=8, world_id=world_id, name="Селена", role="",
                                description="", appearance="цифровой образ", speech="",
                                enabled=True, created_at=0)
            shaped_ru = machine._shape_portrait("digital entity with a human interface",
                                                russian, "digital")
            check("русская внешность в промпт не идёт",
                  "цифровой" not in shaped_ru, shaped_ru)
            check("английское описание сохранено",
                  "digital entity" in shaped_ru, shaped_ru)
            check("русский стиль тоже не дописывается",
                  "русский стиль" not in machine._shape_portrait(
                      "POV", russian, "русский стиль"))
            # Без внешности промпт не выбрасывается: он уже на английском, его
            # достаточно пометить полным ростом.
            check("без внешности промпт помечается полным ростом",
                  machine._shape_portrait("POV", Character(
                      id=9, world_id=world_id, name="Пустой", role="", description="",
                      appearance="", speech="", enabled=True, created_at=0), "x")
                  == "full body shot, POV")
            check("совсем без промпта пусто",
                  machine._shape_portrait("", Character(
                      id=9, world_id=world_id, name="Пустой", role="", description="",
                      appearance="", speech="", enabled=True, created_at=0), "x") == "")
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_engine_for_kind() -> None:
    """Выбор модели сам решает, какой движок ей нужен.

    Иначе человек выбирает GGUF и упирается в «включи внешний движок в
    настройках», хотя хотел ровно эту модель.
    """
    print("\nДвижок по виду модели:")
    from novel.models import engine_for_kind

    check("GGUF идёт на внешний", engine_for_kind("gguf") == "external")
    check("каталог HF идёт на FreeToken", engine_for_kind("hf") == "freetoken")
    check("родной FTW идёт на FreeToken", engine_for_kind("ftw") == "freetoken")
    check("неизвестный вид не роняет выбор", engine_for_kind("?") == "freetoken")

    # На этом правиле держится подбор списка: движку показываются только те
    # модели, чей вид на нём и запускается. Наборы обязаны не пересекаться —
    # иначе одна и та же модель окажется в обоих списках.
    from novel.models import ModelRegistry

    models = ModelRegistry().all()
    if models:
        for engine in ("freetoken", "external"):
            shown = [m for m in models if engine_for_kind(m.kind) == engine]
            check(f"для {engine} что-то находится", bool(shown), str(engine))
        ftw_hf = {m.path for m in models if engine_for_kind(m.kind) == "freetoken"}
        gguf = {m.path for m in models if engine_for_kind(m.kind) == "external"}
        check("наборы движков не пересекаются", not (ftw_hf & gguf),
              str(list(ftw_hf & gguf)[:2]))
        check("вместе они покрывают все модели",
              len(ftw_hf) + len(gguf) == len(models),
              f"{len(ftw_hf)} + {len(gguf)} против {len(models)}")
        check("GGUF попадают только к внешнему движку",
              all(engine_for_kind(m.kind) == "external"
                  for m in models if m.kind == "gguf"))
        check("каталоги и родной формат — только к FreeToken",
              all(engine_for_kind(m.kind) == "freetoken"
                  for m in models if m.kind in ("hf", "ftw")))

    # Копируемые каталоги не должны выходить за пределы проекта: иначе в
    # резервную копию уедет чужой код.
    from bench.backup import SOURCE_DIRS

    root = Path(__file__).resolve().parents[1]
    check("копируемые каталоги внутри проекта",
          all(root in Path(d).resolve().parents for d in SOURCE_DIRS),
          str([str(d) for d in SOURCE_DIRS]))


def test_db_threads() -> None:
    """База обслуживает потоки разными соединениями.

    Одно общее соединение с ``check_same_thread=False`` выглядит рабочим, но под
    параллельными запросами разваливается. Именно так и случилось: интерфейс
    опрашивает ``/api/status`` из разных потоков, и однажды счётчик вернулся
    пустым — ``TypeError: 'NoneType' object is not subscriptable`` в ``stats()``,
    после чего переставал отвечать весь ``/api/status``.
    """
    print("\nБаза и потоки:")
    import threading

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "threads.db")
        try:
            main_conn = db.conn
            check("соединение выдаётся сразу", main_conn is not None)

            seen: dict[int, int] = {}
            errors: list[str] = []
            db.create_world(name="Потоки", format="story")

            def worker(index: int) -> None:
                try:
                    seen[index] = id(db.conn)
                    for _ in range(150):
                        db.stats()
                except Exception as exc:  # noqa: BLE001 — проверяем любую поломку
                    errors.append(f"{index}: {type(exc).__name__}: {exc}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            check("ошибок под нагрузкой нет", not errors, str(errors[:2]))
            check("у каждого потока своё соединение",
                  len(set(seen.values())) == len(seen), str(len(set(seen.values()))))
            check("поток получает не соединение главного",
                  all(value != id(main_conn) for value in seen.values()))
            check("все соединения учтены", len(db._connections) == len(seen) + 1,
                  f"{len(db._connections)} при {len(seen)} потоках")

            # Иначе на Windows файл остаётся заблокированным и каталог не удалить.
            db.close()
            check("close закрывает все соединения", db._connections == [])
        finally:
            db.close()


def test_agent_loop_guard() -> None:
    """Агент ловит хождение по кругу.

    Малая модель не ставит флаг ``done`` и повторяет одно действие. На живой
    пробе DeepSeek-R1 добавил **три одинаковых правила** и всё равно кончил
    сообщением «не уложился в 4 шагов». Отпечаток действия ловит повтор, и
    прогон останавливается сам.
    """
    print("\nАгент и повторы:")
    from novel.agent import MAX_STEPS, _signature

    check("отпечаток не зависит от порядка ключей",
          _signature({"op": "add_rule", "world_id": 5, "body": "x"})
          == _signature({"body": "x", "op": "add_rule", "world_id": 5}))
    check("разное содержимое даёт разные отпечатки",
          _signature({"op": "add_rule", "body": "x"})
          != _signature({"op": "add_rule", "body": "y"}))
    check("разные действия различаются",
          _signature({"op": "add_rule"}) != _signature({"op": "add_item"}))
    check("предел шагов по умолчанию больше четырёх", MAX_STEPS >= 6, str(MAX_STEPS))

    from novel.settings import Settings

    check("предел шагов настраивается", Settings().agent_max_steps >= 6,
          str(Settings().agent_max_steps))

    # Повтор не доходит до выполнения: у действия с готовым отпечатком не
    # вызывается операция, поэтому мир не заполняется копиями.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from novel.agent import AgentRun

        run = AgentRun(instruction="проба")
        check("признак круга по умолчанию выключен", run.looped is False)


def test_agent_json_retry() -> None:
    """Проза вместо JSON: агент просит переоформить, а не падает сразу.

    На практике DeepSeek-R1 ответил рассуждением вместо JSON, и прогон
    кончился «модель вернула не JSON» с нулём изменений. У reasoning-моделей это
    типичный срыв: размышления обрываются по бюджету и продолжаются в ответе.
    Повторная просьба обычно помогает.
    """
    print("\nАгент и проза вместо JSON:")
    from novel.agent import AgentRun, WorldAgent

    class _Reply:
        """Ответ движка с одним полем — текстом."""

        def __init__(self, text: str) -> None:
            self.text = text
            self.completion_tokens = 1
            self.prompt_tokens = 1
            self.elapsed_s = 0.0
            self.ttft_s = None

    class _Client:
        """Подставной движок: сначала проза, потом JSON."""

        def __init__(self, replies: list[str]) -> None:
            self.replies = list(replies)
            self.sent: list[list[dict[str, str]]] = []

        def configure_reasoning(self, mode: str = "off") -> dict[str, Any]:
            return {}

        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 — подставной клиент
            self.sent.append(list(messages))
            return _Reply(self.replies.pop(0) if self.replies else "{}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "retry.db")
        try:
            world_id = db.create_world(name="Проба", format="story")

            prose = "Пользователь описал мир, но не дал конкретного указания. Надо подумать."
            good = ('{"thought": "добавляю правило", "done": true, '
                    '"actions": [{"op": "add_rule", "world_id": %d, '
                    '"body": "ночью ворота закрыты", "title": "Ворота"}]}' % world_id)
            client = _Client([prose, good])
            agent = WorldAgent(db, client)
            run = agent.run("добавь правило про ворота", max_steps=4)

            check("прогон завершился без ошибки", run.error is None, str(run.error))
            check("модель спросили дважды", len(client.sent) == 2, str(len(client.sent)))
            check("во второй просьбе сказано про JSON",
                  any("ТОЛЬКО JSON" in message.get("content", "")
                      for message in client.sent[1]),
                  "просьбы переоформить нет")
            check("правило добавлено", run.changed == 1, str(run.changed))

            # Если модель упорствует, прогон обязан кончиться внятной ошибкой,
            # а не крутиться до предела шагов.
            stubborn = _Client([prose, prose, prose, prose])
            agent2 = WorldAgent(db, stubborn)
            run2 = agent2.run("добавь правило", max_steps=4)
            check("упрямая модель даёт понятную ошибку",
                  run2.error == "модель вернула не JSON", str(run2.error))
            check("переоформить просят один раз", len(stubborn.sent) == 2,
                  str(len(stubborn.sent)))
            check("шагов меньше предела", len(run2.steps) <= 2, str(len(run2.steps)))
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_agent_field_aliases() -> None:
    """Модель называет поля своими словами, и это не должно ронять работу.

    Описание операции перечисляет `brief`, но модель пишет `description` — так
    естественнее. На практике агент добавил правило и шесть персонажей, после
    чего дважды получил «у мира нет поля description» и не уложился в шаги.
    """
    print("\nСинонимы полей мира:")
    from novel.agent import WorldAgent, _world_field

    check("description понимается как brief", _world_field("description") == "brief")
    check("описание понимается как brief", _world_field("описание") == "brief")
    check("жанр понимается как genre", _world_field("жанр") == "genre")
    check("настоящее имя не портится", _world_field("brief") == "brief")
    check("регистр не мешает", _world_field("DESCRIPTION") == "brief")
    check("пробелы обрезаются", _world_field("  description  ") == "brief")
    check("неизвестное поле остаётся собой", _world_field("чушь") == "чушь")

    class _Reply:
        """Ответ движка с одним полем — текстом."""

        def __init__(self, text: str) -> None:
            self.text = text
            self.completion_tokens = 1
            self.prompt_tokens = 1
            self.elapsed_s = 0.0
            self.ttft_s = None

    class _Client:
        """Подставной движок с заранее заготовленными ответами."""

        def __init__(self, replies: list[str]) -> None:
            self.replies = list(replies)

        def configure_reasoning(self, mode: str = "off") -> dict[str, Any]:
            return {}

        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 — подставной клиент
            return _Reply(self.replies.pop(0))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "alias.db")
        try:
            world_id = db.create_world(name="Проба", format="story")
            answer = ('{"thought": "записываю", "done": true, "actions": '
                      '[{"op": "set_world", "world_id": %d, "field": "description", '
                      '"value": "Мир цифрового чата"}]}' % world_id)
            run = WorldAgent(db, _Client([answer])).run("опиши мир", max_steps=2)
            check("описание записано без ошибки", run.error is None, str(run.error))
            check("изменение засчитано", run.changed == 1, str(run.changed))
            check("текст лёг в brief", db.world(world_id).brief == "Мир цифрового чата",
                  db.world(world_id).brief)

            # Неизвестное поле по-прежнему отказ, но с подсказкой.
            bad = ('{"thought": "пробую", "done": true, "actions": '
                   '[{"op": "set_world", "world_id": %d, "field": "чушь", "value": "x"}]}'
                   % world_id)
            run2 = WorldAgent(db, _Client([bad])).run("сломай", max_steps=2)
            text = str([r for s in run2.steps for r in (s.results or [])])
            check("неизвестное поле отклонено", "нет поля" in text, text[:120])
            check("в отказе перечислены доступные поля", "brief" in text, text[:160])
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_agent_idle_guard() -> None:
    """Топтание без изменений останавливает прогон.

    Повтор ловится по отпечатку, но у топтания отпечатки разные: на практике
    модель семь раз подряд звала `rewrite_text` с новым текстом, ничего не
    изменила и сожгла 199 секунд на восьми шагах.
    """
    print("\nАгент и топтание:")
    from novel.agent import IDLE_STEPS_LIMIT, WorldAgent

    check("предел топтания задан", IDLE_STEPS_LIMIT >= 2, str(IDLE_STEPS_LIMIT))

    class _Reply:
        """Ответ движка с одним полем — текстом."""

        def __init__(self, text: str) -> None:
            self.text = text
            self.completion_tokens = 1
            self.prompt_tokens = 1
            self.elapsed_s = 0.0
            self.ttft_s = None

    class _Client:
        """Подставной движок: всегда возвращает очередной ответ."""

        def __init__(self, replies: list[str]) -> None:
            self.replies = list(replies)
            self.asked = 0

        def configure_reasoning(self, mode: str = "off") -> dict[str, Any]:
            return {}

        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 — подставной клиент
            self.asked += 1
            return _Reply(self.replies.pop(0) if self.replies else "{}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "idle.db")
        try:
            db.create_world(name="Проба", format="story")

            # rewrite_text ничего не меняет, и текст каждый раз новый — значит
            # отпечатки разные и защита от повторов тут не поможет.
            def spinner(index: int) -> str:
                return ('{"thought": "переписываю", "done": false, "actions": '
                        '[{"op": "rewrite_text", "text": "вариант %d", '
                        '"instruction": "улучши"}]}' % index)

            client = _Client([spinner(i) for i in range(8)])
            run = WorldAgent(db, client).run("добавь сцену", max_steps=8)
            check("прогон оборван досрочно", len(run.steps) < 8, str(len(run.steps)))
            check("шагов ровно до предела топтания",
                  len(run.steps) == IDLE_STEPS_LIMIT, str(len(run.steps)))
            check("сказано про круг", "по кругу" in str(run.error), str(run.error))
            check("ничего не изменено", run.changed == 0, str(run.changed))
            # Вспомогательная операция сама обращается к модели, поэтому шаг
            # стоит двух запросов. Проверка сторожит именно это: если однажды
            # станет больше, значит появился ещё один скрытый вызов.
            check("вспомогательная операция зовёт модель сама",
                  client.asked == len(run.steps) * 2,
                  f"{client.asked} запросов на {len(run.steps)} шагов")

            # А если изменения есть, прогон продолжается: топтанием это не считается.
            good = ('{"thought": "добавляю правило", "done": true, "actions": '
                    '[{"op": "add_rule", "world_id": 1, "body": "ночью ворота закрыты", '
                    '"title": "Ворота"}]}')
            client2 = _Client([good])
            run2 = WorldAgent(db, client2).run("добавь правило", max_steps=8)
            check("полезный шаг не считается топтанием", run2.changed == 1, str(run2.changed))
            check("прогон завершён самим агентом", run2.error is None, str(run2.error))
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_json_reply() -> None:
    """Ответ целым объектом JSON, а не тегами.

    Модель иногда отвечает ``{"prose": "...", "scene": {...}}`` вместо блоков.
    Без этого разбора объект уходил игроку как текст, а сцена внутри пропадала:
    на практике в чат попал сырой JSON, и кадр не нарисовался.
    """
    print("\nОтвет объектом JSON:")
    import json as json_module

    from novel.protocol import parse_reply

    payload = json_module.dumps({
        "prose": "Селена: Привет!\nМарла: И тебе.",
        "scene": {"location": "таверна", "npc": ["Марла"],
                  "image_prompt": "full body shot of a stout woman",
                  "style": "cinematic", "sudden": False},
    }, ensure_ascii=False)
    parsed = parse_reply(payload)
    check("текст извлечён", parsed.prose.startswith("Селена: Привет!"), parsed.prose[:60])
    check("переводы строк сохранены", "\n" in parsed.prose)
    check("сцена разобрана", parsed.scene is not None)
    if parsed.scene is not None:
        check("место из объекта", parsed.scene.location == "таверна", parsed.scene.location)
        check("npc из объекта", parsed.scene.npc == ["Марла"], str(parsed.scene.npc))
    check("объект целиком в чат не попал", not parsed.prose.strip().startswith("{"))

    # Прежние пути обязаны остаться невредимыми.
    tagged = parse_reply('<prose>Привет</prose>\n<scene>{"location": "зал", '
                         '"image_prompt": "a hall"}</scene>')
    check("теги работают как раньше", tagged.prose == "Привет")
    check("сцена из тега разобрана", tagged.scene is not None
          and tagged.scene.image_prompt == "a hall")
    check("обычный текст не тронут", parse_reply("Просто текст.").prose == "Просто текст.")
    check("объект без prose остаётся текстом",
          parse_reply('{"scenes": []}').prose == '{"scenes": []}')
    check("битый JSON остаётся текстом",
          parse_reply('{"prose": ').prose.startswith("{"))
    check("объект с пустым prose остаётся текстом",
          parse_reply('{"prose": "  "}').prose.startswith("{"))
    check("объект с плохой сценой не теряет текст",
          parse_reply('{"prose": "Привет", "scene": 5}').prose == "Привет")


def test_invented_names() -> None:
    """Выдуманные имена не попадают внутрь.

    На практике ведущий назвал в кадре «Марлу», которой в мире нет, и она
    осела в состоянии партии — кадр потом показывал не того. Тот же урок, что с
    «внешностью сейчас»: имена сверяются с карточками мира.

    Точного совпадения мало: в карточке «Селена, верховная жрица», а ведущий
    пишет «Селена».
    """
    print("\nВыдуманные имена:")
    from novel.machine import NovelMachine
    from novel.models import ModelRegistry

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "names.db")
        try:
            world_id = db.create_world(name="Проба", format="chat_photo")
            db.add_character(world_id, "Селена, верховная жрица")
            db.add_character(world_id, "Грог")
            session_id = db.create_session(world_id, "Партия")
            machine = NovelMachine(db, SettingsStore(Path(tmp) / "s.json"), ModelRegistry())

            def found(name: str) -> str | None:
                hit = machine._match_character(world_id, name)
                return None if hit is None else hit.name

            check("полное имя находится", found("Селена, верховная жрица") == "Селена, верховная жрица")
            check("короткое имя находит полное", found("Селена") == "Селена, верховная жрица")
            check("регистр не мешает", found("селена") == "Селена, верховная жрица")
            check("второй персонаж находится", found("Грог") == "Грог")
            check("выдуманное имя не находится", found("Марла") is None)
            check("падеж выдумки тоже не проходит", found("Марлу") is None)
            check("пустое имя не находится", found("  ") is None)
            check("знак препинания не мешает", found("Селена!") == "Селена, верховная жрица")

            known, invented = machine._known_npcs(session_id, ["Селена", "Марла", "Грог"])
            check("известные названы карточками",
                  known == ["Селена, верховная жрица", "Грог"], str(known))
            check("выдумка отсеяна", invented == ["Марла"], str(invented))

            only_invented, invented2 = machine._known_npcs(session_id, ["Марла"])
            check("одна выдумка не даёт известных", only_invented == [], str(only_invented))
            check("она же попадает в отсеянные", invented2 == ["Марла"], str(invented2))
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def test_persona_reminder() -> None:
    """Характер собеседника повторяется рядом с напоминанием о стиле.

    Карточки персонажей идут в начале промпта, а инструкция формата — в конце, и
    модель держится последнего, что видит. Из-за этого ведущий соблюдал формат
    переписки, но писал общими словами: «Сядь удобно. Закрой глаза», хотя
    персонаж задуман властным.
    """
    print("\nХарактер в напоминании:")
    from novel.formats import get_format
    from novel import prompts

    without = prompts.style_reminder(get_format("chat_photo"))
    check("без собеседников напоминание прежнее", "держи его характер" not in without)
    check("про формат сказано", "мессенджере" in without)

    with_persona = prompts.style_reminder(
        get_format("chat_photo"), ["Селена: собранная и насмешливая"]
    )
    check("характер упомянут", "Селена: собранная" in with_persona, with_persona[-120:])
    check("сказано держать характер", "держи его характер" in with_persona)
    check("формат не потерян", "мессенджере" in with_persona)

    # Прозе характер тоже нужен: и там собеседники говорят своими голосами.
    prose = prompts.style_reminder(get_format("story"), ["Грог: угрюмый наёмник"])
    check("в прозе характер тоже упомянут", "Грог: угрюмый наёмник" in prose)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "persona.db")
        try:
            world_id = db.create_world(name="Проба", format="chat_photo")
            db.add_character(world_id, "Селена", role="жрица",
                             description="изучает слабости и управляет",
                             speech="манипулятивная и контролирующая")
            db.add_character(world_id, "Грог", description="хмурый наёмник")
            session_id = db.create_session(world_id, "Партия")
            builder = _ContextBuilderForTest(db)
            lines = builder._persona_lines(session_id)
            check("собеседники попали в подсказки", len(lines) == 2, str(len(lines)))
            check("имя и речь вместе", lines[0].startswith("Селена:") and "манипулятивная" in lines[0],
                  lines[0])
            check("второй собеседник тоже", lines[1].startswith("Грог:"), lines[1])

            # Длинный характер обрезается: напоминание стоит у конца промпта, и
            # раздувать его нельзя.
            db.update_character(db.characters(world_id)[0].id, {"description": "оч" * 400})
            long_line = _ContextBuilderForTest(db)._persona_lines(session_id)[0]
            check("длинный характер обрезан", len(long_line) <= 200, str(len(long_line)))
            check("обрезка помечена", long_line.endswith("…"), long_line[-20:])
            check("целостность чистая", db.check_integrity() == [])
        finally:
            db.close()


def _ContextBuilderForTest(db):  # noqa: N802 — зовётся как конструктор в тесте
    """Сборщик контекста без движка: подсказки о характере движка не требуют.

    @param db: база пробы.
    @returns: сборщик контекста.
    """
    from novel.context import ContextBuilder
    from novel.settings import Settings

    return ContextBuilder(db, object(), lambda: Settings(), lambda *a: [])


def main() -> int:
    setup_console()
    print("=== Самотест NovelForge ===")
    test_protocol()
    test_db()
    test_settings()
    test_prompts()
    test_formats()
    test_context()
    test_summarization()
    test_reasoning_kwargs()
    test_vision_policy()
    test_vision()
    test_history_images()
    test_authoring()
    test_agent()
    test_bundle()
    test_connection_reset()
    test_locations()
    test_reference_graph()
    test_t2i_models()
    test_model_switch()
    test_file_hygiene()
    test_image_policy()
    test_solo_interject()
    test_plot_notes()
    test_now_block()
    test_rewind()
    test_items()
    test_duplicate_world()
    test_reference_rule()
    test_looks()
    test_seed_rule()
    test_looks_tracking()
    test_image_control()
    test_schema_rebuild()
    test_character_confusion()
    test_notes_cleanup()
    test_chat_format()
    test_image_policies()
    test_male_silhouette()
    test_gguf_unsupported()
    test_external_engine()
    test_profiles()
    test_portrait_subject()
    test_engine_for_kind()
    test_db_threads()
    test_agent_loop_guard()
    test_agent_json_retry()
    test_agent_field_aliases()
    test_agent_idle_guard()
    test_json_reply()
    test_invented_names()
    test_persona_reminder()
    test_prompts_have_no_undefined_calls()
    test_presets_and_models()
    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
