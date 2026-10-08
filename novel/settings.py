"""Настройки: параметры модели, бюджет контекста, политика картинок.

Хранятся отдельным JSON-файлом, а не в базе: их правят руками и переносят между
мирами, а история — нет.

Настройки делятся на три группы: параметры сэмплинга (что и как долго пишет
модель), бюджет контекста (сколько истории она видит) и поведение генератора
изображений (когда и как рисуются кадры).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from novel import config

#: Порт внешнего движка по умолчанию. Профиль может задать свой через
#: ``NOVELFORGE_EXTERNAL_URL``: иначе приватный профиль подключился бы к движку
#: основного — на том же порту 1919 — и молча говорил бы не с той моделью.
EXTERNAL_DEFAULT_URL = os.environ.get("NOVELFORGE_EXTERNAL_URL",
                                      "http://127.0.0.1:1919")

SETTINGS_PATH = config.DATA_DIR / "settings.json"

#: Ступени качества кадра: ключ — (сторона, шагов, подпись для интерфейса).
#: Ориентировочное время кадра при свободной видеопамяти: 640/12 — около 20 c,
#: 768/20 — около 38 c, 1024/25 — около 69 c.
IMAGE_QUALITY: dict[str, tuple[int, int, str]] = {
    "fast": (640, 12, "быстро — кадр вдвое-втрое скорее"),
    "normal": (768, 20, "обычно — разумный баланс скорости и качества"),
    "quality": (1024, 25, "качественно — самый подробный кадр"),
}


@dataclass
class Settings:
    """Все изменяемые параметры пайплайна."""

    # --- модель ---
    model_path: str = str(config.FREETOKEN_MODEL_PATH)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 900
    request_timeout_s: float = 600.0
    #: Пустая строка — брать параметры сэмплинга из самого чекпоинта.
    sampling_from_model: bool = True
    #: off | on | auto — что делать с размышлениями reasoning-моделей.
    #: Для роли ведущего они вредны: съедают бюджет ответа и время, а игрок их
    #: всё равно не видит.
    reasoning_mode: str = "off"
    #: Сколько токенов отдать размышлениям сверх ответа. Размышления и ответ
    #: делят один предел, и без запаса думающая модель не начинает отвечать.
    reasoning_reserve_tokens: int = 3000
    #: Какой движок держит текстовую модель. ``freetoken`` умеет каталоги HF и
    #: родной формат FTW, ``external`` — llama.cpp, а он грузит GGUF, который
    #: FreeToken не берёт вовсе.
    engine_kind: str = "freetoken"
    #: Путь к ``llama-server.exe``. Пусто — искать рабочую сборку самому.
    external_server_path: str = ""
    #: Адрес внешнего сервера. Для llama.cpp порт наш, для LM Studio — его 1234.
    external_url: str = EXTERNAL_DEFAULT_URL
    #: Запускать сервер своими силами. Ложь — подключаться к чужому: сервер
    #: LM Studio поднимает сам пользователь, и поднимать второй нельзя.
    external_manage: bool = True
    #: Каталоги локальных моделей через ``;``. Пусто — ``models`` рядом с
    #: проектом. У каждого модели лежат в своём месте, поэтому путь задаёт
    #: пользователь: программа обходит каталоги и показывает список найденного.
    models_dirs: str = ""
    #: Последняя модель, выбранная для каждого движка. Движки не взаимозаменяемы
    #: (у FreeToken каталог HF, у llama.cpp — GGUF или имя в LM Studio), поэтому
    #: помнится отдельно: при возврате на движок подставляется то, с чем уже
    #: работали, а не первое из списка.
    last_model_freetoken: str = ""
    last_model_external: str = ""
    # --- картинки (ComfyUI) ---
    #: Каталог ComfyUI. Пусто — взять значение по умолчанию из ``config``.
    #: Установлен он у каждого в своём месте, поэтому путь задаёт пользователь.
    comfy_root: str = ""
    #: Каталог, из которого ComfyUI читает картинки для образцов.
    comfy_input_dir: str = ""
    #: Каталог, куда ComfyUI складывает готовые кадры.
    comfy_output_dir: str = ""
    #: Путь к ``main.py`` ComfyUI внутри его каталога.
    comfy_server_script: str = ""
    #: Файл путей к моделям от Comfy Desktop. Пусто — запуск без него.
    comfy_model_paths: str = ""
    #: Длина контекста внешнего сервера. Больше — лучше помнит, но ест память.
    external_context: int = 8192
    #: Сколько слоёв модели отдать видеокарте; 99 означает «все».
    external_gpu_layers: int = 99
    #: Просить шаблон чата не размышлять. У Gemma-4 это убирает служебный блок
    #: ``<|channel>thought`` целиком, а без него размышления съедают весь бюджет
    #: ответа и текста не остаётся вовсе.
    external_disable_thinking: bool = True
    #: Бюджет размышлений внешнего движка, в токенах. Галочка выше помогает не
    #: всем: у DeepSeek-R1 в шаблоне нет ``enable_thinking``, и размышления не
    #: отключить — только ограничить. ``-1`` снимает ограничение. Ноль ставить
    #: нельзя: размышления не пропадают, а переезжают в сам ответ.
    external_reasoning_budget: int = 256
    #: Сколько шагов «подумал — сделал» даётся агенту-редактору за одно указание.
    #: Малая модель не ставит флаг ``done`` и упирается в предел: она повторяет
    #: одно действие и не заканчивает работу. Запаса нужно с избытком — у
    #: reasoning-модели на шаг уходят десятки секунд размышлений, и на указание
    #: с несколькими действиями короткого предела не хватает.
    agent_max_steps: int = 8

    # --- бюджет контекста ---
    #: Сколько токенов отводится на весь запрос, включая ответ модели.
    context_budget_tokens: int = 16384
    #: Жёсткий предел числа сообщений в дословном окне.
    context_turns: int = 12
    #: Сколько выброшенных сообщений накапливается, прежде чем сворачивать историю.
    summarize_after_dropped: int = 6
    #: Предел длины суммаризации.
    summary_max_tokens: int = 400
    #: Сворачивать историю автоматически.
    auto_summarize: bool = True

    # --- движок ---
    engine_memory_ratio: float = 0.9
    engine_moe_strategy: str = "offload"
    engine_autostart: bool = True
    engine_start_timeout_s: float = 900.0

    # --- изображения ---
    #: Когда рисовать кадр:
    #:
    #: * ``minimal`` — минимум картинок: новое место, новое лицо в кадре или
    #:   отмеченная ведущим резкая перемена обстановки. Основания проверяет
    #:   оркестратор, поэтому обещанные кадры не пропадают;
    #: * ``master`` — целиком на усмотрение ведущего: он поставил блок scene —
    #:   кадр рисуется;
    #: * ``on_scene_change`` — при смене названия места;
    #: * ``every_turn``, ``every_n`` — по расписанию, независимо от ведущего;
    #: * ``manual`` — только по кнопке; ``never`` — никогда; ``idle`` — в простое.
    image_policy: str = "minimal"
    #: Шаг для политики every_n.
    image_every_n: int = 3
    #: Ступень качества кадра. ``custom`` — брать размер и шаги из полей ниже.
    #: Ступени заданы готовыми наборами, а не свободным ползунком: размер обязан
    #: быть кратен 32, а время растёт как точки, умноженные на шаги.
    image_quality: str = "normal"
    image_size: int = 1024
    image_steps: int = 25
    image_cfg: float = 1.0
    draft_size: int = 512
    draft_steps: int = 8
    image_timeout_s: float = 1200.0

    #: Что рисовать в форматах-переписке: «portrait» — портрет собеседника,
    #: «scene» — сцену целиком.
    chat_frame: str = "portrait"
    #: Мужские персонажи уходят в призрачные очертания: слова вроде «man»
    #: заменяются прямо в промпте кадра. Скрытая настройка — в интерфейсе
    #: спрятана под раскрывающимся разделом.
    male_silhouette: bool = False

    def output_budget(self) -> int:
        """Предел ответа с запасом под размышления.

        Размышления и ответ идут по одному счёту ``max_tokens``. Когда
        размышления включены, предел складывается из видимого ответа и запаса —
        иначе модель тратит весь предел на размышления и не отвечает.

        @returns: предел вывода в токенах.
        """
        total = max(1, int(self.max_tokens))
        if str(self.reasoning_mode) == "on":
            total += max(0, int(self.reasoning_reserve_tokens))
        return total

    def image_dimensions(self, quality: str | None = None) -> tuple[int, int]:
        """Размер стороны и число шагов для выбранной ступени качества.

        @param quality: ступень; по умолчанию текущая настройка.
        @returns: ``(сторона, шагов)``.
        """
        step = quality or self.image_quality
        preset = IMAGE_QUALITY.get(step)
        if preset is None:
            return self.image_size, self.image_steps
        return preset[0], preset[1]

    # --- промпты изображений ---
    #: inline — модель пишет image_prompt сама в блоке scene;
    #: separate — отдельный короткий вызов переписывает описание в промпт.
    image_prompt_mode: str = "inline"
    image_prompt_max_tokens: int = 200

    # --- изображения на входе модели ---
    #: Когда модель получает изображения. Зрение стоит времени и токенов: кадр
    #: 768x768 — это около 258 токенов входа и заметная добавка ко времени
    #: ответа. Поэтому оно применяется по необходимости, а не на каждом ходу.
    #:
    #: * ``off`` — вложения сохраняются в истории, но модели не отправляются;
    #: * ``on_attach`` — только то, что приложил игрок: он приложил фото
    #:   осознанно, значит, хочет, чтобы модель на него посмотрела;
    #: * ``on_attach_and_last_frame`` — плюс последний сгенерированный кадр,
    #:   чтобы модель держала визуальную непрерывность между ходами.
    vision_mode: str = "on_attach"
    #: Сколько прошлых вложений пересылать вместе с текущей репликой. Ноль —
    #: только текущее: старый кадр стоит столько же и нужен редко.
    history_images: int = 0

    # --- постоянные места ---
    #: Напоминать ведущему названия уже известных мест. Стоит около тридцати
    #: токенов и удерживает места постоянными: без этого «замок» и «цитадель» —
    #: два разных места, и кадр каждый раз рисуется заново, без образца.
    location_memory: bool = True
    #: Показывать ведущему вещи игрока и персонажей. Он о них знает и может
    #: описывать их вид, но не распоряжается: тратить и отдавать может только игрок.
    items_memory: bool = True
    #: Спрашивать модель отдельным коротким запросом, изменилась ли чья-то
    #: внешность. В основном ответе модель про этот блок забывает, поэтому
    #: вопрос задаётся напрямую. Стоит ещё одного запроса на каждый ход.
    looks_tracking: bool = True

    # --- предгенерация ---
    speculative_enabled: bool = False
    idle_generation: bool = False
    idle_budget_s: float = 30.0

    # --- отладка ---
    debug_mode: bool = True
    #: Хранить полный текст последнего запроса и ответа.
    debug_keep_payloads: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Представление для сохранения и передачи в интерфейс."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Собирает настройки из словаря, игнорируя незнакомые ключи.

        Старый булев переключатель зрения переносится в режим: файлы настроек,
        записанные до появления ``vision_mode``, должны читаться по-прежнему, а
        не сбрасывать выбор пользователя на значение по умолчанию.

        @param data: словарь настроек, обычно прочитанный из файла.
        @returns: настройки с подставленными значениями по умолчанию.
        """
        known = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        if "vision_mode" not in filtered and data.get("vision_enabled") is False:
            filtered["vision_mode"] = "off"
        return cls(**filtered)

    def update(self, changes: dict[str, Any]) -> list[str]:
        """Применяет изменения на месте.

        @param changes: пары «поле — новое значение».
        @returns: имена полей, которые не удалось применить.
        """
        rejected: list[str] = []
        known = {item.name: item for item in fields(self)}
        for key, value in changes.items():
            spec = known.get(key)
            if spec is None:
                rejected.append(key)
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    coerced: Any = bool(value)
                elif isinstance(current, int):
                    coerced = int(value)
                elif isinstance(current, float):
                    coerced = float(value)
                else:
                    coerced = str(value)
            except (TypeError, ValueError):
                rejected.append(key)
                continue
            setattr(self, key, coerced)
        return rejected

    def remember_model(self, engine: str, path: str) -> None:
        """Запоминает модель, выбранную для движка.

        Пригодится при возврате на этот движок: подставится то, с чем уже
        работали, а не первое из списка.

        @param engine: ``external`` или ``freetoken``.
        @param path: путь к каталогу HF либо имя модели сервера.
        """
        field_name = "last_model_external" if engine == "external" else "last_model_freetoken"
        if hasattr(self, field_name):
            setattr(self, field_name, str(path or ""))

    def last_model_for(self, engine: str) -> str:
        """Модель, с которой в прошлый раз работали на этом движке.

        @param engine: ``external`` или ``freetoken``.
        @returns: путь или имя модели; пустая строка, если движок ещё не выбирали.
        """
        field_name = "last_model_external" if engine == "external" else "last_model_freetoken"
        return str(getattr(self, field_name, "") or "")

    def apply_model_preset(self, preset: dict[str, Any]) -> None:
        """Подставляет пресет модели, не трогая выбор модели и отладку.

        @param preset: словарь из :func:`novel.models.preset_for`.
        """
        for key in ("temperature", "top_p", "top_k", "max_tokens",
                    "engine_memory_ratio", "engine_moe_strategy"):
            if key in preset and preset[key] is not None:
                setattr(self, key, preset[key])


@dataclass
class SettingsStore:
    """Читает и пишет настройки, сохраняя их между запусками."""

    path: Path = SETTINGS_PATH
    settings: Settings = field(default_factory=Settings)

    def load(self) -> Settings:
        """Читает настройки с диска; при любой проблеме возвращает значения по умолчанию."""
        if not self.path.exists():
            return self.settings
        try:
            # ``utf-8-sig``, а не ``utf-8``: Блокнот и PowerShell пишут JSON
            # с меткой порядка байтов, и обычное чтение на ней падает. Тогда
            # настройки молча заменялись бы умолчаниями — вместе с путями.
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return self.settings
        if isinstance(data, dict):
            self.settings = Settings.from_dict(data)
        return self.settings

    def save(self, settings: Settings | None = None) -> None:
        """Записывает настройки на диск."""
        if settings is not None:
            self.settings = settings
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
