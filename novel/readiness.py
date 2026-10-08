"""Проверка, всё ли на месте для работы программы.

Новый пользователь ставит программу на чистую машину: у него может не быть ни
одного движка, ни одной модели, ни ComfyUI. Тогда интерфейс не должен молчать и
показывать пустые списки — он должен сказать, чего нет и что с этим делать.

Модуль ничего не запускает и не занимает память: только смотрит, что есть.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from novel import config, external, metrics, models

#: Ссылки для подсказок. Держатся здесь, а не в интерфейсе: их читает и README.
FREETOKEN_URL = "https://www.flashml.ai/"
FREETOKEN_REPO = "https://github.com/FlashML-org/FreeToken"
LMSTUDIO_URL = "https://lmstudio.ai/"
COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI"


@dataclass
class Check:
    """Одна проверка готовности.

    :param key: короткое имя для интерфейса.
    :param title: что проверялось, словами.
    :param ok: всё ли на месте.
    :param detail: что именно найдено или не найдено.
    :param hint: что делать, если не на месте.
    :param essential: без этого история не пойдёт. Необязательное — картинки и
        чужой сервер — не повод показывать предупреждение поверх всего.
    """

    key: str
    title: str
    ok: bool
    detail: str
    hint: str = ""
    essential: bool = False


@dataclass
class Readiness:
    """Сводка готовности: что есть, чего нет и что делать.

    :param checks: проведённые проверки по порядку важности.
    """

    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Готова ли программа к работе хотя бы одним движком."""
        return all(check.ok for check in self.checks if check.essential)

    @property
    def blocking(self) -> list[Check]:
        """Невыполненные обязательные проверки.

        Только они заслуживают предупреждения поверх всего: без движка или
        моделей история не пойдёт. Отсутствие картинок — не беда, о ней
        достаточно сказать в настройках.
        """
        return [check for check in self.checks if check.essential and not check.ok]

    @property
    def optional_missing(self) -> list[Check]:
        """Невыполненные необязательные проверки."""
        return [check for check in self.checks if not check.essential and not check.ok]

    @property
    def text_ready(self) -> bool:
        """Есть ли чем вести историю: движок плюс хотя бы одна модель."""
        engines = [check for check in self.checks if check.key in ("freetoken", "llamacpp")]
        models_ok = next((c.ok for c in self.checks if c.key == "models"), False)
        return models_ok and any(check.ok for check in engines)

    def as_dict(self) -> dict[str, object]:
        """Представление для интерфейса."""
        def row(c: Check) -> dict[str, object]:
            return {"key": c.key, "title": c.title, "ok": c.ok,
                    "detail": c.detail, "hint": c.hint, "essential": c.essential}

        return {
            "ok": self.ok,
            "text_ready": self.text_ready,
            "blocking": [row(c) for c in self.blocking],
            "optional_missing": [row(c) for c in self.optional_missing],
            "checks": [row(c) for c in self.checks],
        }


def _freetoken_check() -> Check:
    """Проверяет текстовый движок FreeToken."""
    problem = config.freetoken_problem()
    if not problem:
        return Check("freetoken", "FreeToken", True, f"найден: {config.FREETOKEN_FT_EXE}",
                     essential=True)
    return Check(
        "freetoken", "FreeToken", False,
        f"нет движка в {config.FREETOKEN_HOME}",
        f"Поставьте FreeToken с {FREETOKEN_URL} — приложение создаст движок само. "
        f"Либо укажите каталог переменной NOVELFORGE_FT_HOME. "
        f"Без него остаётся внешний движок.",
        essential=True,
    )


def _llamacpp_check() -> Check:
    """Проверяет внешний движок llama.cpp."""
    server = external.find_server()
    if server is not None:
        return Check("llamacpp", "llama.cpp", True, f"найден: {server}", essential=True)
    return Check(
        "llamacpp", "llama.cpp", False,
        "рабочей сборки llama-server.exe не найдено",
        f"Сборка идёт в комплекте с LM Studio ({LMSTUDIO_URL}). Либо положите "
        f"llama-server.exe рядом с проектом, либо укажите путь в настройках, "
        f"либо задайте список каталогов переменной NOVELFORGE_LLAMACPP_SERVERS.",
        essential=True,
    )


def _external_server_check(settings: Any) -> Check:
    """Проверяет внешний сервер текстового движка.

    Сервер нужен не всегда: когда сборка llama.cpp есть у программы, она
    поднимает его сама, и «никто не отвечает» — обычное состояние, а не беда.
    Ошибкой это становится, только если сервер поднимает пользователь.

    @param settings: настройки приложения.
    @returns: проверка сервера.
    """
    url = str(getattr(settings, "external_url", "") or "http://127.0.0.1:1919")
    manages = bool(getattr(settings, "external_manage", True))
    if external.is_external_url_alive(url):
        loaded = external.lmstudio_loaded(url)
        state = (f"загружено моделей: {len(loaded)}" if loaded
                 else "ни одна модель не загружена")
        return Check("server", "Внешний сервер", True, f"отвечает на {url}, {state}")
    if manages:
        return Check(
            "server", "Внешний сервер", True,
            f"на {url} никто не отвечает — программа поднимет свой",
        )
    return Check(
        "server", "Внешний сервер", False,
        f"на {url} никто не отвечает, а поднимать его должна программа извне",
        "Сервер запускается там, где вы его держите: в LM Studio это вкладка "
        "Developer, кнопка Start Server. Адрес задаётся в настройках: "
        "у LM Studio это http://127.0.0.1:1234.",
    )


def _models_check(roots: list[Path], found: int) -> Check:
    """Проверяет, нашлись ли модели."""
    if found:
        where = ", ".join(str(path) for path in roots) or "каталоги не заданы"
        return Check("models", "Модели", True, f"найдено {found} в {where}", essential=True)
    return Check(
        "models", "Модели", False,
        "ни одной модели не найдено",
        "Модели программа не скачивает. Укажите каталог, где они лежат, в поле "
        "«Каталоги с моделями»: она обойдёт его и покажет, что нашла.",
        essential=True,
    )


def _comfyui_check() -> Check:
    """Проверяет генератор изображений ComfyUI."""
    root = config.comfy_root()
    python = config.comfy_python()
    script = Path(config.comfy_server_script())
    if script.is_absolute():
        script_path = script
    else:
        # Путь к main.py отсчитывается от рабочего каталога запуска, а не от
        # каталога установки: у Comfy Desktop они разные.
        script_path = config.comfy_server_cwd() / script
    if python.is_file() and script_path.is_file():
        return Check("comfyui", "ComfyUI", True, f"найден: {root}")
    missing = []
    if not python.is_file():
        missing.append(f"интерпретатор {python}")
    if not script_path.is_file():
        missing.append(f"скрипт {script_path}")
    return Check(
        "comfyui", "ComfyUI", False,
        "не найдено: " + ", ".join(missing),
        f"Картинки рисует ComfyUI ({COMFYUI_URL}), он ставится отдельно. "
        f"Задайте его каталог и путь к main.py в настройках. Без него история "
        f"работает, иллюстраций не будет.",
    )


def check_readiness(
    registry: models.ModelRegistry | None = None,
    settings: Any = None,
) -> Readiness:
    """Собирает сводку готовности.

    Проверки идут от важного к необязательному: сначала текстовые движки и
    модели, потом внешний сервер и картинки. Ничего не запускается.

    @param registry: реестр моделей; при ``None`` создаётся свой.
    @param settings: настройки приложения; при ``None`` читаются с диска.
    @returns: сводка с проверками.
    """
    if settings is None:
        from novel.settings import SettingsStore

        try:
            settings = SettingsStore().load()
        except (OSError, ValueError):
            settings = None
    found_models = (registry.all() if registry is not None else models.discover_models())
    roots = config.models_dirs()
    return Readiness([
        _freetoken_check(),
        _llamacpp_check(),
        _models_check(roots, len(found_models)),
        _external_server_check(settings),
        _comfyui_check(),
    ])


def free_vram_gb() -> float:
    """Сколько видеопамяти свободно.

    @returns: гигабайты; 0, если прочитать не удалось.
    """
    try:
        return round(metrics.gpu_stats()["free_mb"] / 1024, 1)
    except (RuntimeError, OSError):
        return 0.0
