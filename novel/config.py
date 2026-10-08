"""Пути, порты и внешние исполняемые файлы проекта NovelForge.

Каждое значение можно переопределить переменной окружения с префиксом
``NOVELFORGE_``. Это нужно, чтобы тестовый прогон мог занять другие порты или
каталоги, не правя код.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

#: Корень проекта. По умолчанию берётся каталог, где лежит сам код: копия
#: проекта тогда работает сама по себе и не трогает данные оригинала. Жёсткий
#: путь здесь означал бы, что копия пишет в чужую базу.
ROOT = Path(os.environ.get("NOVELFORGE_ROOT") or Path(__file__).resolve().parents[1])

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "novel.db"
LOGS_DIR = ROOT / "logs"
BENCH_DIR = ROOT / "bench"
MEASUREMENTS_DIR = BENCH_DIR / "measurements"
DOCS_DIR = ROOT / "docs"

# --- FreeToken ---------------------------------------------------------------

FREETOKEN_BASE_URL = os.environ.get("NOVELFORGE_FT_URL", "http://127.0.0.1:1919")

#: Каталог модели по умолчанию: рядом с проектом, в ``models``. Реальный путь
#: почти всегда задаётся переменной окружения или в настройках — каталог моделей
#: у всех свой.
DEFAULT_MODELS_DIR = Path(os.environ.get("NOVELFORGE_MODELS_DIR", ROOT / "models"))

FREETOKEN_MODEL_PATH = Path(
    os.environ.get("NOVELFORGE_FT_MODEL", DEFAULT_MODELS_DIR / "model")
)
FREETOKEN_SERVED_NAME = os.environ.get("NOVELFORGE_FT_NAME", FREETOKEN_MODEL_PATH.name)
FREETOKEN_HOME = Path(
    os.environ.get("NOVELFORGE_FT_HOME", str(Path(os.environ["LOCALAPPDATA"]) / "FreeToken"))
)
FREETOKEN_FT_EXE = FREETOKEN_HOME / "venv" / "Scripts" / "ft.exe"
FREETOKEN_PYTHON = FREETOKEN_HOME / "venv" / "Scripts" / "python.exe"

# --- ComfyUI -----------------------------------------------------------------
#
# Где стоит ComfyUI, у каждого своё, поэтому путь задаёт пользователь — во
# вкладке «Машина». Значения ниже нужны, пока он этого не сделал: сначала
# берётся переменная окружения, потом значение по умолчанию.
#
# Функции ``comfy_*`` читают настройки на каждом вызове, а не один раз при
# загрузке модуля: пользователь может поменять путь на ходу, и следующий кадр
# должен уйти уже по новому адресу.

COMFY_BASE_URL = os.environ.get("NOVELFORGE_COMFY_URL", "http://127.0.0.1:8188")

#: Значения по умолчанию: пустая настройка означает именно их.
DEFAULT_COMFY_ROOT = Path(os.environ.get("NOVELFORGE_COMFY_ROOT", ROOT / "ComfyUI"))
DEFAULT_COMFY_INPUT_DIR = Path(os.environ.get("NOVELFORGE_COMFY_INPUT",
                                              ROOT / "comfy-shared" / "input"))
DEFAULT_COMFY_OUTPUT_DIR = Path(os.environ.get("NOVELFORGE_COMFY_OUTPUT",
                                               ROOT / "comfy-shared" / "output"))
DEFAULT_COMFY_DESKTOP_EXE = Path(
    os.environ.get("NOVELFORGE_COMFY_DESKTOP", r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe")
)
DEFAULT_COMFY_SERVER_SCRIPT = os.environ.get("NOVELFORGE_COMFY_SCRIPT", r"ComfyUI\main.py")
DEFAULT_COMFY_MODEL_PATHS = Path(os.environ.get(
    "NOVELFORGE_COMFY_PATHS",
    str(Path(os.environ.get("APPDATA", "")) / "Comfy Desktop"
        / "instance-model-paths" / "inst-1781266641216.yaml"),
))

#: Настройки пользователя. Заполняет ``novel.settings`` при запуске: так пути
#: из интерфейса попадают сюда, не заставляя модули импортировать друг друга.
_user_settings: Any = None


def use_settings(settings: Any) -> None:
    """Подключает настройки пользователя к путям ComfyUI.

    @param settings: объект настроек либо ``None``, чтобы вернуться к умолчаниям.
    """
    global _user_settings
    _user_settings = settings


def _from_settings(name: str, fallback: Path) -> Path:
    """Берёт путь из настроек, если он задан.

    @param name: имя поля настроек, например ``comfy_root``.
    @param fallback: значение по умолчанию.
    @returns: путь из настроек либо ``fallback``.
    """
    value = getattr(_user_settings, name, "") if _user_settings is not None else ""
    return Path(str(value)) if value else fallback


def models_dirs() -> list[Path]:
    """Каталоги, где лежат локальные модели.

    Каталог моделей у каждого свой, поэтому он задаётся пользователем: списком
    во вкладке «Машина» либо переменной окружения. Пустая настройка означает
    ``models`` рядом с проектом.

    @returns: список существующих каталогов; пустой, если ни одного нет.
    """
    raw = ""
    if _user_settings is not None:
        raw = str(getattr(_user_settings, "models_dirs", "") or "")
    if not raw:
        raw = os.environ.get("NOVELFORGE_MODELS_DIRS", "")
    if not raw:
        candidates = [DEFAULT_MODELS_DIR]
    else:
        candidates = [Path(item) for item in raw.split(os.pathsep) if item.strip()]
    return [path for path in candidates if path.is_dir()]


def comfy_root() -> Path:
    """Каталог ComfyUI, выбранный пользователем.

    @returns: путь к каталогу установки.
    """
    return _from_settings("comfy_root", DEFAULT_COMFY_ROOT)


def comfy_python() -> Path:
    """Интерпретатор, которым запускается сервер ComfyUI.

    @returns: путь к ``python.exe`` внутри окружения ComfyUI.
    """
    return comfy_root() / ".venv" / "Scripts" / "python.exe"


def comfy_input_dir() -> Path:
    """Каталог, откуда ComfyUI читает картинки-образцы.

    @returns: путь к каталогу входов.
    """
    return _from_settings("comfy_input_dir", DEFAULT_COMFY_INPUT_DIR)


def comfy_output_dir() -> Path:
    """Каталог, куда ComfyUI складывает готовые кадры.

    @returns: путь к каталогу выходов.
    """
    return _from_settings("comfy_output_dir", DEFAULT_COMFY_OUTPUT_DIR)


def comfy_server_cwd() -> Path:
    """Рабочий каталог запуска сервера ComfyUI.

    Скрипт запуска задан относительно него, поэтому каталог выбирается так,
    чтобы скрипт в нём нашёлся. У сборки Comfy Desktop скрипт лежит в
    подкаталоге ``ComfyUI``, у обычной — прямо в корне.

    @returns: путь к рабочему каталогу.
    """
    root = comfy_root()
    # Скрипт уже задан от корня — значит запускать надо из корня.
    if (root / comfy_server_script()).is_file():
        return root
    # Скрипт по умолчанию ищется в подкаталоге: так устроена сборка Desktop.
    if (root / "ComfyUI" / "main.py").is_file():
        return root
    # Ничего не нашлось — берём каталог, куда указывает сам путь.
    return root.parent if root.name.lower() == "comfyui" else root


def comfy_server_script() -> str:
    """Путь к ``main.py`` ComfyUI относительно рабочего каталога.

    @returns: относительный путь к скрипту запуска.
    """
    value = getattr(_user_settings, "comfy_server_script", "") if _user_settings else ""
    return str(value) if value else DEFAULT_COMFY_SERVER_SCRIPT


def comfy_desktop_exe() -> Path:
    """Путь к приложению Comfy Desktop, если оно установлено.

    @returns: путь к исполняемому файлу.
    """
    return DEFAULT_COMFY_DESKTOP_EXE


def comfy_model_paths() -> Path:
    """Файл путей к моделям от Comfy Desktop.

    @returns: путь к файлу; может не существовать.
    """
    value = getattr(_user_settings, "comfy_model_paths", "") if _user_settings else ""
    return Path(str(value)) if value else DEFAULT_COMFY_MODEL_PATHS


#: Совместимость: код читает эти имена как атрибуты.
COMFY_ROOT = DEFAULT_COMFY_ROOT
COMFY_PYTHON = DEFAULT_COMFY_ROOT / ".venv" / "Scripts" / "python.exe"
COMFY_INPUT_DIR = DEFAULT_COMFY_INPUT_DIR
COMFY_OUTPUT_DIR = DEFAULT_COMFY_OUTPUT_DIR
COMFY_DESKTOP_EXE = DEFAULT_COMFY_DESKTOP_EXE
COMFY_SERVER_CWD = DEFAULT_COMFY_ROOT
COMFY_SERVER_SCRIPT = DEFAULT_COMFY_SERVER_SCRIPT
COMFY_MODEL_PATHS_YAML = DEFAULT_COMFY_MODEL_PATHS


def comfy_server_argv() -> list[str]:
    """Команда прямого запуска сервера ComfyUI.

    Повторяет аргументы, с которыми сервер поднимает Comfy Desktop, включая
    дополнительный файл путей к моделям: без него генератор не увидит ни одного
    чекпоинта.

    @returns: список аргументов процесса.
    """
    return [
        str(comfy_python()),
        "-s",
        comfy_server_script(),
        "--enable-manager",
        "--extra-model-paths-config",
        str(comfy_model_paths()),
        "--input-directory",
        str(comfy_input_dir()),
        "--output-directory",
        str(comfy_output_dir()),
    ]

# --- Модели графического пайплайна (имена, как их видит загрузчик ComfyUI) ----
#
# Имя файла — это то, что ComfyUI видит в своём каталоге, а не путь. Каталог у
# каждого свой, поэтому имена ищутся среди файлов, а не заданы строкой: иначе
# значение по умолчанию указывало бы на файл, которого у пользователя нет, и
# генерация падала бы с «Value not in list», не объясняя причину.
#
# Если файлов несколько, берётся первый по алфавиту; конкретный выбирается
# переменной окружения.

#: Подкаталоги ComfyUI, где лежат нужные для графа файлы.
T2I_MODEL_FOLDERS = {
    "clip_name": "text_encoders",
    "unet_name": "diffusion_models",
    "vae_name": "vae",
}

#: Каталоги моделей ComfyUI. Основной — внутри установки; рядом с каталогом
#: выходов проверяется второй: у Comfy Desktop модели часто лежат вне установки.
def comfy_model_roots() -> list[Path]:
    """Каталоги, где искать модели пайплайна.

    @returns: список каталогов по убыванию приоритета.
    """
    roots: list[Path] = []
    from_env = os.environ.get("NOVELFORGE_COMFY_MODELS")
    if from_env:
        roots.append(Path(from_env))
    roots.append(comfy_root() / "models")
    roots.append(comfy_output_dir().parent / "models")
    shared = os.environ.get("NOVELFORGE_COMFY_SHARED_MODELS")
    if shared:
        roots.append(Path(shared))
    return roots


#: Что искать в этих подкаталогах. Порядок шаблонов — от точного к общему:
#: берётся первый, по которому что-то нашлось.
#:
#: ``mmproj-`` исключается у текстового кодировщика: это проектор зрения для
#: моделей с картинками, а не сам кодировщик текста, и загрузчик его не примет.
T2I_MODEL_PATTERNS = {
    "clip_name": ("qwen3vl*.gguf", "*qwen3vl*.gguf", "*qwen*vl*.gguf"),
    "unet_name": ("qwen_image*.gguf", "*qwen*image*.gguf"),
    "vae_name": ("qwen_image_2*vae*.safetensors", "qwen_image*vae*.safetensors"),
}

#: Что исключать из найденного, кроме подкаталогов.
T2I_MODEL_EXCLUDE = ("mmproj",)

#: Переменные окружения, перекрывающие найденное.
T2I_MODEL_ENV = {
    "clip_name": "NOVELFORGE_T2I_CLIP",
    "unet_name": "NOVELFORGE_T2I_UNET",
    "vae_name": "NOVELFORGE_T2I_VAE",
}


def _find_t2i_model(role: str) -> str:
    """Ищет файл модели для узла графа ComfyUI.

    @param role: ``clip_name``, ``unet_name`` или ``vae_name``.
    @returns: имя файла, как его видит ComfyUI; пустая строка, если не найден.
    """
    override = os.environ.get(T2I_MODEL_ENV[role])
    if override:
        return override

    # Шаблоны перебираются во внешнем цикле, каталоги — во внутреннем: точное
    # имя важнее того, в каком каталоге оно нашлось. Иначе общий шаблон в первом
    # каталоге перебил бы точный файл во втором.
    for pattern in T2I_MODEL_PATTERNS[role]:
        for root in comfy_model_roots():
            folder = root / T2I_MODEL_FOLDERS[role]
            if not folder.is_dir():
                continue
            found = [
                path for path in sorted(folder.glob(pattern))
                if not any(word in path.name.lower() for word in T2I_MODEL_EXCLUDE)
            ]
            if found:
                return found[0].name
    return ""


def t2i_models() -> dict[str, str]:
    """Имена моделей пайплайна для текущего каталога ComfyUI.

    Ищутся при каждом обращении: пользователь может сменить каталог на ходу, и
    следующий кадр должен уйти уже с моделями из нового места.

    @returns: ключи ``clip_name``, ``unet_name``, ``vae_name``.
    """
    return {role: _find_t2i_model(role) for role in T2I_MODEL_FOLDERS}


#: Совместимость: значения на момент загрузки модуля.
COMFY_MODEL_ROOTS = [DEFAULT_COMFY_ROOT / "models", DEFAULT_COMFY_OUTPUT_DIR.parent / "models"]
QWEN_T2I_GRAPH_MODELS = {role: _find_t2i_model(role) for role in T2I_MODEL_FOLDERS}

# --- Прочее ------------------------------------------------------------------

#: Интерпретатор, под которым запускаются сам оркестратор и GUI.
PYTHON = Path(os.environ.get("NOVELFORGE_PYTHON", sys.executable))

#: Сколько ждать, пока nvidia-smi вернёт ответ, секунды.
NVIDIA_SMI_TIMEOUT_S = 15.0

#: Интервал опроса nvidia-smi в фоновом замерщике, секунды.
SAMPLE_INTERVAL_S = 0.5


def ensure_dirs() -> None:
    """Создаёт каталоги данных, если их ещё нет."""
    for path in (DATA_DIR, IMAGES_DIR, UPLOADS_DIR, LOGS_DIR, MEASUREMENTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
