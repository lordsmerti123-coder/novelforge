"""Реестр локальных моделей и пресеты настроек под каждую.

Движок FreeToken обслуживает один чекпоинт за запуск, поэтому смена модели — это
перезапуск движка с другим ``--model``. Реестр нужен, чтобы пользователь выбирал
модель из списка, а не вписывал путь руками, и чтобы к каждой модели
подставлялись осмысленные параметры сэмплинга.

Пресеты — это только значения по умолчанию. Всё, что они задают, пользователь
может переопределить: настройки хранятся отдельно и имеют приоритет.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from novel import config

#: Каталоги, в которых ищутся чекпоинты. Задаются пользователем: у каждого свои.
def _search_roots() -> list[Path]:
    """Каталоги поиска чекпоинтов, которые есть на диске.

    Отсутствующий каталог молча пропускается: на чистой копии проекта каталога
    моделей ещё нет, и это не ошибка.

    @returns: список существующих каталогов; пустой, если ни одного нет.
    """
    return config.models_dirs()


def search_roots() -> list[Path]:
    """Каталоги поиска на текущий момент.

    Читаются при каждом обращении: пользователь может добавить каталог на ходу,
    и список моделей должен обновиться.

    @returns: список существующих каталогов.
    """
    return _search_roots()


#: Совместимость: значения на момент загрузки модуля.
SEARCH_ROOTS = _search_roots()

#: Архитектуры, которые движок умеет обслуживать.
SUPPORTED_TYPES = {
    "gemma4", "gemma3", "qwen2", "qwen3", "qwen3_moe", "qwen3_vl", "qwen3_5_moe",
    "qwen4_exp", "glm4_moe", "glm5_next", "gpt_oss", "deepseek_v4", "llama",
    "mistral", "minimax_m2", "minimax_m3", "muse_glimmer",
}

#: Пресеты сэмплинга и запуска по семейству модели.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gemma4": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 900,
        "engine_memory_ratio": 0.9,
        "engine_moe_strategy": "offload",
        "note": "Рекомендованные самой моделью значения; хороша в прозе, любит длинные ответы.",
    },
    "qwen3_5_moe": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 900,
        "engine_memory_ratio": 0.9,
        "engine_moe_strategy": "offload",
        "note": (
            "Больше 35B параметров при 3B активных. По умолчанию уходит в длинные "
            "размышления и тратит на них весь бюджет ответа — держи размышления выключенными."
        ),
    },
    "gpt_oss": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "max_tokens": 900,
        "engine_memory_ratio": 0.9,
        "engine_moe_strategy": "offload",
        "note": "Reasoning-модель: тратит токены на размышление перед ответом.",
    },
    "qwen3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 900,
        "engine_memory_ratio": 0.9,
        "engine_moe_strategy": "offload",
        "note": "Плотная модель без MoE: помещается целиком, но требует больше VRAM.",
    },
    "_default": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 900,
        "engine_memory_ratio": 0.9,
        "engine_moe_strategy": "offload",
        "note": "Общие значения: подойдут, пока не подобраны свои.",
    },
}


#: Архитектуры GGUF, зарегистрированные в движке FreeToken. Одна, и та падает
#: на замороженном конфиге, поэтому GGUF считаются неподдерживаемыми целиком.
KNOWN_GGUF_ARCHITECTURES = frozenset({"gemma4"})

def engine_for_kind(kind: str) -> str:
    """Какой движок нужен модели этого вида.

    Выбор модели сам переключает движок: GGUF грузит только llama.cpp, каталоги
    HF и родной формат FTW — только FreeToken. Иначе человек выбирает модель и
    упирается в «включи движок», хотя хотел ровно эту модель.

    @param kind: вид чекпоинта — ``gguf``, ``hf`` или ``ftw``.
    @returns: ``external`` или ``freetoken``.
    """
    return "external" if kind == "gguf" else "freetoken"


#: Порядок вывода по формату: сначала готовые к запуску, потом требующие проверки.
KIND_ORDER = {"ftw": 0, "hf": 1, "gguf": 2}


@dataclass
class LocalModel:
    """Один найденный чекпоинт."""

    path: str
    name: str
    kind: str
    model_type: str
    size_gb: float
    experts: int | None = None
    experts_per_tok: int | None = None
    quant: str | None = None
    max_ctx: int | None = None
    supported: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса."""
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "model_type": self.model_type,
            "size_gb": self.size_gb,
            "experts": self.experts,
            "experts_per_tok": self.experts_per_tok,
            "quant": self.quant,
            "max_ctx": self.max_ctx,
            "supported": self.supported,
            "note": self.note,
            "preset": preset_for(self.model_type),
        }


def preset_for(model_type: str) -> dict[str, Any]:
    """Пресет настроек для семейства модели.

    @param model_type: значение ``model_type`` из конфига чекпоинта.
    @returns: словарь значений по умолчанию с пометкой, откуда он взялся.
    """
    preset = dict(MODEL_PRESETS.get(model_type, MODEL_PRESETS["_default"]))
    preset["source"] = model_type if model_type in MODEL_PRESETS else "_default"
    return preset


def _folder_size_gb(path: Path, limit: int = 200) -> float:
    total = 0
    seen = 0
    for item in path.iterdir():
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
        seen += 1
        if seen > limit:
            break
    return round(total / (1024**3), 2)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _scan_folders(root: Path) -> list[LocalModel]:
    """Ищет чекпоинты-каталоги: Hugging Face и родной формат движка FTW.

    У FTW-моделей рядом с ``config.json`` лежат шарды ``*.ftw`` и описание
    ``freetoken_weight.json`` — отдельный формат, поэтому он и помечается
    отдельно: такие модели уже готовы к запуску и не требуют конвертации.
    """
    found: list[LocalModel] = []
    for config_path in root.rglob("config.json"):
        folder = config_path.parent
        safetensors = any(folder.glob("*.safetensors"))
        ftw = any(folder.glob("*.ftw"))
        if not safetensors and not ftw:
            continue
        config = _read_json(config_path)
        if not config:
            continue
        architectures = config.get("architectures") or []
        model_type = str(config.get("model_type") or (architectures[0] if architectures else "?"))
        quant = None
        quant_file = folder / "hf_quant_config.json"
        if quant_file.is_file():
            quant = _read_json(quant_file).get("quant_algo")
        experts = (
            config.get("num_experts")
            or config.get("num_local_experts")
            or config.get("n_routed_experts")
        )
        supported = model_type.lower() in SUPPORTED_TYPES
        kind = "ftw" if ftw else "hf"
        found.append(
            LocalModel(
                path=str(folder),
                name=folder.name,
                kind=kind,
                model_type=model_type,
                size_gb=_folder_size_gb(folder, limit=400),
                experts=int(experts) if experts else None,
                experts_per_tok=config.get("num_experts_per_tok"),
                quant=str(quant) if quant else None,
                max_ctx=config.get("max_position_embeddings"),
                supported=supported,
                note=(
                    "родной формат движка, конвертация не нужна"
                    if kind == "ftw"
                    else ("" if supported else "движок не умеет эту архитектуру")
                ),
            )
        )
    return found


def _scan_gguf(root: Path, min_gb: float = 3.0) -> list[LocalModel]:
    """Ищет одиночные GGUF-файлы — только чтобы показать, что они есть.

    **Движок FreeToken их не запускает.** В реестре архитектур GGUF у него
    зарегистрирована одна `gemma4`, и даже она падает: конфиг из GGUF —
    замороженный dataclass, а движок присваивает ему поля. Поэтому файлы
    помечаются неподдерживаемыми: иначе список моделей состоит из мёртвых
    вариантов, а выбор каждого оборачивается пятиминутным отказом запуска.

    Найти их всё равно полезно: видно, что лежит на диске. Путь к рабочей
    модели — выгрузка safetensors и ``ft checkpoint``.

    @param root: каталог поиска.
    @param min_gb: ниже этого размера файл не считается моделью.
    @returns: найденные GGUF, помеченные как неподдерживаемые.
    """
    found: list[LocalModel] = []
    known = sorted(SUPPORTED_TYPES, key=len, reverse=True)
    for path in root.rglob("*.gguf"):
        try:
            size_gb = round(path.stat().st_size / (1024**3), 2)
        except OSError:
            continue
        if size_gb < min_gb:
            continue
        lowered = path.name.lower()
        guess = next((kind for kind in known if kind.replace("_", "") in lowered.replace("_", "").replace("-", "")), "?")
        found.append(
            LocalModel(
                path=str(path),
                name=path.stem,
                kind="gguf",
                model_type=guess,
                size_gb=size_gb,
                supported=False,
                note=("GGUF: движок FreeToken их не грузит — нужна выгрузка "
                      "safetensors и «ft checkpoint»"),
            )
        )
    return found


def discover_models(roots: list[Path] | None = None) -> list[LocalModel]:
    """Собирает все локальные модели из каталогов поиска.

    @param roots: каталоги поиска; по умолчанию берутся из настроек.
    @returns: список моделей, отсортированный по имени.
    """
    models: list[LocalModel] = []
    for root in roots or search_roots():
        if not root.is_dir():
            continue
        models.extend(_scan_folders(root))
        models.extend(_scan_gguf(root))
    models.sort(key=lambda item: (not item.supported, KIND_ORDER.get(item.kind, 9), item.name.lower()))
    return models


@dataclass
class ModelRegistry:
    """Кэш найденных моделей: обход диска небыстрый, а нужен на каждый запрос."""

    #: Пусто — брать каталоги из настроек при каждом обходе: пользователь может
    #: добавить свой каталог на ходу, и список должен обновиться.
    roots: list[Path] = field(default_factory=list)
    _models: list[LocalModel] | None = None

    def all(self, refresh: bool = False) -> list[LocalModel]:
        """Список моделей, при необходимости перечитанный с диска."""
        if self._models is None or refresh:
            self._models = discover_models(self.roots or None)
        return self._models

    def by_path(self, path: str) -> LocalModel | None:
        """Находит модель по пути."""
        for model in self.all():
            if model.path == path:
                return model
        return None

    def as_dicts(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Список моделей для интерфейса."""
        return [model.as_dict() for model in self.all(refresh=refresh)]
