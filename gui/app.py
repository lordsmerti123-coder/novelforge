"""Веб-интерфейс NovelForge.

Один локальный сервер на ``127.0.0.1:8760``: он же отдаёт страницу, он же
проксирует запросы к движку и генератору. Браузер напрямую с FreeToken и ComfyUI
не общается, поэтому CORS не нужен.

Тяжёлые операции — ход игрока, разбор мира, генерация картинок — идут в рабочих
потоках. Интерфейс опрашивает ``/api/status`` и по нему рисует состояние,
поэтому страница не висит сто секунд на одном запросе.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel import config  # noqa: E402
from novel.comfy import ComfyClient, ComfyError  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.formats import FORMATS, get_format, list_formats  # noqa: E402
from novel.freetoken import FreeTokenError  # noqa: E402
from novel.machine import (  # noqa: E402
    DEFAULT_NOTE_HORIZON,
    MODEL_NOTE_LIMIT,
    PLAYER_NOTE_LIMIT,
    NovelMachine,
)
from novel.external import is_external_url_alive, list_server_models  # noqa: E402
from novel.models import ModelRegistry, engine_for_kind, preset_for  # noqa: E402
from novel.presets import get_preset, list_presets  # noqa: E402
from novel.settings import IMAGE_QUALITY, SettingsStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NovelForge", version="0.2.0")

config.ensure_dirs()
_db = NovelDB()
_settings_store = SettingsStore()
_settings_store.load()
# Пути к ComfyUI пользователь задаёт сам, поэтому настройки передаются в config:
# иначе он брал бы значения по умолчанию, а у пользователя всё стоит в других
# местах.
config.use_settings(_settings_store.settings)
_registry = ModelRegistry()
_machine = NovelMachine(_db, _settings_store, _registry)

_task_lock = threading.Lock()
_task: dict[str, Any] = {"running": False, "kind": None, "started_at": None, "error": None}


class ChatRequest(BaseModel):
    """Тело запроса на ход игрока."""

    session_id: int
    text: str
    interject: str = ""
    #: Изображения, приложенные к реплике, ссылками ``data:``.
    images: list[str] = []


class SettingsRequest(BaseModel):
    """Тело запроса на изменение настроек."""

    changes: dict[str, Any]


class WorldRequest(BaseModel):
    """Тело запроса на создание мира."""

    name: str
    brief: str = ""
    format: str = "story"
    preset_key: str | None = None


class WorldUpdate(BaseModel):
    """Правка полей мира."""

    changes: dict[str, Any]


class RuleRequest(BaseModel):
    """Новое правило."""

    body: str
    title: str = ""
    kind: str = "rule"
    priority: int = 100
    enabled: bool = True


class CharacterRequest(BaseModel):
    """Новая карточка персонажа."""

    name: str
    role: str = ""
    description: str = ""
    appearance: str = ""
    speech: str = ""
    enabled: bool = True


class SessionRequest(BaseModel):
    """Новая партия."""

    world_id: int
    title: str = ""


class GenerateRequest(BaseModel):
    """Запрос на генерацию сцен."""

    session_id: int
    scene_ids: list[int] | None = None
    #: Ступень качества на этот заход: fast, normal, quality или custom.
    quality: str | None = None


class MessageEdit(BaseModel):
    """Правка сообщения."""

    content: str


class RewindRequest(BaseModel):
    """Откат партии к сообщению."""

    message_id: int


class DraftRequest(BaseModel):
    """Черновая генерация."""

    prompt: str
    seed: int | None = None


class KillRequest(BaseModel):
    """Аварийная остановка."""

    stop_comfy: bool = True


class AgentRequest(BaseModel):
    """Указание агенту-редактору."""

    instruction: str
    preview: bool = False
    allow_destructive: bool = False
    #: Мир, открытый в интерфейсе: действия без явного номера мира идут в него.
    world_id: int | None = None


@app.post("/api/agent")
def api_agent(request: AgentRequest) -> dict[str, Any]:
    """Запускает агента-редактора в фоне.

    Агент может работать минутами: он думает, действует, смотрит на результат и
    думает снова. Поэтому запуск фоновый, а отчёт забирается через ``GET``.
    """
    instruction = request.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="пустое указание")
    if not _run_task(
        "agent",
        lambda: _machine.run_agent(
            instruction,
            preview=request.preview,
            allow_destructive=request.allow_destructive,
            world_id=request.world_id,
        ),
    ):
        return _busy_response()
    return {"accepted": True}


@app.get("/api/agent")
def api_agent_report() -> dict[str, Any]:
    """Отчёт последнего прогона агента и список доступных ему действий."""
    from novel.agent import WorldAgent

    agent = WorldAgent(_db, _machine.ft)
    return {
        "last_run": _machine.last_agent_run,
        "running": bool(_task.get("running") and _task.get("kind") == "agent"),
        "operations": [
            {
                "name": operation.name,
                "summary": operation.summary,
                "destructive": operation.destructive,
            }
            for operation in agent.operations.values()
        ],
    }


@app.post("/api/agent/stop")
def api_agent_stop() -> dict[str, Any]:
    """Просит агента остановиться после текущего шага.

    Прервать запрос к модели немедленно нельзя: ответ уже генерируется. Всё, что
    агент успел применить, остаётся в базе и попадает в отчёт.
    """
    _machine.request_agent_stop()
    return {"requested": True}


def _run_task(kind: str, target: Any) -> bool:
    """Запускает операцию в отдельном потоке, если ничего не выполняется.

    Исключение запоминается в состоянии задачи: поток умирает молча, и без этого
    пользователь увидел бы отчёт от прошлого прогона, а причина отказа осталась
    бы только в логе процесса.

    @param kind: метка операции для интерфейса.
    @param target: вызываемый объект без аргументов.
    @returns: ``False``, если уже что-то выполняется.
    """
    with _task_lock:
        if _task["running"]:
            return False
        _task.update(running=True, kind=kind, started_at=time.time(), error=None)

    def wrapper() -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001 — задача не должна исчезать молча
            with _task_lock:
                _task["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{kind}] ошибка: {type(exc).__name__}: {exc}", flush=True)
        finally:
            with _task_lock:
                _task["running"] = False
                _task["kind"] = None

    threading.Thread(target=wrapper, daemon=True).start()
    return True


def _busy_response() -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": "операция уже выполняется"})


def _world_or_404(world_id: int) -> Any:
    world = _db.world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail=f"мир {world_id} не найден")
    return world


def _session_or_404(session_id: int) -> Any:
    session = _db.session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"партия {session_id} не найдена")
    return session


def _scene_or_404(scene_id: int) -> Any:
    scene = _db.scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"сцена {scene_id} не найдена")
    return scene


# --- страница и наблюдаемость ------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Отдаёт страницу интерфейса.

    Заголовки запрещают кэширование: приложение локальное и правится часто, а без
    них браузер держит свою копию, не спрашивая сервер. Тогда любая правка
    интерфейса остаётся невидимой до ручной перезагрузки с обходом кэша, и
    сломанным выглядит исправленный интерфейс.

    В страницу подставляется метка сборки — по ней видно, какая версия открыта.
    """
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="не найден index.html")
    stamp = time.strftime("%d.%m %H:%M:%S", time.localtime(page.stat().st_mtime))
    return HTMLResponse(
        page.read_text(encoding="utf-8").replace("<!--ВЕРСИЯ-->", f"сборка {stamp}"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Состояние автомата, памяти и журнала."""
    doc = _machine.status()
    with _task_lock:
        doc["task"] = dict(_task)
        if _task["running"] and _task["started_at"]:
            doc["task"]["elapsed_s"] = round(time.time() - _task["started_at"], 1)
    return doc


@app.get("/api/debug")
def api_debug() -> dict[str, Any]:
    """Содержимое отладочной панели."""
    return _machine.debug_payload()


@app.get("/api/formats")
def api_formats() -> list[dict[str, Any]]:
    """Список форматов диалога."""
    return list_formats()


@app.get("/api/presets")
def api_presets() -> list[dict[str, Any]]:
    """Список готовых миров."""
    return list_presets()


# --- настройки и модели ------------------------------------------------------


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    """Текущие настройки вместе со ступенями качества кадра.

    Ступени отдаются сервером, а не записаны в странице: размер и шаги живут в
    одном месте — в настройках, — и интерфейс не разъедется с ними.
    """
    data = _settings_store.settings.to_dict()
    data["image_quality_options"] = [
        {"key": key, "size": size, "steps": steps, "title": title}
        for key, (size, steps, title) in IMAGE_QUALITY.items()
    ]
    return data


@app.post("/api/settings")
def api_set_settings(request: SettingsRequest) -> dict[str, Any]:
    """Применяет и сохраняет настройки."""
    rejected = _settings_store.settings.update(request.changes)
    _settings_store.save()
    if "reasoning_mode" in request.changes and not rejected:
        # Новый режим применится к следующему запросу, а не к уже начатому.
        _machine.invalidate_reasoning()
    engine_fields = {"engine_kind", "external_url", "external_server_path",
                     "external_manage", "external_context", "external_gpu_layers"}
    if engine_fields & set(request.changes) and not rejected:
        # Движок мог смениться целиком: у FreeToken и llama.cpp разные клиент,
        # контроллер и, главное, адрес.
        engine = _machine.sync_engine()
        _machine.invalidate_reasoning()
        return {"settings": _settings_store.settings.to_dict(),
                "rejected": rejected, "engine": engine}
    comfy_fields = {"comfy_root", "comfy_input_dir", "comfy_output_dir",
                    "comfy_server_script", "comfy_model_paths"}
    if comfy_fields & set(request.changes) and not rejected:
        # Пути к ComfyUI config читает на каждом обращении, поэтому достаточно
        # передать ему новые настройки: следующий кадр уйдёт по новому адресу.
        config.use_settings(_settings_store.settings)
        status = _machine.comfy_status() if hasattr(_machine, "comfy_status") else {}
        return {"settings": _settings_store.settings.to_dict(),
                "rejected": rejected, "comfy": status}
    if "models_dirs" in request.changes and not rejected:
        # Сменился каталог моделей — список надо перечитать с диска.
        config.use_settings(_settings_store.settings)
        _registry.all(refresh=True)
        found = len(_registry.all())
        return {"settings": _settings_store.settings.to_dict(),
                "rejected": rejected, "models_found": found}
    return {"settings": _settings_store.settings.to_dict(), "rejected": rejected}


@app.post("/api/comfy/check")
def api_comfy_check() -> dict[str, Any]:
    """Проверяет, что пути к ComfyUI ведут куда нужно.

    Пользователь задаёт путь вручную, и ошибка в нём иначе всплыла бы только
    при первом кадре — то есть через несколько минут ожидания. Проверка отвечает
    сразу: что найдено, что нет и чего не хватает.

    @returns: отчёт с флагом ``ok`` и разбором по каждому пути.
    """
    root = config.comfy_root()
    script = config.comfy_server_script()
    checks: list[dict[str, Any]] = []

    def add(title: str, path: Path, ok: bool, hint: str = "") -> None:
        checks.append({"title": title, "path": str(path), "ok": ok, "hint": hint})

    add("Каталог ComfyUI", root, root.is_dir(),
        "укажи каталог, где лежат main.py и .venv")
    add("Скрипт запуска", config.comfy_server_cwd() / script,
        (config.comfy_server_cwd() / script).is_file(),
        "путь к main.py от каталога ComfyUI")
    add("Интерпретатор", config.comfy_python(), config.comfy_python().is_file(),
        "внутри каталога ComfyUI должно быть окружение .venv")

    inputs = config.comfy_input_dir()
    add("Входные картинки", inputs, inputs.is_dir(),
        "каталог создастся сам при первой генерации")

    outputs = config.comfy_output_dir()
    add("Готовые кадры", outputs, outputs.is_dir(),
        "каталог создастся сам при первой генерации")

    # Сервер может быть уже поднят кем-то другим — это тоже рабочий случай.
    alive = False
    try:
        alive = ComfyClient().is_alive()
    except Exception:  # noqa: BLE001 — проверка не должна падать
        alive = False

    return {
        "ok": all(item["ok"] for item in checks[:3]),
        "checks": checks,
        "server_alive": alive,
        "url": config.COMFY_BASE_URL,
        "models": dict(config.QWEN_T2I_GRAPH_MODELS),
    }


@app.get("/api/models")
def api_models(refresh: bool = False) -> dict[str, Any]:
    """Модели под текущий движок и текущий выбор.

    Список подбирается под движок: у llama.cpp это GGUF, у FreeToken — каталоги
    HF и родной формат FTW. Показывать вперемешку оба набора незачем: половина
    строк заведомо не запустится, и выбор превращается в угадывание. Сколько
    моделей скрыто, сообщается отдельно — иначе непонятно, куда делись
    остальные.
    """
    engine = str(_settings_store.settings.engine_kind)
    external = engine == "external"
    models = _registry.as_dicts(refresh=refresh)

    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for model in models:
        fits = engine_for_kind(str(model["kind"])) == engine
        model["supported"] = fits
        model["note"] = (
            "GGUF: движок llama.cpp" if model["kind"] == "gguf"
            else "каталог движка FreeToken"
        )
        (visible if fits else hidden).append(model)

    # У подключённого сервера свои имена моделей: LM Studio ждёт
    # «имя-модели-в-нижнем-регистре», а не путь к файлу. Его список
    # идёт первым, потому что именно эти имена сервер и принимает.
    server_models: list[dict[str, Any]] = []
    if external:
        url = str(_settings_store.settings.external_url)
        if is_external_url_alive(url):
            for name in list_server_models(url):
                server_models.append({
                    "path": name, "name": name, "kind": "server",
                    "model_type": "внешний", "size_gb": 0.0,
                    "experts": None, "experts_per_tok": None, "quant": None,
                    "max_ctx": None, "supported": True,
                    "note": "модель внешнего сервера",
                })
    return {
        "models": server_models + visible,
        "hidden": len(hidden),
        "current": str(_settings_store.settings.model_path),
        "engine_kind": engine,
        "server_models": len(server_models),
        "supported": [model for model in server_models + visible if model["supported"]],
    }


class SelectModelRequest(BaseModel):
    """Выбор модели."""

    path: str
    apply_preset: bool = True


@app.post("/api/models/select")
def api_select_model(request: SelectModelRequest) -> dict[str, Any]:
    """Выбирает модель, подставляет пресет и перезапускает движок.

    Движок обслуживает один чекпоинт за запуск, поэтому смена модели без
    перезапуска ничего не даёт: выбор бы применился только при следующем ходе и
    выглядел бы как «кнопка не работает». Перезапуск идёт в фоне, интерфейс
    показывает его как обычную операцию.
    """
    model = _registry.by_path(request.path)
    external = str(_settings_store.settings.engine_kind) == "external"
    # Имена моделей внешнего сервера в реестре не лежат — их предлагает сам
    # сервер, и проверять их по диску нечем.
    from_server = model is None and external and is_external_url_alive(
        str(_settings_store.settings.external_url)
    ) and request.path in list_server_models(str(_settings_store.settings.external_url))
    if model is None and not from_server:
        raise HTTPException(status_code=404, detail="модель не найдена в реестре")

    settings = _settings_store.settings
    previous = settings.model_path

    # Запоминаем движок ДО выбора модели: выбор GGUF переводит проект с FreeToken
    # на llama.cpp, и это само по себе требует перезапуска — старый движок
    # обслуживает другой чекпоинт и другую модель вовсе.
    previous_engine = str(settings.engine_kind)

    # Движок выбирается по виду модели, а не отдельной галочкой: GGUF умеет
    # только llama.cpp, каталоги HF и родной FTW — только FreeToken. Иначе выбор
    # модели упирался бы в «включи движок сам», хотя выбрать нужно было ровно её.
    engine_note = ""
    if from_server:
        wanted = "external"
    elif model is not None:
        wanted = engine_for_kind(model.kind)
    else:
        wanted = str(settings.engine_kind)
    if wanted != str(settings.engine_kind):
        settings.engine_kind = wanted
        info = _machine.sync_engine()
        engine_note = (
            f"движок переключён на {'llama.cpp' if wanted == 'external' else 'FreeToken'}"
        )
        if wanted == "external" and not info.get("server"):
            engine_note = ("движок внешний, но рабочая сборка llama.cpp не найдена — "
                           "укажи путь в настройках")

    # Проверку «движок запущен» снимаем ДО смены настроек, но решение о
    # перезапуске принимаем и по смене движка тоже: порт у FreeToken и llama.cpp
    # один и тот же (1919), поэтому на нём может ещё висеть прежний процесс.
    # Без этого условия ensure_engine() видел бы живой чужой движок, отвечал бы
    # «уже поднят» и модель применялась бы только со второго раза.
    engine_running = _machine.controller.port_pid() is not None
    engine_changed = wanted != previous_engine

    settings.model_path = request.path if from_server else model.path
    applied: dict[str, Any] = {}
    if request.apply_preset and model is not None:
        applied = preset_for(model.model_type)
        settings.apply_model_preset(applied)
    _settings_store.save()
    _machine.invalidate_reasoning()

    restarted = False
    needs_restart = previous != settings.model_path or engine_changed
    if needs_restart and engine_running:
        restarted = _run_task("model_switch", _machine.restart_with_current_model)
    return {
        "settings": settings.to_dict(),
        "preset": applied,
        "engine_kind": str(settings.engine_kind),
        "engine_changed": engine_changed,
        "engine_note": engine_note,
        "engine_running": engine_running,
        "restart_started": restarted,
        "restart_needed": needs_restart,
    }


# --- аварийная остановка -----------------------------------------------------


@app.post("/api/kill")
def api_kill(request: KillRequest) -> dict[str, Any]:
    """Гасит движок и, если попросят, сервер ComfyUI."""
    try:
        return _machine.restart_all(stop_comfy=request.stop_comfy)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/engine/start")
def api_engine_start() -> dict[str, Any]:
    """Поднимает движок FreeToken в фоне."""
    if not _run_task("engine_start", _machine.ensure_engine):
        return _busy_response()
    return {"accepted": True}


@app.post("/api/engine/stop")
def api_engine_stop() -> dict[str, Any]:
    """Останавливает движок FreeToken."""
    try:
        return _machine.stop_engine()
    except (RuntimeError, FreeTokenError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- миры --------------------------------------------------------------------


@app.get("/api/worlds")
def api_worlds() -> list[dict[str, Any]]:
    """Список миров."""
    return [
        {
            **world.as_dict(),
            "format_title": get_format(world.format).title,
            # Здесь это счётчики, а не списки: список миров не должен тащить
            # содержимое каждого мира.
            "rules": len(_db.rules(world.id)),
            "characters": len(_db.characters(world.id)),
            "sessions": len(_db.sessions(world.id)),
        }
        for world in _db.worlds()
    ]


@app.post("/api/worlds")
def api_create_world(request: WorldRequest) -> dict[str, Any]:
    """Создаёт мир, при необходимости из пресета."""
    preset = get_preset(request.preset_key) if request.preset_key else None
    world_id = _db.create_world(
        name=request.name or (preset["title"] if preset else "Новый мир"),
        format=request.format if request.format in FORMATS else "story",
        brief=request.brief or (preset["brief"] if preset else ""),
        genre=preset["genre"] if preset else "",
        tone=preset["tone"] if preset else "",
        style=preset["style"] if preset else "",
        narrator=preset["narrator"] if preset else "",
    )
    if preset:
        for rule in preset["rules"]:
            _db.add_rule(world_id, rule["body"], title=rule["title"], kind="rule")
        for character in preset["characters"]:
            _db.add_character(
                world_id,
                character["name"],
                role=character["role"],
                description=character["description"],
                appearance=character["appearance"],
                speech=character["speech"],
            )
    session_id = _db.create_session(world_id, title="Первая партия")
    return {"world_id": world_id, "session_id": session_id}


@app.get("/api/worlds/{world_id}")
def api_world(world_id: int) -> dict[str, Any]:
    """Мир со всеми правилами и персонажами."""
    world = _world_or_404(world_id)
    return {
        "world": world.as_dict(),
        "rules": [
            {
                "id": rule.id,
                "title": rule.title,
                "body": rule.body,
                "kind": rule.kind,
                "enabled": bool(rule.enabled),
                "priority": rule.priority,
            }
            for rule in _db.rules(world_id)
        ],
        "characters": [
            {
                "id": item.id,
                "name": item.name,
                "role": item.role,
                "description": item.description,
                "appearance": item.appearance,
                "speech": item.speech,
                "enabled": bool(item.enabled),
            }
            for item in _db.characters(world_id)
        ],
        "sessions": [
            {"id": session.id, "title": session.title, "updated_at": session.updated_at}
            for session in _db.sessions(world_id)
        ],
        "items": [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "properties": item.properties,
                "quantity": item.quantity,
                #: ``None`` — вещь у игрока: отдельной записи персонажа для него нет.
                "character_id": item.character_id,
                "holder": (
                    "игрок" if item.is_player
                    else next(
                        (c.name for c in _db.characters(world_id) if c.id == item.character_id),
                        "неизвестный",
                    )
                ),
            }
            for item in _db.items(world_id)
        ],
    }


class ItemRequest(BaseModel):
    """Новая вещь."""

    name: str
    character_id: int | None = None
    description: str = ""
    properties: str = ""
    quantity: int = 1


@app.post("/api/worlds/{world_id}/items")
def api_add_item(world_id: int, request: ItemRequest) -> dict[str, Any]:
    """Кладёт вещь в мир.

    Владелец — персонаж по номеру либо игрок, если номер не указан. Ведущий о
    вещи узнает, но распоряжаться ею не станет: это оговорено в промпте.
    """
    _world_or_404(world_id)
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="у вещи нет названия")
    if request.character_id is not None and _db.character(request.character_id) is None:
        raise HTTPException(status_code=404, detail="владелец не найден")
    item_id = _db.add_item(
        world_id,
        name,
        character_id=request.character_id,
        description=request.description,
        properties=request.properties,
        quantity=request.quantity,
    )
    return {"item_id": item_id}


@app.delete("/api/items/{item_id}")
def api_delete_item(item_id: int) -> dict[str, Any]:
    """Убирает вещь из мира."""
    if _db.item(item_id) is None:
        raise HTTPException(status_code=404, detail="вещь не найдена")
    _db.delete_item(item_id)
    return {"deleted": True}


@app.post("/api/worlds/{world_id}")
def api_update_world(world_id: int, request: WorldUpdate) -> dict[str, Any]:
    """Правит поля мира."""
    _world_or_404(world_id)
    rejected = _db.update_world(world_id, request.changes)
    return {"rejected": rejected}


class DuplicateRequest(BaseModel):
    """Копирование мира."""

    with_sessions: bool = False


@app.post("/api/worlds/{world_id}/duplicate")
def api_duplicate_world(world_id: int, request: DuplicateRequest) -> dict[str, Any]:
    """Копирует мир — для отладки: ломать копию, а не рабочий мир.

    Файлы изображений не дублируются: оба мира ссылаются на одни и те же, и
    уборка мусора считает файл нужным, пока на него ссылается хоть одна запись.
    """
    _world_or_404(world_id)
    try:
        copy_id, sessions = _db.duplicate_world(world_id, with_sessions=request.with_sessions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"world_id": copy_id, "session_ids": sessions}


@app.delete("/api/worlds/{world_id}")
def api_delete_world(world_id: int) -> dict[str, Any]:
    """Удаляет мир со всем содержимым: правила, персонажей, вещи, места, партии."""
    _world_or_404(world_id)
    _db.delete_world(world_id)
    return {"deleted": world_id}


@app.post("/api/worlds/{world_id}/clear")
def api_clear_world(world_id: int) -> dict[str, Any]:
    """Стирает историю всех партий мира, оставляя сам мир и его настройки.

    Нужно, чтобы начать заново в том же мире: правила, персонажи, вещи и места
    остаются, а сообщения, кадры, сводки и внешность исчезают.
    """
    _world_or_404(world_id)
    sessions = _db.sessions(world_id)
    # Первую партию очищаем, лишние удаляем: после «стереть историю» в мире
    # должна остаться ровно одна пустая партия, а не гора пустых.
    for index, session in enumerate(sessions):
        if index == 0:
            _db.clear_session(session.id)
        else:
            _db.delete_session(session.id)
    if sessions:
        return {"cleared": len(sessions), "session_id": sessions[0].id}
    return {"cleared": 0, "session_id": _db.create_session(world_id, "Первая партия")}


@app.post("/api/worlds/{world_id}/structure")
def api_structure_world(world_id: int) -> dict[str, Any]:
    """Разбирает описание мира на структуру силами модели."""
    _world_or_404(world_id)
    if not _run_task("structure", lambda: _machine.structure_world(world_id)):
        return _busy_response()
    return {"accepted": True}


@app.post("/api/worlds/{world_id}/structure/apply")
def api_structure_world_sync(world_id: int) -> dict[str, Any]:
    """То же, но синхронно — удобно для отладки из скриптов."""
    _world_or_404(world_id)
    try:
        return _machine.structure_world(world_id)
    except (RuntimeError, FreeTokenError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/worlds/{world_id}/rules")
def api_add_rule(world_id: int, request: RuleRequest) -> dict[str, Any]:
    """Добавляет правило."""
    _world_or_404(world_id)
    rule_id = _db.add_rule(
        world_id, request.body, title=request.title, kind=request.kind, priority=request.priority
    )
    if not request.enabled:
        _db.update_rule(rule_id, {"enabled": 0})
    return {"rule_id": rule_id}


@app.post("/api/rules/{rule_id}")
def api_update_rule(rule_id: int, request: WorldUpdate) -> dict[str, Any]:
    """Правит правило."""
    return {"rejected": _db.update_rule(rule_id, request.changes)}


@app.delete("/api/rules/{rule_id}")
def api_delete_rule(rule_id: int) -> dict[str, Any]:
    """Удаляет правило."""
    _db.delete_rule(rule_id)
    return {"deleted": rule_id}


@app.post("/api/worlds/{world_id}/characters")
def api_add_character(world_id: int, request: CharacterRequest) -> dict[str, Any]:
    """Добавляет персонажа."""
    _world_or_404(world_id)
    character_id = _db.add_character(
        world_id,
        request.name,
        role=request.role,
        description=request.description,
        appearance=request.appearance,
        speech=request.speech,
    )
    if not request.enabled:
        _db.update_character(character_id, {"enabled": 0})
    return {"character_id": character_id}


@app.post("/api/characters/{character_id}")
def api_update_character(character_id: int, request: WorldUpdate) -> dict[str, Any]:
    """Правит карточку персонажа."""
    return {"rejected": _db.update_character(character_id, request.changes)}


@app.delete("/api/characters/{character_id}")
def api_delete_character(character_id: int) -> dict[str, Any]:
    """Удаляет персонажа."""
    _db.delete_character(character_id)
    return {"deleted": character_id}


# --- партии ------------------------------------------------------------------


@app.get("/api/sessions")
def api_sessions(world_id: int | None = None) -> list[dict[str, Any]]:
    """Список партий."""
    return [
        {
            "id": session.id,
            "world_id": session.world_id,
            "title": session.title,
            "messages": _db.count_messages(session.id),
            "updated_at": session.updated_at,
        }
        for session in _db.sessions(world_id)
    ]


@app.post("/api/sessions")
def api_create_session(request: SessionRequest) -> dict[str, Any]:
    """Создаёт партию в мире."""
    _world_or_404(request.world_id)
    session_id = _db.create_session(request.world_id, request.title or "Новая партия")
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: int) -> dict[str, Any]:
    """Удаляет партию."""
    _session_or_404(session_id)
    _db.delete_session(session_id)
    return {"deleted": session_id}


class RenameRequest(BaseModel):
    """Переименование партии."""

    title: str


class ImportRequest(BaseModel):
    """Импорт выгрузки мира или партии."""

    payload: dict[str, Any]
    world_id: int | None = None


@app.post("/api/sessions/{session_id}/rename")
def api_rename_session(session_id: int, request: RenameRequest) -> dict[str, Any]:
    """Переименовывает партию."""
    _session_or_404(session_id)
    _db.rename_session(session_id, request.title)
    return {"session_id": session_id, "title": request.title}


@app.get("/api/sessions/{session_id}/stats")
def api_session_stats(session_id: int) -> dict[str, Any]:
    """Сводка по партии: объём, время, расход токенов."""
    _session_or_404(session_id)
    return _db.session_stats(session_id)


def _safe_filename(name: str, fallback: str) -> str:
    """Готовит имя файла для заголовка ответа.

    Заголовки HTTP кодируются в latin-1, поэтому кириллица в имени файла рвёт
    ответ. Всё, что не ASCII, заменяется подчёркиванием.

    @param name: исходное имя.
    @param fallback: имя, если не осталось ни одного допустимого символа.
    @returns: безопасное имя файла.
    """
    safe = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in name
    ).strip("_")
    return safe or fallback


@app.get("/api/sessions/{session_id}/export")
def api_export_session(session_id: int) -> JSONResponse:
    """Отдаёт партию файлом JSON."""
    session = _session_or_404(session_id)
    payload = _db.export_session(session_id)
    safe = _safe_filename(session.title, "session")
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="novelforge-{safe}.json"'},
    )


@app.get("/api/worlds/{world_id}/export")
def api_export_world(world_id: int) -> JSONResponse:
    """Отдаёт мир со всеми партиями файлом JSON."""
    world = _world_or_404(world_id)
    payload = _db.export_world(world_id)
    safe = _safe_filename(world.name, "world")
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="novelforge-{safe}.json"'},
    )


@app.get("/api/worlds/{world_id}/bundle")
def api_export_bundle(world_id: int) -> FileResponse:
    """Отдаёт мир пакетом: JSON и изображения в одном архиве.

    Обычная выгрузка несёт только пути к файлам, поэтому на другой машине мир
    остался бы без картинок. Пакет переносит и их.
    """
    from novel.bundle import export_bundle

    world = _world_or_404(world_id)
    safe = _safe_filename(world.name, "world")
    target = config.DATA_DIR / "bundles" / f"novelforge-{safe}.zip"
    try:
        report = export_bundle(_db, world_id, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        report.path,
        media_type="application/zip",
        filename=f"novelforge-{safe}.zip",
    )


@app.post("/api/import-bundle")
async def api_import_bundle(request: Request) -> dict[str, Any]:
    """Загружает мир из ZIP-пакета.

    Тело запроса — сам архив: разбирать multipart ради одного файла незачем.
    """
    from novel.bundle import import_bundle

    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="пустой архив")
    incoming = config.DATA_DIR / "bundles" / f"incoming-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_bytes(data)
    try:
        report = import_bundle(_db, incoming)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        incoming.unlink(missing_ok=True)
    return report.as_dict()


@app.get("/api/uploads/orphans")
def api_orphan_uploads() -> dict[str, Any]:
    """Показывает файлы NovelForge, на которые уже никто не ссылается.

    Удаление партии убирает записи из базы, но не файлы: они лежат вне неё.
    Отдельно считаются вложения, сгенерированные кадры и копии образцов в
    каталоге входов ComfyUI.
    """
    from novel.bundle import orphan_files

    groups = orphan_files(_db)
    labels = {"uploads": "вложения", "frames": "кадры", "staged": "копии образцов"}
    detail = []
    total = 0
    total_mb = 0.0
    for kind, paths in groups.items():
        size = sum(path.stat().st_size for path in paths)
        total += len(paths)
        total_mb += size
        if paths:
            detail.append({
                "kind": kind,
                "label": labels.get(kind, kind),
                "count": len(paths),
                "mb": round(size / (1024 * 1024), 2),
                "files": [path.name for path in paths[:20]],
            })
    return {
        "count": total,
        "total_mb": round(total_mb / (1024 * 1024), 2),
        "groups": detail,
        "files": [
            {"name": path.name, "mb": round(path.stat().st_size / (1024 * 1024), 2)}
            for paths in groups.values() for path in paths[:100]
        ],
    }


@app.delete("/api/uploads/orphans")
def api_purge_orphan_uploads() -> dict[str, Any]:
    """Удаляет осиротевшие файлы с диска."""
    from novel.bundle import purge_orphans

    return purge_orphans(_db)


@app.post("/api/maintenance/compact")
def api_compact_database() -> dict[str, Any]:
    """Сжимает базу и переносит журнал в основной файл.

    SQLite в режиме WAL копит изменения в отдельном файле, который может стать
    больше самой базы. Пока интерфейс работает, соединение открыто и перенос не
    случается сам.
    """
    before = {
        name: (config.DB_PATH.parent / name).stat().st_size
        for name in ("novel.db", "novel.db-wal", "novel.db-shm")
        if (config.DB_PATH.parent / name).exists()
    }
    _db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _db.conn.execute("VACUUM")
    _db.conn.commit()
    after = {
        name: (config.DB_PATH.parent / name).stat().st_size
        for name in ("novel.db", "novel.db-wal", "novel.db-shm")
        if (config.DB_PATH.parent / name).exists()
    }
    return {
        "before_mb": round(sum(before.values()) / (1024 * 1024), 2),
        "after_mb": round(sum(after.values()) / (1024 * 1024), 2),
    }


@app.get("/api/sessions/{session_id}/weight")
def api_session_weight(session_id: int) -> dict[str, Any]:
    """Сколько места занимает партия: текст против картинок."""
    from novel.bundle import session_weight

    if _db.session(session_id) is None:
        raise HTTPException(status_code=404, detail="партия не найдена")
    return session_weight(_db, session_id)


@app.post("/api/import")
def api_import(request: ImportRequest) -> dict[str, Any]:
    """Загружает выгрузку мира или партии.

    Мир создаётся новым; партия добавляется в указанный мир или в первый
    подходящий.
    """
    kind = (request.payload or {}).get("kind")
    try:
        if kind == "novelforge.world":
            world_id, sessions = _db.import_world(request.payload)
            return {"world_id": world_id, "session_ids": sessions}
        if kind == "novelforge.session":
            world_id = request.world_id
            if world_id is None:
                worlds = _db.worlds()
                if not worlds:
                    raise HTTPException(status_code=400, detail="сначала создай мир")
                world_id = worlds[0].id
            _world_or_404(world_id)
            session_id = _db.import_session(world_id, request.payload)
            return {"world_id": world_id, "session_ids": [session_id]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="неизвестный тип выгрузки")


@app.post("/api/backup")
def api_backup() -> dict[str, Any]:
    """Делает копию файла базы рядом с оригиналом.

    Сохранение идёт на каждый ход: SQLite фиксирует изменения сразу. Копия нужна
    на случай порчи файла или неудачного эксперимента с миром.
    """
    import shutil

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = config.DATA_DIR / "backups" / f"novel-{stamp}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _db.conn.commit()
        shutil.copy2(_db.path, target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"не удалось сделать копию: {exc}") from exc
    return {
        "path": str(target),
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
        "counts": _db.stats(),
    }


@app.get("/api/backups")
def api_backups() -> list[dict[str, Any]]:
    """Список сделанных копий базы."""
    folder = config.DATA_DIR / "backups"
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.glob("novel-*.db"), reverse=True)[:20]:
        items.append(
            {
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "modified": path.stat().st_mtime,
            }
        )
    return items


@app.post("/api/sessions/{session_id}/clear")
def api_clear_session(session_id: int) -> dict[str, Any]:
    """Очищает партию, оставляя мир."""
    _session_or_404(session_id)
    _db.clear_session(session_id)
    return {"cleared": session_id}


@app.post("/api/sessions/{session_id}/summarize")
def api_summarize(session_id: int) -> dict[str, Any]:
    """Сворачивает историю в суммаризацию."""
    _session_or_404(session_id)
    _machine.ensure_engine()
    try:
        report = _machine.context.summarize(session_id)
    except FreeTokenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return report or {"skipped": "история ещё короткая"}


@app.get("/api/sessions/{session_id}/validate")
def api_validate(session_id: int) -> dict[str, Any]:
    """Проверки мира и партии."""
    _session_or_404(session_id)
    return _machine.validate(session_id)


@app.post("/api/sessions/{session_id}/rewind")
def api_rewind(session_id: int, request: RewindRequest) -> dict[str, Any]:
    """Откатывает партию к сообщению.

    Вместе с сообщениями уходят сцены этих ходов и сводки, которые их описывали.
    Файлы кадров остаются на диске: база их не хранит, и удалять чужие файлы
    молча нельзя — их находит уборка мусора.
    """
    _session_or_404(session_id)
    return _db.rewind_to(session_id, request.message_id)


@app.get("/api/sessions/{session_id}/history")
def api_history(session_id: int) -> dict[str, Any]:
    """Сообщения, сцены и состояние мира партии."""
    session = _session_or_404(session_id)
    world = _db.world(session.world_id)
    fmt = get_format(world.format if world else "story")
    return {
        "session": {"id": session.id, "title": session.title, "world_id": session.world_id},
        "world": {"id": world.id, "name": world.name, "format": fmt.key} if world else None,
        "format": fmt.as_dict(),
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "kind": message.kind,
                "tokens": message.tokens,
                "ts": message.ts,
                "attachments": len(message.attachment_paths),
            }
            for message in _db.messages(session_id)
        ],
        "scenes": [
            {
                "id": scene.id,
                "message_id": scene.message_id,
                "prompt": scene.prompt,
                "raw_description": scene.raw_description,
                "seed": scene.seed,
                "path": scene.path,
                "status": scene.status,
                "elapsed_s": scene.elapsed_s,
                "has_image": bool(scene.path and Path(scene.path).exists()),
                "location_id": scene.location_id,
                "location": (place.name if (place := _db.location(scene.location_id)) else None),
                "used_reference": bool(scene.used_reference),
                "is_reference": bool(
                    scene.location_id
                    and (place := _db.location(scene.location_id))
                    and place.reference_path == scene.path
                ),
            }
            for scene in _db.scenes(session_id)
        ],
        "memories": [
            {"id": memory.id, "through_message_id": memory.through_message_id,
             "summary": memory.summary, "tokens": memory.tokens}
            for memory in _db.memories(session_id)
        ],
        "state": _db.world_state(session_id),
        "notes": [
            {
                "id": note.id,
                "text": note.text,
                "source": note.source,
                "horizon": note.horizon,
                # Возраст в ходах: по нему видно, какая задумка залежалась.
                "age": max(0, _db.count_messages(session_id)
                           - (note.anchor_message_id or 0)) // 2,
                "created_at": note.created_at,
            }
            for note in _db.notes(session_id, limit=20)
        ],
        "note_limits": {
            "player": PLAYER_NOTE_LIMIT,
            "model": MODEL_NOTE_LIMIT,
            "used_player": _db.count_notes(session_id, "player"),
            "used_model": _db.count_notes(session_id, "model"),
        },
    }


class NoteRequest(BaseModel):
    """Заметка на потом."""

    text: str = ""


class LookRequest(BaseModel):
    """Правка внешности «сейчас»."""

    forget: str = ""
    name: str = ""
    look: str = ""


@app.post("/api/sessions/{session_id}/looks")
def api_edit_look(session_id: int, request: LookRequest) -> dict[str, Any]:
    """Записывает или забывает, как персонаж выглядит сейчас.

    Ведущий сообщает об этом блоком ``looks``, но модель бывает занята другим и
    блок пропускает. Тогда внешность вписывается руками — результат тот же.

    @param session_id: партия.
    @param request: ``forget`` — кого забыть, либо ``name`` и ``look`` — что записать.
    @returns: новое содержимое блока внешности.
    """
    _session_or_404(session_id)
    looks = dict(_db.get_state(session_id, "looks", {}) or {})
    if request.forget.strip():
        looks.pop(request.forget.strip(), None)
    elif request.name.strip() and request.look.strip():
        looks[request.name.strip()] = " ".join(request.look.split())
    else:
        raise HTTPException(status_code=400, detail="нужно имя и описание либо кого забыть")
    _db.set_state(session_id, "looks", looks)
    return {"looks": looks}


@app.post("/api/sessions/{session_id}/notes")
def api_add_note(session_id: int, request: NoteRequest) -> dict[str, Any]:
    """Заводит заметку на потом: что должно случиться позже."""
    _session_or_404(session_id)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="пустая заметка")
    if _db.count_notes(session_id, "player") >= PLAYER_NOTE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"своих задумок уже {PLAYER_NOTE_LIMIT} — "
                   "пусть сбудутся или убери лишние",
        )
    note_id = _db.add_note(
        session_id, text, source="player",
        anchor_message_id=_db.count_messages(session_id),
    )
    return {"note_id": note_id}


class InventRequest(BaseModel):
    """Заказ ведущему придумать, куда может пойти история."""

    horizon: int = DEFAULT_NOTE_HORIZON


@app.post("/api/sessions/{session_id}/invent")
def api_invent_notes(session_id: int, request: InventRequest) -> dict[str, Any]:
    """Просит ведущего придумать задумки на несколько ходов вперёд."""
    _session_or_404(session_id)
    if not _run_task(
        "invent",
        lambda: _machine.invent_notes(session_id, request.horizon),
    ):
        return _busy_response()
    return {"accepted": True}


class NoteEdit(BaseModel):
    """Правка текста заметки."""

    text: str = ""


@app.post("/api/notes/{note_id}")
def api_edit_note(note_id: int, request: NoteEdit) -> dict[str, Any]:
    """Переписывает заметку: своя мысль важнее, чем вышло с первого раза."""
    note = _db.note(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="заметка не найдена")
    if not _db.set_note_text(note_id, request.text):
        raise HTTPException(status_code=400, detail="пустой текст")
    return {"note_id": note_id, "text": request.text.strip()}


@app.post("/api/notes/{note_id}/close")
def api_close_note(note_id: int, status: str = "done") -> dict[str, Any]:
    """Отмечает заметку сбывшейся или снятой — ведущий её больше не увидит."""
    if _db.note(note_id) is None:
        raise HTTPException(status_code=404, detail="заметка не найдена")
    _db.close_note(note_id, status if status in ("done", "dropped") else "done")
    return {"closed": True}


@app.delete("/api/notes/{note_id}")
def api_delete_note(note_id: int) -> dict[str, Any]:
    """Удаляет заметку насовсем."""
    if _db.note(note_id) is None:
        raise HTTPException(status_code=404, detail="заметка не найдена")
    _db.delete_note(note_id)
    return {"deleted": True}


@app.delete("/api/sessions/{session_id}/notes")
def api_delete_notes(session_id: int, source: str = "all") -> dict[str, Any]:
    """Удаляет заметки партии разом.

    Задумки копятся и мешают; чистить их по одной — десять нажатий и десять
    подтверждений. Здесь выбор называет сам себя: ``player``, ``model``,
    ``finished`` или ``all``.

    @param session_id: партия.
    @param source: что именно стереть.
    @returns: сколько удалено.
    """
    _session_or_404(session_id)
    if source == "player":
        removed = _db.delete_notes(session_id, "player")
    elif source == "model":
        removed = _db.delete_notes(session_id, "model")
    elif source == "finished":
        removed = _db.delete_finished_notes(session_id)
    elif source == "all":
        removed = _db.delete_notes(session_id)
    else:
        raise HTTPException(
            status_code=400,
            detail="можно стереть player, model, finished или all",
        )
    return {"deleted": removed, "left": len(_db.notes(session_id, limit=100))}


# --- ход и генерация ---------------------------------------------------------


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> dict[str, Any]:
    """Принимает реплику игрока и запускает ход в фоне.

    Реплика может быть пустой, если задана врезка: это ход, в котором игрок не
    действует, а только направляет сцену.
    """
    text = request.text.strip()
    interject = (request.interject or "").strip()
    if not text and not interject:
        raise HTTPException(status_code=400, detail="пустая реплика")
    _session_or_404(request.session_id)
    if not _run_task(
        "chat",
        lambda: _machine.send(
            request.session_id, text, interject=interject, images=request.images
        ),
    ):
        return _busy_response()
    return {"accepted": True}


@app.get("/api/worlds/{world_id}/locations")
def api_locations(world_id: int) -> list[dict[str, Any]]:
    """Постоянные места мира и их кадры-образцы."""
    _world_or_404(world_id)
    return [
        {
            "id": item.id,
            "name": item.name,
            "prompt": item.prompt,
            "style": item.style,
            "visits": item.visits,
            "has_reference": bool(item.reference_path and Path(item.reference_path).exists()),
            "reference": item.reference_path,
        }
        for item in _db.locations(world_id)
    ]


class ReferenceRequest(BaseModel):
    """Назначение кадра-образца."""

    scene_id: int | None = None


@app.post("/api/locations/{location_id}/reference")
def api_set_location_reference(location_id: int, request: ReferenceRequest) -> dict[str, Any]:
    """Назначает образец места: кадр сцены или снятие образца.

    Образец задаёт внешний вид места. По умолчанию им становится первый удачный
    кадр, но если он вышел неудачным, образцом можно сделать любой другой.
    """
    location = _db.location(location_id)
    if location is None:
        raise HTTPException(status_code=404, detail="место не найдено")
    if request.scene_id is None:
        _db.set_location_reference(location_id, None)
        return {"location_id": location_id, "reference": None}
    scene = _db.scene(request.scene_id)
    if scene is None or not scene.path or not Path(scene.path).exists():
        raise HTTPException(status_code=404, detail="у сцены нет готового кадра")
    _db.set_location_reference(location_id, scene.path)
    return {"location_id": location_id, "reference": scene.path}


@app.get("/api/locations/reference/{location_id}")
def api_location_reference(location_id: int) -> FileResponse:
    """Отдаёт кадр-образец места."""
    location = _db.location(location_id)
    if location is None or not location.reference_path:
        raise HTTPException(status_code=404, detail="у места нет образца")
    path = Path(location.reference_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="файл образца пропал с диска")
    return FileResponse(path)


@app.post("/api/comfy/start")
def api_comfy_start() -> dict[str, Any]:
    """Поднимает сервер ComfyUI напрямую, без Comfy Desktop."""
    if not _run_task("comfy_start", _machine.ensure_comfy):
        return _busy_response()
    return {"accepted": True}


@app.post("/api/generate")
def api_generate(request: GenerateRequest) -> dict[str, Any]:
    """Запускает генерацию сцен."""
    _session_or_404(request.session_id)
    if not _run_task(
        "images",
        lambda: _machine.generate_pending_now(
            request.session_id, request.scene_ids, request.quality
        ),
    ):
        return _busy_response()
    return {"accepted": True}


class ScenePromptRequest(BaseModel):
    """Правка описания кадра и его перерисовка."""

    prompt: str = ""
    quality: str | None = None


@app.post("/api/scenes/{scene_id}/prompt")
def api_set_scene_prompt(scene_id: int, request: ScenePromptRequest) -> dict[str, Any]:
    """Записывает новое описание кадра и ставит его в очередь на перерисовку.

    Описание уходит генератору как есть, минуя ведущего. Это единственный способ
    нарисовать то, что модель писать отказывается.
    """
    _scene_or_404(scene_id)
    prompt = " ".join(request.prompt.split())
    if not prompt:
        raise HTTPException(status_code=400, detail="пустое описание")
    _db.set_scene_prompt(scene_id, prompt)
    return {"scene_id": scene_id, "prompt": prompt, "status": "pending"}


@app.post("/api/scenes/{scene_id}/redraw")
def api_redraw_scene(scene_id: int, request: ScenePromptRequest) -> dict[str, Any]:
    """Перерисовывает один кадр, при необходимости с другим качеством."""
    scene = _scene_or_404(scene_id)
    if request.prompt.strip():
        _db.set_scene_prompt(scene_id, " ".join(request.prompt.split()))
    else:
        _db.set_scene_prompt(scene_id, scene.prompt)
    if not _run_task(
        "images",
        lambda: _machine.generate_pending_now(scene.session_id, [scene_id], request.quality),
    ):
        return _busy_response()
    return {"accepted": True, "scene_id": scene_id}


@app.post("/api/draft")
def api_draft(request: DraftRequest) -> dict[str, Any]:
    """Черновая генерация малым разрешением."""
    try:
        return _machine.generate_draft(request.prompt, request.seed)
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/messages/{message_id}")
def api_edit_message(message_id: int, request: MessageEdit) -> dict[str, Any]:
    """Правит текст сообщения — ручная починка ответа модели."""
    _db.update_message(message_id, request.content)
    return {"updated": message_id}


@app.delete("/api/messages/{message_id}")
def api_delete_message(message_id: int) -> dict[str, Any]:
    """Удаляет сообщение."""
    _db.delete_message(message_id)
    return {"deleted": message_id}


@app.delete("/api/scenes/{scene_id}")
def api_delete_scene(scene_id: int) -> dict[str, Any]:
    """Удаляет запись о сцене."""
    _db.delete_scene(scene_id)
    return {"deleted": scene_id}


@app.get("/api/image/{scene_id}")
def api_image(scene_id: int) -> FileResponse:
    """Отдаёт файл картинки для сцены."""
    scene = _db.scene(scene_id)
    if scene is None or not scene.path:
        raise HTTPException(status_code=404, detail="картинка не найдена")
    path = Path(scene.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="файл картинки пропал с диска")
    return FileResponse(path)


@app.get("/api/messages/{message_id}/attachment/{index}")
def api_attachment(message_id: int, index: int) -> FileResponse:
    """Отдаёт изображение, приложенное игроком к сообщению.

    Путь берётся только из записи в базе и сверяется с каталогом загрузок:
    подставлять произвольный путь из запроса нельзя.
    """
    row = _db.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="сообщение не найдено")
    paths = json.loads(row["attachments"] or "[]")
    if not isinstance(paths, list) or index >= len(paths):
        raise HTTPException(status_code=404, detail="вложение не найдено")
    path = Path(str(paths[index])).resolve()
    if config.UPLOADS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="файл вложения недоступен")
    return FileResponse(path)
