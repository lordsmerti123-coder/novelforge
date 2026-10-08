"""Конечный автомат, связывающий текстовую модель и генератор изображений.

Автомат существует из-за измеренной цифры: на 12 GB VRAM FreeToken и ComfyUI не
помещаются одновременно. С работающим движком свободно максимум ~2.9 GB, а
Qwen-Image-2.1 нужно около 8 GB. Поэтому ход с картинкой выглядит так:

    IDLE -> LLM_ACTIVE -> SWITCH_TO_IMAGE -> IMAGE_GEN -> SWITCH_TO_LLM -> IDLE

``SWITCH_TO_IMAGE`` останавливает движок целиком (это освобождает ~9.8 GB за
0.75 с), ``SWITCH_TO_LLM`` поднимает его заново (~44 с до первого токена).
Картинки поэтому копятся в очередь и рисуются пачкой за одно переключение.

Сервер ComfyUI живёт постоянно — его поднимает пользователь один раз. Автомат
управляет только выгрузкой моделей из VRAM через ``POST /free``.
"""

from __future__ import annotations

import random
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from novel import config, metrics, prompts
from novel.comfy import ComfyClient, ComfyError, build_t2i_graph, stage_reference
from novel.context import AssembledPrompt, ContextBuilder
from novel.db import Character, NovelDB
from novel.engine import EngineConfig, EngineController
from novel.external import (
    ExternalConfig,
    ExternalController,
    find_server,
    is_external_url_alive,
    list_server_models,
)
from novel.formats import get_format
from novel.freetoken import FreeTokenClient, FreeTokenError
from novel.models import ModelRegistry
from novel.protocol import ParsedReply, SceneSpec, loads_json, looks_from_text, parse_reply
from novel.settings import SettingsStore
from novel.vision import prepare_image, save_upload, user_message

#: Потолки задумок на потом. У игрока их больше, и они главнее: выдумки
#: ведущего не должны вытеснять то, что задумал человек.
PLAYER_NOTE_LIMIT = 4
MODEL_NOTE_LIMIT = 3
#: Сколько ходов вперёд ведущий задумывает по умолчанию.
DEFAULT_NOTE_HORIZON = 5


class State(str, Enum):
    """Состояния автомата."""

    IDLE = "IDLE"
    LLM_ACTIVE = "LLM_ACTIVE"
    SWITCH_TO_IMAGE = "SWITCH_TO_IMAGE"
    IMAGE_GEN = "IMAGE_GEN"
    SWITCH_TO_LLM = "SWITCH_TO_LLM"
    SUMMARIZING = "SUMMARIZING"
    ERROR = "ERROR"


@dataclass
class TurnOutcome:
    """Результат одного хода игрока."""

    prose: str = ""
    parsed: ParsedReply = field(default_factory=lambda: ParsedReply(prose="", raw=""))
    timings: dict[str, float] = field(default_factory=dict)
    scene_ids: list[int] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    total_s: float = 0.0
    assistant_message_id: int | None = None
    #: Место в этом ходу встретилось впервые.
    new_place: bool = False
    #: Персонажи, которых в кадре ещё не было.
    new_characters: list[str] = field(default_factory=list)
    #: Ведущий отметил резкую перемену обстановки на знакомом месте.
    sudden: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Представление для интерфейса."""
        return {
            "prose": self.prose,
            "scene_ids": self.scene_ids,
            "images": self.images,
            "errors": self.errors,
            "summary": self.summary,
            "timings": {key: round(value, 2) for key, value in self.timings.items()},
            "total_s": round(self.total_s, 2),
            "assistant_message_id": self.assistant_message_id,
            "new_place": self.new_place,
            "new_characters": self.new_characters,
            "sudden": self.sudden,
        }


def _looks_english(text: str) -> bool:
    """Похож ли текст на английский — годится ли он генератору изображений.

    Генератор понимает только английский. Внешность персонажа пишут по-русски, и
    подстановка её в промпт даёт «full body shot of цифровой образ»: кадр
    получается мусорным. Поэтому текст проверяется до отправки.

    @param text: проверяемый текст.
    @returns: ``True``, если кириллицы в тексте нет и он непустой.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return not any("\u0400" <= char <= "\u04ff" for char in stripped)


class NovelMachine:
    """Оркестратор: один движок, один сервер ComfyUI, много миров и партий."""

    def __init__(
        self,
        db: NovelDB,
        settings_store: SettingsStore,
        registry: ModelRegistry | None = None,
        *,
        ft: FreeTokenClient | None = None,
        comfy: ComfyClient | None = None,
        controller: EngineController | None = None,
    ) -> None:
        self.db = db
        self.settings_store = settings_store
        self.registry = registry or ModelRegistry()
        self.ft = ft or FreeTokenClient()
        self.comfy = comfy or ComfyClient()
        self.controller = controller or EngineController(port=1919)
        # Выбор движка делается до конструктора контекста: тот запоминает клиента
        # у себя и по нему считает токены.
        if ft is None and controller is None:
            self.sync_engine()
        self.context = ContextBuilder(
            db, self.ft, lambda: self.settings_store.settings, self._load_images
        )
        self.state = State.IDLE
        self.log: deque[str] = deque(maxlen=400)
        self.last_outcome: TurnOutcome | None = None
        #: Отчёт последнего прогона агента-редактора.
        self.last_agent_run: dict[str, Any] | None = None
        self.debug: dict[str, Any] = {}
        self._busy = threading.Lock()
        self._reasoning_ready = False
        #: Просьба остановить агента: выставляется, когда движок выгружают.
        self._agent_cancel = threading.Event()
        #: Момент перехода в текущую фазу — для показа «сколько уже длится».
        self._phase_started = time.time()
        #: Начало текущего хода игрока; ``None`` в покое.
        self.turn_started: float | None = None
        #: Сцена, которая рисуется сейчас, и очередь за ней.
        self._current_scene: dict[str, Any] | None = None
        #: Номера сцен, ожидающих генерации в этом заходе.
        self._scene_queue: list[int] = []
        #: Держит ли ComfyUI модели в памяти — видно в интерфейсе.
        self.comfy_models_loaded = False
        #: Подготовленные изображения по пути: повторная сборка запроса не должна
        #: заново уменьшать и кодировать те же файлы.
        self._image_cache: dict[str, str] = {}

    # --- журнал и наблюдаемость --------------------------------------------

    def note(self, message: str) -> None:
        """Добавляет строку в журнал, который показывает интерфейс."""
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"{stamp}  {message}")
        print(f"[{stamp}] {message}", flush=True)

    def _set_state(self, state: State) -> None:
        """Переводит автомат в новое состояние и запоминает момент перехода.

        Время перехода нужно интерфейсу: по нему видно, сколько длится текущая
        фаза и не зависла ли она.
        """
        if state is not self.state:
            self._phase_started = time.time()
        self.state = state

    #: Как называть фазы человеку и сколько они обычно длятся.
    PHASES: dict[str, tuple[str, float, float]] = {
        "IDLE": ("свободен", 0.0, 0.0),
        "LLM_ACTIVE": ("отвечает ведущий", 10.0, 40.0),
        "SWITCH_TO_IMAGE": ("выгружаю движок под генерацию", 1.0, 8.0),
        "IMAGE_GEN": ("рисуется кадр", 80.0, 110.0),
        "SWITCH_TO_LLM": ("возвращаю движок", 35.0, 60.0),
        "ERROR": ("ошибка", 0.0, 0.0),
    }

    def _now(self) -> dict[str, Any]:
        """Что происходит прямо сейчас: фаза, сколько длится, сколько обычно.

        Отдельный блок нужен потому, что имя состояния автомата ничего не
        говорит читателю: ``IMAGE_GEN`` — это «рисуется кадр #13, 47 секунд,
        обычно 80–110».
        """
        phase = self.state.value
        title, low, high = self.PHASES.get(phase, (phase, 0.0, 0.0))
        elapsed = max(0.0, time.time() - self._phase_started)
        busy = phase not in ("IDLE", "ERROR")

        detail = title
        if phase == "IMAGE_GEN" and self._current_scene:
            detail = f"рисуется кадр #{self._current_scene.get('id')}"
            if self._current_scene.get("reference"):
                detail += " по образцу места"
        elif phase == "LLM_ACTIVE" and self.turn_started:
            detail = "отвечает ведущий"

        queue: list[int] = []
        if self._current_scene and self._scene_queue:
            current = self._current_scene.get("id")
            queue = [item for item in self._scene_queue if item != current]

        return {
            "phase": phase,
            "detail": detail,
            "since_s": round(elapsed, 1) if busy else 0.0,
            "usual_low_s": low,
            "usual_high_s": high,
            "busy": busy,
            "turn_s": round(time.time() - self.turn_started, 1) if self.turn_started else 0.0,
            "scene_id": (self._current_scene or {}).get("id"),
            "scene_prompt": (self._current_scene or {}).get("prompt"),
            "queue": queue,
            "overdue": bool(busy and high and elapsed > high * 1.6),
        }

    @property
    def settings(self) -> Any:
        """Текущие настройки."""
        return self.settings_store.settings

    def status(self) -> dict[str, Any]:
        """Снимок состояния для интерфейса."""
        try:
            gpu = metrics.gpu_stats()
            memory = metrics.ram_stats()
        except (RuntimeError, OSError) as exc:
            gpu = {"error": str(exc)}
            memory = {"error": str(exc)}

        engine: dict[str, Any] = {"port_pid": self.controller.port_pid(), "healthy": False}
        if engine["port_pid"] is not None:
            try:
                engine["healthy"] = self.controller.is_healthy()
                if self.uses_external():
                    # У llama.cpp нет ни /v1/stats, ни геометрии кэша: спрашивать
                    # их — гарантированная ошибка в каждой сводке состояния.
                    # Модель показывается из настроек, а не первая из списка:
                    # список сервера отсортирован и к выбору отношения не имеет.
                    engine["external"] = True
                    engine["model"] = Path(str(self.settings.model_path)).name
                    engine["server_models"] = len(list_server_models(str(
                        getattr(self.settings, "external_url", "")))) or None
                else:
                    stats = self.ft.stats()
                    engine["vram_mb"] = round((stats.get("vram_bytes") or 0) / (1024 * 1024))
                    engine["ctx"] = (stats.get("model") or {}).get("ctx")
                    engine["model"] = (stats.get("model") or {}).get("id")
                    engine["geometry"] = self.ft.geometry().describe()
                    engine["tps"] = (stats.get("throughput") or {}).get("decode_tps")
            except FreeTokenError as exc:
                engine["error"] = str(exc)

        return {
            "state": self.state.value,
            "now": self._now(),
            "gpu": gpu,
            "memory": memory,
            "engine": engine,
            "comfyui": {
                "alive": self.comfy.is_alive(),
                "models_loaded": self.comfy_models_loaded,
            },
            "log": list(self.log)[-80:],
            "counts": self.db.stats(),
            "last_turn": None if self.last_outcome is None else self.last_outcome.as_dict(),
            "current_model": str(self.settings.model_path),
        }

    def debug_payload(self) -> dict[str, Any]:
        """Содержимое отладочной панели: последний запрос, ответ и разбор."""
        return dict(self.debug)

    # --- движок -------------------------------------------------------------

    def engine_config(self) -> EngineConfig:
        """Конфигурация запуска движка из текущих настроек."""
        settings = self.settings
        return EngineConfig(
            name=f"ratio{int(settings.engine_memory_ratio * 100):03d}",
            memory_ratio=settings.engine_memory_ratio,
            moe_strategy=settings.engine_moe_strategy,
            model_path=Path(settings.model_path),
        )

    # --- выбор движка -------------------------------------------------------

    def uses_external(self) -> bool:
        """Работает ли текстовая модель на внешнем движке (llama.cpp).

        @returns: ``True``, если выбран внешний движок.
        """
        return str(getattr(self.settings, "engine_kind", "freetoken")) == "external"

    def external_port(self) -> int:
        """Порт внешнего сервера из настроек.

        @returns: номер порта; при неудачном адресе — 1919.
        """
        url = str(getattr(self.settings, "external_url", "") or "")
        tail = url.rstrip("/").rsplit(":", 1)[-1]
        return int(tail) if tail.isdigit() else 1919

    def sync_engine(self) -> dict[str, Any]:
        """Пересобирает клиента и контроллер под выбранный движок.

        Вызывается при старте и после сохранения настроек: пользователь может
        переключить движок на ходу, а клиент и контроллер разные.

        @returns: отчёт о том, что выбрано.
        """
        settings = self.settings
        if self.uses_external():
            port = self.external_port()
            url = str(getattr(self.settings, "external_url", "") or f"http://127.0.0.1:{port}")
            self.ft = FreeTokenClient(url, timeout_s=float(settings.request_timeout_s))
            server = find_server(getattr(settings, "external_server_path", "") or None)
            manages = bool(getattr(settings, "external_manage", True))
            if manages and server is not None:
                self.controller = ExternalController(server, port=port)
            else:
                # Сервер поднимает кто-то другой — LM Studio. Запускать и
                # останавливать его отсюда нельзя: мы бы убили чужой процесс.
                self.controller = ExternalController(
                    server or Path("llama-server.exe"), port=port
                )
            if getattr(self, "context", None) is not None:
                self.context.ft = self.ft
            return {
                "kind": "external",
                "url": url,
                "server": str(server) if server else None,
                "manage": manages,
            }
        self.ft = FreeTokenClient(timeout_s=float(settings.request_timeout_s))
        self.controller = EngineController(port=1919)
        if getattr(self, "context", None) is not None:
            self.context.ft = self.ft
        return {"kind": "freetoken", "url": "http://127.0.0.1:1919", "manage": True}

    def external_config(self) -> ExternalConfig:
        """Конфигурация запуска llama.cpp из текущих настроек.

        @returns: конфигурация для внешнего контроллера.
        """
        settings = self.settings
        return ExternalConfig(
            model_path=Path(settings.model_path),
            port=self.external_port(),
            context=int(getattr(settings, "external_context", 8192)),
            gpu_layers=int(getattr(settings, "external_gpu_layers", 99)),
            disable_thinking=bool(getattr(settings, "external_disable_thinking", True)),
            reasoning_budget=int(getattr(settings, "external_reasoning_budget", 256)),
            alias="novelforge",
        )

    def ensure_comfy(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """Гарантирует, что сервер ComfyUI запущен.

        Сервер поднимается напрямую, без Comfy Desktop: приложение не стартует
        backend само, а ждёт нажатия Start в окне. Прямой запуск занимает около
        пятнадцати секунд против нескольких минут у оболочки.

        @param timeout_s: сколько ждать готовности сервера.
        @returns: отчёт о запуске; ``ready`` показывает, удалось ли.
        """
        if self.comfy.is_alive():
            return {"already_running": True, "ready": True, "seconds": 0.0}
        self.note("сервер ComfyUI не отвечает — запускаю напрямую")
        started = time.time()
        try:
            alive, waited = self.comfy.ensure_running(timeout_s=timeout_s, mode="direct")
        except ComfyError as exc:
            self.note(f"не удалось запустить ComfyUI: {exc}")
            return {"ready": False, "error": str(exc), "seconds": round(time.time() - started, 2)}
        seconds = round(time.time() - started, 2)
        if alive:
            self.note(f"ComfyUI поднят за {waited:.1f} c")
        else:
            self.note(
                f"ComfyUI не поднялся за {waited:.0f} c — смотри {config.LOGS_DIR / 'comfyui_server.log'}"
            )
        return {"ready": alive, "seconds": seconds, "waited_s": round(waited, 1)}

    def _configure_reasoning(self, force: bool = False) -> dict[str, Any]:
        """Договаривается с движком о размышлениях модели.

        Делается один раз на запуск движка: набор допустимых аргументов зависит
        от модели, а сам запрос — лишний обмен по HTTP.

        @param force: перечитать настройки даже если уже применялись.
        @returns: отчёт о применённых аргументах.
        """
        if self._reasoning_ready and not force:
            return {"skipped": True}
        if self.uses_external():
            # llama.cpp не принимает настройку размышлений после запуска: её
            # задают ключом шаблона при старте сервера.
            self._reasoning_ready = True
            off = bool(getattr(self.settings, "external_disable_thinking", True))
            return {"mode": "off" if off else "on", "applied": False, "external": True,
                    "reason": "задаётся при запуске llama.cpp"}
        report = self.ft.configure_reasoning(self.settings.reasoning_mode)
        self._reasoning_ready = True
        if report.get("applied"):
            self.note(
                f"размышления модели: {report['mode']} "
                f"(у модели по умолчанию {report.get('model_default')})"
            )
        elif report.get("model_default") == "on" and self.settings.reasoning_mode == "off":
            self.note(
                "размышления выключить не удалось — "
                f"движок не предложил аргументов ({report.get('reason') or report.get('error')})"
            )
        return report

    def invalidate_reasoning(self) -> None:
        """Заставляет перечитать настройки размышлений при следующем запросе.

        Вызывается после смены режима в интерфейсе: уже запущенный движок
        помнит прежние аргументы, а новый запрос должен идти с новыми.
        """
        self._reasoning_ready = False

    def ensure_engine(self, timeout_s: float | None = None) -> dict[str, Any]:
        """Гарантирует, что движок запущен и отвечает.

        Перед стартом проверяется свободная VRAM: если её держит ComfyUI, он
        сначала выгружается. Движку нужно около 9.6 GB, а ComfyUI после
        генерации продолжает занимать почти всю карту, пока его не попросят.

        @returns: отчёт о старте; ``ready`` показывает, удалось ли дождаться.
        """
        settings = self.settings
        if self.controller.is_healthy():
            running = self.status().get("engine", {}).get("model")
            wanted = str(settings.model_path)
            if running and wanted and running not in wanted and wanted not in running:
                self.note(f"движок обслуживает {running}, а выбрана {wanted} — перезапускаю")
                self.stop_engine()
            else:
                self._configure_reasoning()
                return {"already_running": True, "ready": True, "seconds": 0.0}
        elif self.controller.port_pid() is not None:
            # Порт занят, но на нём не наш движок: отвечает не то, что мы ждём,
            # либо не отвечает вовсе. У FreeToken и llama.cpp порт общий (1919),
            # поэтому после смены движка здесь оказывается процесс прежнего
            # движка. Считать его готовым нельзя — иначе модель применится
            # только со второго выбора, когда порт успеет освободиться.
            running_kind = "external" if self.uses_external() else "freetoken"
            self.note(
                f"порт {self.controller.port} занят не нашим движком, "
                f"а нужен {running_kind} — перезапускаю"
            )
            self.stop_engine()
            if self.controller.port_pid() is not None:
                # Процесс не наш: убивать его нельзя. Пробуем подождать, пока
                # порт освободится сам, и только потом стартуем.
                self.controller.wait_port_free(timeout_s=30.0)

        if self.uses_external() and not bool(getattr(settings, "external_manage", True)):
            # Сервер LM Studio поднимает пользователь. Сказать об этом прямо
            # полезнее, чем молча ждать три минуты и выдать «не поднялся».
            url = str(getattr(settings, "external_url", ""))
            self.note(f"внешний сервер {url} не отвечает — запусти его в LM Studio")
            return {
                "ready": False,
                "error": f"сервер {url} не отвечает; запусти LM Studio и его сервер",
                "seconds": 0.0,
            }

        self._release_comfy_if_needed()
        started = time.time()
        config = self.external_config() if self.uses_external() else self.engine_config()
        report = self.controller.cold_start(
            config,
            timeout_s=timeout_s or settings.engine_start_timeout_s,
        )
        report["seconds"] = round(time.time() - started, 2)
        if report.get("ready"):
            timeline = report.get("timeline") or {}
            ready_s = timeline.get("spawn_to_ready_s", timeline.get("ready_s"))
            self.note(f"движок поднят за {ready_s} c")
            self._configure_reasoning(force=True)
        else:
            errors = (report.get("timeline") or {}).get("errors") or []
            self.note(f"движок не поднялся: {errors[-1][:200] if errors else 'причина неизвестна'}")
        return report

    def _release_comfy_if_needed(self, needed_mb: int = 8500) -> None:
        """Выгружает модели ComfyUI, если они мешают движку занять VRAM."""
        try:
            free_mb = metrics.gpu_stats()["free_mb"]
        except (RuntimeError, OSError) as exc:
            self.note(f"не удалось прочитать свободную VRAM: {exc}")
            return
        if free_mb >= needed_mb or not self.comfy.is_alive():
            return
        self.note(f"свободно {free_mb} MB — выгружаю модели ComfyUI перед стартом движка")
        try:
            self.comfy.free()
        except ComfyError as exc:
            self.note(f"не удалось выгрузить модели ComfyUI: {exc}")
            return
        settled, last = metrics.wait_for_free_vram(threshold_mb=needed_mb, timeout_s=30.0)
        self.note(f"после выгрузки свободно {last} MB" + ("" if settled else " — мало"))

    def stop_engine(self) -> dict[str, Any]:
        """Останавливает движок, освобождая VRAM под генерацию.

        Чужой сервер не трогаем: если модель держит LM Studio, убивать его
        процесс нельзя — пользователь поднимал его сам, и там могут быть свои
        дела. В этом случае сообщаем, что освободить память нужно вручную.
        """
        if self.uses_external() and not bool(getattr(self.settings, "external_manage", True)):
            # Проверка идёт первой: чужой сервер не останавливаем ни при каких
            # условиях, даже если порт почему-то свободен.
            self.note(
                "внешний сервер не наш — выгрузи модель в LM Studio перед генерацией кадров"
            )
            return {"external_untouched": True, "manage": False}
        if self.controller.port_pid() is None:
            return {"already_stopped": True}
        self._set_state(State.SWITCH_TO_IMAGE)
        self._agent_cancel.set()
        report = self.controller.stop(timeout_s=30.0)
        self._reasoning_ready = False
        self.note(
            f"движок остановлен за {report['stop_seconds']} c, "
            f"свободно VRAM {report['gpu_free_after_mb']} MB"
        )
        return report

    # --- миры и проверки ----------------------------------------------------

    def structure_world(self, world_id: int) -> dict[str, Any]:
        """Разбирает свободное описание мира на структуру силами модели.

        Пользователь пишет мир текстом, модель предлагает жанр, тон, стиль,
        правила и персонажей. Ничего не сохраняется молча: результат
        возвращается в интерфейс, пользователь его правит.

        @param world_id: мир, у которого заполнено поле ``brief``.
        @returns: разобранная структура и что именно удалось заполнить.
        """
        world = self.db.world(world_id)
        if world is None:
            raise RuntimeError(f"мир {world_id} не найден")
        if not world.brief.strip():
            raise RuntimeError("у мира пустое описание — структурировать нечего")

        self._set_state(State.LLM_ACTIVE)
        report = self.ensure_engine()
        if not report.get("ready"):
            self._set_state(State.ERROR)
            raise RuntimeError("движок FreeToken не удалось запустить")

        settings = self.settings
        started = time.time()
        result = self.ft.chat(
            [{"role": "user", "content": prompts.render(prompts.STRUCTURE_WORLD_PROMPT, brief=world.brief)}],
            max_tokens=1600,
            temperature=0.4,
            timeout_s=settings.request_timeout_s,
        )
        try:
            payload = loads_json(result.text)
        except ValueError as exc:
            self._set_state(State.IDLE)
            raise RuntimeError(f"модель вернула не JSON: {exc}") from exc
        self._set_state(State.IDLE)
        if not isinstance(payload, dict):
            raise RuntimeError("модель вернула не объект JSON — структуру разобрать не удалось")

        written = {"rules": 0, "characters": 0}
        fields_to_set = {
            key: payload[key]
            for key in ("name", "genre", "tone", "narrator", "style")
            if isinstance(payload.get(key), str) and payload[key].strip()
        }
        self.db.update_world(world_id, fields_to_set)

        for rule in payload.get("rules") or []:
            if not isinstance(rule, dict) or not str(rule.get("body", "")).strip():
                continue
            self.db.add_rule(
                world_id,
                str(rule["body"]).strip(),
                title=str(rule.get("title", "")).strip(),
                kind=str(rule.get("kind", "rule")),
            )
            written["rules"] += 1

        for character in payload.get("characters") or []:
            if not isinstance(character, dict) or not str(character.get("name", "")).strip():
                continue
            self.db.add_character(
                world_id,
                str(character["name"]).strip(),
                role=str(character.get("role", "")).strip(),
                description=str(character.get("description", "")).strip(),
                appearance=str(character.get("appearance", "")).strip(),
                speech=str(character.get("speech", "")).strip(),
            )
            written["characters"] += 1

        self.debug["structure"] = {
            "elapsed_s": round(time.time() - started, 2),
            "raw": result.text if settings.debug_keep_payloads else "",
            "parsed": payload,
            "written": written,
        }
        self.note(
            f"мир #{world_id} структурирован за {time.time() - started:.1f} c: "
            f"правил {written['rules']}, персонажей {written['characters']}"
        )
        return {"world": fields_to_set, "written": written, "parsed": payload}

    def validate(self, session_id: int) -> dict[str, Any]:
        """Проверяет мир и партию на типовые дефекты.

        @param session_id: партия, для которой проверяется контекст.
        @returns: список замечаний с признаком серьёзности и подсказкой.
        """
        issues: list[dict[str, str]] = []
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session else None
        if world is None:
            return {"issues": [{"level": "error", "text": "партия не привязана к миру"}]}

        fmt = get_format(world.format)
        if not world.brief.strip():
            issues.append({
                "level": "warn",
                "text": "у мира нет описания — модель не знает обстановку",
                "fix": "заполни поле «описание мира» и нажми «Разобрать»",
            })
        rules = self.db.rules(world.id)
        if len(rules) < 3:
            issues.append({
                "level": "warn",
                "text": f"правил всего {len(rules)} — модель будет импровизировать за тебя",
                "fix": "добавь 3-8 правил: чего в мире не бывает и что запрещено",
            })
        disabled = [rule for rule in rules if not rule.enabled]
        if disabled:
            issues.append({
                "level": "info",
                "text": f"{len(disabled)} правил выключено",
                "fix": "включи, если они всё ещё действуют",
            })
        characters = self.db.characters(world.id)
        if fmt.wants_scene and characters and any(not c.appearance for c in characters):
            missing = [c.name for c in characters if not c.appearance]
            issues.append({
                "level": "warn",
                "text": f"у персонажей нет внешности: {', '.join(missing)}",
                "fix": "без appearance генератор нарисует разных людей в разных кадрах",
            })
        if fmt.wants_scene and not world.style.strip():
            issues.append({
                "level": "info",
                "text": "стиль иллюстраций не задан",
                "fix": "задай стиль, иначе кадры будут выглядеть по-разному",
            })

        # Имена, различающиеся только пробелами или регистром, в слое внешности
        # сливаются в одну запись, а в промпте читаются как два человека.
        messy = [c.name for c in characters if c.name != c.name.strip()]
        if messy:
            issues.append({
                "level": "warn",
                "text": "в именах персонажей лишние пробелы: "
                        + ", ".join(repr(name) for name in messy),
                "fix": "убери пробелы: по имени ведущий отличает одного от другого",
            })
        by_key: dict[str, list[str]] = {}
        for character in characters:
            by_key.setdefault(character.name.strip().casefold(), []).append(character.name)
        twins = [names for names in by_key.values() if len(names) > 1]
        if twins:
            issues.append({
                "level": "error",
                "text": "персонажи с одинаковыми именами: "
                        + "; ".join(", ".join(names) for names in twins),
                "fix": "оставь одну карточку: иначе ведущий путает их между собой",
            })
        # Одинаковая внешность у разных имён — почти всегда копия карточки.
        by_look: dict[str, list[str]] = {}
        for character in characters:
            look = character.appearance.strip().casefold()
            if look:
                by_look.setdefault(look, []).append(character.name)
        clones = [names for names in by_look.values() if len(names) > 1]
        if clones:
            issues.append({
                "level": "warn",
                "text": "одинаковая внешность у разных персонажей: "
                        + "; ".join(", ".join(names) for names in clones),
                "fix": "генератор нарисует их одинаково, и ведущий тоже начнёт путать",
            })

        try:
            prompt = self.context.build(session_id, "проверка", count_exactly=False)
            stable = sum(layer.tokens for layer in prompt.layers if layer.stable)
            if stable > prompt.budget_tokens * 0.6:
                issues.append({
                    "level": "warn",
                    "text": f"неизменяемая часть промпта занимает {stable} токенов "
                            f"из бюджета {prompt.budget_tokens}",
                    "fix": "сократи правила и карточки персонажей или подними бюджет",
                })
            if prompt.dropped_message_ids:
                issues.append({
                    "level": "info",
                    "text": f"{len(prompt.dropped_message_ids)} сообщений не помещаются в окно",
                    "fix": "сверни историю в суммаризацию",
                })
        except (FreeTokenError, ValueError) as exc:
            issues.append({"level": "error", "text": f"контекст не собирается: {exc}"})

        if self.controller.port_pid() is None:
            issues.append({
                "level": "info",
                "text": "движок остановлен",
                "fix": "поднимется сам при первом ходе",
            })
        if not self.comfy.is_alive():
            issues.append({
                "level": "warn",
                "text": "ComfyUI не отвечает на 8188",
                "fix": "нажми «Поднять ComfyUI» — сервер поднимется сам за 15 секунд",
            })

        for problem in self.db.check_integrity():
            issues.append({
                "level": "error",
                "text": f"целостность базы: {problem['check']} ({problem['count']})",
                "fix": "выгрузи мир в JSON и загрузи обратно — связи восстановятся",
            })

        return {"issues": issues, "world_id": world.id, "format": fmt.key}

    def run_agent(
        self,
        instruction: str,
        *,
        preview: bool = False,
        allow_destructive: bool = False,
        world_id: int | None = None,
    ) -> dict[str, Any]:
        """Выполняет указание пользователя силами агента-редактора.

        Агент правит миры, правила и персонажей по текстовому указанию. Он
        использует тот же движок, что и игра, поэтому перед работой движок
        поднимается, а размышления выключаются.

        Прогон держит тот же замок, что и ход игрока: иначе ход с картинкой
        остановил бы движок посреди работы агента. Если движок всё-таки выгрузили
        (кнопкой «Стоп всё» или сменой модели), агент останавливается на границе
        шага и сообщает, что успел применить.

        @param instruction: что нужно сделать, словами.
        @param preview: показать план и ничего не менять.
        @param allow_destructive: разрешить удаление.
        @param world_id: мир, открытый в интерфейсе; действия без явного номера
            мира применяются к нему.
        @returns: отчёт о прогоне со всеми шагами.
        """
        from novel.agent import WorldAgent

        with self._busy:
            self._agent_cancel.clear()
            self._set_state(State.LLM_ACTIVE)
            try:
                report = self.ensure_engine()
                if not report.get("ready"):
                    raise RuntimeError("движок FreeToken не удалось запустить")
                self._configure_reasoning()

                agent = WorldAgent(
                    self.db,
                    self.ft,
                    registry=self.registry,
                    settings_store=self.settings_store,
                )
                run = agent.run(
                    instruction,
                    preview=preview,
                    allow_destructive=allow_destructive,
                    should_continue=lambda: not self._agent_cancel.is_set(),
                    revive=self._revive_engine,
                    world_id=world_id,
                    max_steps=int(getattr(self.settings, "agent_max_steps", 8)),
                )
            except (RuntimeError, FreeTokenError, OSError) as exc:
                # Отчёт обязан быть свежим: иначе интерфейс покажет прошлый прогон,
                # и отказ агента останется незамеченным.
                from novel.agent import AgentRun

                run = AgentRun(instruction=instruction.strip(), error=str(exc))
                self.note(f"агент не смог начать: {exc}")
                self._set_state(State.ERROR)
            self.last_agent_run = run.as_dict()
            if run.pending and not preview:
                self.last_agent_run["pending_results"] = [
                    self._apply_pending(action) for action in run.pending
                ]
            if run.interrupted:
                self.note(
                    f"агент прерван: применено {run.changed} изменений — "
                    f"{', '.join(run.applied()) or 'ничего'}"
                )
            elif run.error:
                self.note(
                    f"агент остановился: {run.error}"
                    + (f"; применено {run.changed}" if run.changed else "")
                )
            else:
                self.note(
                    f"агент закончил: изменений {run.changed}, "
                    f"шагов {len(run.steps)}, {run.seconds:.1f} c"
                )
            self._set_state(State.IDLE)
            return self.last_agent_run
    def _apply_pending(self, action: dict[str, Any]) -> dict[str, Any]:
        """Выполняет действие, отложенное агентом до конца прогона.

        Смена модели перезапускает движок, поэтому внутри прогона она невозможна:
        агент погасил бы сам себя. Здесь прогон уже закончен, и перезапуск
        безопасен.

        @param action: словарь действия из отчёта агента.
        @returns: отчёт о выполнении для интерфейса.
        """
        from novel.agent import find_model
        from novel.models import preset_for

        name = str(action.get("op") or "")
        if name != "switch_model":
            return {"op": name, "ok": False, "error": "неизвестное отложенное действие"}

        found = find_model(self.registry, str(action.get("model", "")))
        if found is None:
            self.note(f"модель «{action.get('model')}» не найдена")
            return {"op": name, "ok": False, "error": "модель не найдена"}

        settings = self.settings_store.settings
        if settings.model_path == found.path:
            return {"op": name, "ok": True, "model": found.name, "note": "уже включена"}

        settings.model_path = found.path
        settings.apply_model_preset(preset_for(found.model_type))
        self.settings_store.save()
        self.invalidate_reasoning()
        self.note(f"агент просил сменить модель — переключаю на {found.name}")

        report = self.restart_with_current_model()
        if not report.get("ready"):
            errors = (report.get("timeline") or {}).get("errors") or []
            return {
                "op": name,
                "ok": False,
                "model": found.name,
                "error": errors[-1][:200] if errors else "движок не поднялся",
            }
        return {
            "op": name,
            "ok": True,
            "model": found.name,
            "seconds": report.get("seconds"),
        }

    def _revive_engine(self) -> bool:
        """Пробует поднять движок заново после выгрузки.

        Вызывается агентом, когда запрос к модели не прошёл. Выгрузка бывает
        штатной — например, перед генерацией картинки, — поэтому одну попытку
        восстановления имеет смысл сделать.

        @returns: ``True``, если движок снова отвечает.
        """
        self.note("движок выгружен посреди работы агента — поднимаю заново")
        self._reasoning_ready = False
        report = self.ensure_engine()
        return bool(report.get("ready"))

    def request_agent_stop(self) -> None:
        """Просит агента остановиться на границе следующего шага.

        Немедленно прервать запрос к модели нельзя: ответ уже генерируется.
        Остановка происходит между шагами, а всё применённое к этому моменту
        остаётся в базе и попадает в отчёт.
        """
        self._agent_cancel.set()

    def restart_with_current_model(self) -> dict[str, Any]:
        """Перезапускает движок под модель, выбранную в настройках.

        Нужно после смены модели: движок обслуживает один чекпоинт за запуск и
        сам на другой не переключится.

        @returns: отчёт о запуске; ``ready`` показывает, удалось ли.
        """
        self.stop_engine()
        self._reasoning_ready = False
        report = self.ensure_engine()
        return report

    def restart_all(self, stop_comfy: bool = True) -> dict[str, Any]:
        """Аварийная остановка: гасит движок и, если попросят, сервер ComfyUI.

        Просьба остановить агента выставляется до гашения процессов: запрос
        агента обрывается в тот же момент, когда умирает движок, и если флаг
        поставить позже, агент успеет поднять движок заново — на полминуты и всю
        видеопамять — ради того, чтобы сразу же остановиться.
        """
        from novel.procutil import kill_everything

        self._agent_cancel.set()
        report = kill_everything(stop_comfy=stop_comfy)
        self._reasoning_ready = False
        self.note(
            f"аварийная остановка: движок pid {report.engine_pids}, "
            f"ComfyUI pid {report.comfy_pids}, освобождено {report.gpu_free_after_mb - report.gpu_free_before_mb} MB"
        )
        self._set_state(State.IDLE)
        return report.as_dict()

    # --- ход игрока ---------------------------------------------------------

    def send(
        self,
        session_id: int,
        user_input: str,
        *,
        interject: str = "",
        images: list[str] | None = None,
    ) -> TurnOutcome:
        """Полный ход: ответ модели, суммаризация при необходимости, картинки.

        @param session_id: партия.
        @param user_input: реплика игрока.
        @param interject: указание ведущему, действующее только на этот ход.
        @param images: ссылки ``data:`` с изображениями, которые игрок приложил
            к реплике. Работают только на моделях со зрением.
        @returns: результат хода.
        """
        with self._busy:
            started = time.time()
            self.turn_started = started
            outcome = TurnOutcome()
            try:
                self._set_state(State.LLM_ACTIVE)
                outcome = self._llm_turn(session_id, user_input, interject, images or [])
                if outcome.scene_ids and self._should_generate_now(session_id, outcome):
                    outcome.images = self._image_phase(session_id, outcome)
                    if any(not item.get("error") for item in outcome.images):
                        self._note_shown(session_id, outcome)
            except (FreeTokenError, ComfyError, RuntimeError) as exc:
                outcome.errors.append(str(exc))
                self.note(f"ошибка хода: {exc}")
                self._set_state(State.ERROR)
            except Exception as exc:  # noqa: BLE001 — ход не должен ронять сервер интерфейса
                outcome.errors.append(f"{type(exc).__name__}: {exc}")
                self.note(f"непредвиденная ошибка: {traceback.format_exc(limit=3)}")
                self._set_state(State.ERROR)
            finally:
                if self.state != State.ERROR:
                    self._set_state(State.IDLE)
                outcome.total_s = time.time() - started
                self.last_outcome = outcome
                # Ход закончился: интерфейс больше не должен показывать прогресс.
                self.turn_started = None
                self._current_scene = None
            return outcome

    def _llm_turn(
        self, session_id: int, user_input: str, interject: str, images: list[str]
    ) -> TurnOutcome:
        """Один запрос к модели, разбор ответа и запись в хранилище."""
        timings: dict[str, float] = {}
        if self.controller.port_pid() is None:
            if not self.settings.engine_autostart:
                raise RuntimeError("движок остановлен, а автозапуск выключен")
            report = self.ensure_engine()
            timings["engine_start_cold"] = float(report.get("seconds") or 0.0)
            if not report.get("ready"):
                raise RuntimeError("движок FreeToken не удалось запустить")
        else:
            self._configure_reasoning()

        settings = self.settings
        # Вложение всегда сохраняется в историю: игрок должен видеть своё фото,
        # даже если модель его не получит.
        saved_images = self._store_uploads(session_id, images) if images else []
        model_images = self._images_for_model(session_id, saved_images)

        # Врезка бывает и без реплики: игрок не действует, а только направляет
        # сцену. Тогда пустой реплики в истории не появляется — иначе она висела
        # бы в переписке как сообщение ни о чём.
        speaks = bool(user_input.strip()) or bool(saved_images)
        user_message_id = (
            self.db.add_message(session_id, "user", user_input, attachments=saved_images)
            if speaks
            else None
        )
        prompt = self.context.build(
            session_id,
            user_input,
            interject=interject,
            solo=not speaks and bool(interject.strip()),
            exclude_ids={user_message_id} if user_message_id else set(),
        )

        request: dict[str, Any] = {
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "timeout_s": settings.request_timeout_s,
        }
        if not settings.sampling_from_model:
            request["top_p"] = settings.top_p
            request["top_k"] = settings.top_k

        messages = prompt.chat_messages()
        if model_images:
            messages = self._attach_images(messages, model_images, timings)

        started = time.time()
        result = self.ft.chat_stream(messages, **request)
        timings["llm"] = time.time() - started
        timings["llm_ttft"] = result.ttft_s or 0.0
        self.note(
            f"ответ модели: вход {result.prompt_tokens} ток., выход {result.completion_tokens} ток. "
            f"за {timings['llm']:.1f} c (ttft {timings['llm_ttft']:.2f} c, "
            f"{result.decode_tokens_per_second:.1f} ток/с)"
        )

        parsed = parse_reply(result.text)
        for error in parsed.errors:
            self.note(f"разбор ответа: {error}")

        message_id = self.db.add_message(
            session_id, "assistant", parsed.prose, tokens=result.completion_tokens
        )
        outcome = TurnOutcome(prose=parsed.prose, parsed=parsed, timings=timings,
                              assistant_message_id=message_id)
        outcome.summary = self._maybe_summarize(session_id, prompt)

        if parsed.looks:
            # Ведущий сообщил, как персонажи выглядят теперь. Храним в состоянии
            # партии: сводка истории такие подробности теряет.
            self._merge_looks(session_id, parsed.looks)

        # Ведущий о блоке looks не помнит и почти никогда его не присылает —
        # проверено на живой машине. Поэтому после хода задаём прямой короткий
        # вопрос отдельным запросом.
        tracked = self._track_looks(session_id, user_input, parsed.prose, settings, timings)
        if tracked:
            self._merge_looks(session_id, tracked)

        # Ведущий сам отмечает, что из задуманного сбылось или перестало
        # подходить. Закрываем только те номера, что действительно есть в этой
        # партии: выдуманный номер не должен ничего ломать.
        for note_ids, status, label in (
            (parsed.fulfilled_notes, "done", "сбылось"),
            (parsed.dropped_notes, "dropped", "отброшено"),
        ):
            if not note_ids:
                continue
            closed = 0
            for note_id in note_ids:
                item = self.db.note(note_id)
                if item is not None and item.session_id == session_id:
                    self.db.close_note(note_id, status)
                    closed += 1
            if closed:
                self.note(f"задумано на потом: {label} {closed}")

        if parsed.scene is not None:
            location = parsed.scene.location or self.db.get_state(session_id, "location", "")
            if location:
                self.db.set_state(session_id, "location", location)
            if parsed.scene.npc:
                # Выдуманные имена внутрь не пускаем: на практике ведущий
                # назвал «Марлу», которой в мире нет, и она осела в состоянии
                # партии — кадр потом показывал не того. Тот же урок, что с
                # «внешностью сейчас».
                known, invented = self._known_npcs(session_id, parsed.scene.npc)
                if invented:
                    self.note(
                        "в кадре названы незнакомцы, пропускаю: " + ", ".join(invented[:4])
                    )
                if known:
                    self.db.set_state(session_id, "npc", known)
                    parsed.scene.npc = known
            described = parsed.scene
            final_prompt = described.full_prompt
            # Портрет собеседника: кадр обязан показать того, кто отвечает.
            # Ведущий заполняет npc через раз, поэтому subject ищется ещё и по
            # репликам, а промпт при необходимости пересобирается из внешности.
            if self.context._portrait_mode(self._format_for(session_id)):
                subject = self._portrait_subject(session_id, described, parsed.prose)
                if subject is not None:
                    shaped = self._shape_portrait(
                        final_prompt, subject, self._world_style(session_id)
                    )
                    if shaped != final_prompt:
                        self.note(
                            f"кадр был не портретом — показываю {subject.name} в полный рост"
                        )
                        final_prompt = shaped
            if settings.image_prompt_mode == "separate":
                final_prompt = self._rewrite_image_prompt(session_id, described) or final_prompt
            location_id = self._remember_location(session_id, location, described, outcome)
            self._mark_new_characters(session_id, described, outcome)
            scene_id = self.db.add_scene(
                session_id,
                final_prompt,
                message_id=message_id,
                raw_description=described.image_prompt,
                seed=described.seed,
                location_id=location_id,
            )
            # Генератор понимает только английский. Ведущий иногда пишет описание
            # кадра по-русски — без проверки кадр выходит мусорным, — и кадр получается
            # мусорным. Молчать об этом нельзя: иначе непонятно, почему картинка
            # не та.
            if not _looks_english(final_prompt):
                self.note(
                    f"сцена #{scene_id}: описание кадра не по-английски — "
                    "генератор его не поймёт, кадр может выйти мусорным"
                )
            outcome.scene_ids.append(scene_id)
            self.note(f"сцена #{scene_id} в очереди: {final_prompt[:90]}")

        for item in parsed.speculative:
            if not settings.speculative_enabled:
                break
            self.db.add_speculative(
                session_id, item.prompt, trigger=item.trigger, seed=item.seed
            )

        self.debug = {
            "session_id": session_id,
            "prompt": prompt.as_dict(include_text=settings.debug_keep_payloads),
            "system_text": prompt.system_text if settings.debug_keep_payloads else "",
            "messages": prompt.messages if settings.debug_keep_payloads else [],
            "raw_reply": result.text if settings.debug_keep_payloads else "",
            "parsed": parsed.as_dict(),
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "ttft_s": None if result.ttft_s is None else round(result.ttft_s, 3),
                "decode_tps": round(result.decode_tokens_per_second, 2),
            },
            "timings": {key: round(value, 2) for key, value in timings.items()},
            "engine": self.status().get("engine", {}),
            "ts": time.time(),
        }
        return outcome

    def _images_for_model(self, session_id: int, attached: list[str]) -> list[str]:
        """Решает, какие изображения получит модель на этом ходу.

        Возможность видеть и необходимость видеть — разные вещи: кадр 768x768
        стоит около 258 токенов входа и добавляет 1-4 секунды к ответу. Поэтому
        политика важнее возможности, и по умолчанию модель смотрит только на то,
        что игрок приложил осознанно.

        @param session_id: партия, к которой относится ход.
        @param attached: пути к изображениям, приложенным игроком.
        @returns: пути к изображениям, которые уйдут в запрос.
        """
        mode = getattr(self.settings, "vision_mode", "on_attach")
        if mode == "off":
            if attached:
                self.note("зрение выключено: фото сохранено, модели не отправлено")
            return []

        chosen: list[str] = []
        if attached:
            if not self.ft.supports_images():
                raise RuntimeError(
                    "текущая модель не принимает изображения на вход — "
                    "выбери Gemma-4 или Qwen3.6 в разделе «Модель», "
                    "либо выключи отправку фото"
                )
            chosen.extend(attached)

        if mode == "on_attach_and_last_frame":
            previous = self._last_generated_frame(session_id)
            if previous and previous not in chosen:
                if self.ft.supports_images():
                    chosen.append(previous)
                    self.note(f"к ходу приложен последний кадр: {Path(previous).name}")
                else:
                    self.note("последний кадр не отправлен: модель не принимает изображения")
        return chosen

    def _last_generated_frame(self, session_id: int) -> str | None:
        """Путь к последнему готовому кадру партии.

        Нужен, чтобы модель видела, что уже нарисовано, и держала визуальную
        непрерывность: иначе на следующем ходу она опишет то же место иначе.

        @param session_id: партия.
        @returns: путь к файлу либо ``None``.
        """
        for scene in reversed(self.db.scenes(session_id)):
            if scene.status == "done" and scene.path and Path(scene.path).exists():
                return scene.path
        return None

    def _load_images(self, paths: list[str]) -> list[str]:
        """Готовит изображения к отправке, кэшируя результат по пути.

        Кэш нужен потому, что сборка контекста может запросить одни и те же
        вложения несколько раз подряд, а уменьшение и кодирование стоят времени.

        @param paths: пути к изображениям на диске.
        @returns: ссылки ``data:`` с изображениями.
        """
        urls: list[str] = []
        for raw in paths:
            cached = self._image_cache.get(raw)
            if cached is None:
                try:
                    cached, _w, _h, _orig = prepare_image(Path(raw))
                except Exception as exc:  # noqa: BLE001 — битый файл не должен ронять ход
                    self.note(f"вложение {Path(raw).name} не подготовлено: {exc}")
                    continue
                self._image_cache[raw] = cached
            urls.append(cached)
        return urls

    def _store_uploads(self, session_id: int, images: list[str]) -> list[str]:
        """Сохраняет присланные интерфейсом изображения на диск.

        @param session_id: партия, к которой относятся вложения.
        @param images: ссылки ``data:`` с изображениями.
        @returns: пути к сохранённым файлам.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        saved: list[str] = []
        for index, data_url in enumerate(images):
            target = save_upload(
                data_url, config.UPLOADS_DIR, f"s{session_id}-{stamp}-{index}"
            )
            saved.append(str(target))
        self.note(f"приложено изображений: {len(saved)}")
        return saved

    def _attach_images(
        self, messages: list[dict[str, Any]], paths: list[str], timings: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Превращает последнее сообщение в многочастное с изображениями.

        Картинки уменьшаются перед отправкой: у модели есть бюджет токенов на
        изображение, и полноразмерный кадр стоит заметно дороже, чем нужно для
        описания сцены.

        @param messages: готовые сообщения запроса.
        @param paths: пути к изображениям на диске.
        @param timings: куда записать время подготовки.
        @returns: сообщения с приложенными изображениями.
        """
        started = time.time()
        data_urls = self._load_images(paths)
        for raw in paths:
            original = Path(raw)
            self.note(f"изображение {original.name} приложено к реплике")
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str) and data_urls:
            messages = [*messages[:-1], user_message(content, data_urls)]
        timings["images_prepare"] = round(time.time() - started, 2)
        return messages

    def _maybe_summarize(self, session_id: int, prompt: AssembledPrompt) -> dict[str, Any] | None:
        """Сворачивает историю, если окно перестало вмещать переписку."""
        settings = self.settings
        if not settings.auto_summarize:
            return None
        if not self.context.needs_summary(session_id, len(prompt.dropped_message_ids)):
            return None
        self._set_state(State.SUMMARIZING)
        try:
            report = self.context.summarize(session_id)
        except FreeTokenError as exc:
            self.note(f"суммаризация не удалась: {exc}")
            self._set_state(State.LLM_ACTIVE)
            return {"error": str(exc)}
        self._set_state(State.LLM_ACTIVE)
        if report:
            self.note(
                f"история свёрнута: {report['summarized_messages']} сообщений -> "
                f"{report['tokens']} токенов памяти"
            )
        return report

    def _match_character(self, world_id: int, name: str) -> Character | None:
        """Находит персонажа мира по имени, как его назвала модель.

        Точного совпадения мало. Ведущий пишет «Селена» там, где в карточке
        «Селена, верховная жрица», и пишет «Марлу» там, где такого персонажа нет
        вовсе. Поэтому сравнение идёт по трём ступеням, а выдумка не проходит.

        @param world_id: мир.
        @param name: имя от ведущего.
        @returns: найденный персонаж либо ``None``.
        """
        wanted = " ".join(str(name or "").split()).casefold().strip(" .,!?:;")
        if not wanted:
            return None
        characters = [
            character for character in self.db.characters(world_id)
            if character.name.strip()
        ]
        by_full = {c.name.strip().casefold(): c for c in characters}
        if wanted in by_full:
            return by_full[wanted]
        # «Селена» должна найти «Селена, верховная жрица».
        by_head = {
            c.name.split(",")[0].strip().casefold(): c
            for c in characters
        }
        if wanted in by_head:
            return by_head[wanted]
        for character in characters:
            head = character.name.strip().casefold()
            if head.startswith(wanted) or wanted.startswith(head.split(",")[0].strip()):
                return character
        return None

    def _known_npcs(self, session_id: int, names: list[str]) -> tuple[list[str], list[str]]:
        """Делит имена ведущего на известные миру и выдуманные.

        @param session_id: партия.
        @param names: имена из блока сцены.
        @returns: ``(известные, выдуманные)`` — выдуманные сохраняются как есть,
            чтобы о них можно было сказать в журнале.
        """
        session = self.db.session(session_id)
        if session is None:
            return list(names), []
        known: list[str] = []
        invented: list[str] = []
        for name in names:
            hit = self._match_character(session.world_id, name)
            if hit is None:
                invented.append(str(name).strip())
            else:
                known.append(hit.name)
        return known, invented

    def _format_for(self, session_id: int) -> Any:
        """Формат диалога той партии, в которой идёт ход.

        @param session_id: партия.
        @returns: формат диалога; при отсутствии мира — формат истории.
        """
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session is not None else None
        return get_format(world.format if world is not None else "story")

    def _world_style(self, session_id: int) -> str:
        """Стиль изображений мира этой партии.

        @param session_id: партия.
        @returns: стиль мира либо пустая строка.
        """
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session is not None else None
        return str(getattr(world, "style", "") or "")

    def _portrait_subject(self, session_id: int, scene: SceneSpec, prose: str) -> Character | None:
        """Кто должен быть в кадре, когда формат рисует портрет собеседника.

        Ведущий заполняет ``npc`` через раз, а без него кадр уходит во что
        угодно: на пробе выходило «POV shot from a pilot seat» вместо человека,
        который только что ответил. Поэтому subject определяется и по репликам:
        последний говорящий в ответе — тот, чью реакцию и надо показать.

        @param session_id: партия.
        @param scene: разобранный блок сцены.
        @param prose: текст ответа, из которого берётся говорящий.
        @returns: персонаж мира либо ``None``, если ни одного не опознали.
        """
        session = self.db.session(session_id)
        if session is None:
            return None
        for name in scene.npc or []:
            hit = self._match_character(session.world_id, name)
            if hit is not None:
                return hit
        # Идём с конца: кадр показывает того, кто ответил последним.
        for line in reversed((prose or "").splitlines()):
            match = re.match(r"^\s*[-–—]?\s*([^:]{1,24}):", line)
            if match is None:
                continue
            hit = self._match_character(session.world_id, match.group(1))
            if hit is not None:
                return hit
        return None

    def _shape_portrait(self, prompt: str, character: Character, style: str) -> str:
        """Приводит промпт кадра к портрету в полный рост.

        Если ведущий уже описал полный рост, промпт не трогается: там могут быть
        поза и обстановка, которых в карточке персонажа нет. Иначе промпт
        собирается заново из внешности — иначе кадр покажет не того.

        @param prompt: промпт от ведущего.
        @param character: тот, кого надо показать.
        @param style: стиль мира.
        @returns: промпт кадра.
        """
        if "full body" in (prompt or "").lower():
            return prompt
        look = (character.appearance or character.description or "").strip()
        # Внешность по-русски генератору не годится: подстановка дала бы
        # «full body shot of цифровой образ». Тогда остаётся описание ведущего —
        # оно уже на английском, и его достаточно пометить полным ростом.
        if look and not _looks_english(look):
            look = ""
        if look:
            tail = f", {style.strip()}" if _looks_english(style) else ""
            return f"full body shot of {look}{tail}"
        if prompt.strip():
            return f"full body shot, {prompt.strip()}"
        return prompt

    def _rewrite_image_prompt(self, session_id: int, scene: SceneSpec) -> str | None:
        """Просит модель превратить описание сцены в промпт для генератора."""
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session else None
        characters = self.db.characters(world.id, enabled_only=True) if world else []
        appearances = "\n".join(
            f"- {character.name}: {character.appearance}"
            for character in characters
            if character.appearance
        ) or "- неизвестна"
        body = prompts.render(
            prompts.IMAGE_PROMPT_PROMPT,
            style=(world.style if world and world.style else "oil painting, fantasy"),
            appearances=appearances,
            description=scene.image_prompt,
        )
        try:
            result = self.ft.chat(
                [{"role": "user", "content": body}],
                max_tokens=self.settings.image_prompt_max_tokens,
                temperature=0.4,
            )
        except FreeTokenError as exc:
            self.note(f"промпт для картинки не переписан: {exc}")
            return None
        text = result.text.strip().strip('"').splitlines()[0] if result.text.strip() else ""
        if text:
            self.note(f"промпт изображения переписан моделью ({result.completion_tokens} токенов)")
        return text or None

    # --- политика изображений ----------------------------------------------

    def _should_generate_now(self, session_id: int, outcome: TurnOutcome) -> bool:
        """Решает, рисовать ли картинку прямо сейчас.

        Политика отвечает на вопрос «когда», а не «что»: сцена уже создана в
        любом случае и может быть нарисована позже кнопкой или в простое.

        @param session_id: партия.
        @param outcome: результат хода.
        @returns: ``True``, если запускать генерацию немедленно.
        """
        policy = self.settings.image_policy
        if policy in ("never", "manual", "idle"):
            return False
        if policy == "minimal":
            # Минимум картинок, но без пропусков: новое место и новое лицо в кадре
            # дают кадр всегда, а на знакомом месте — только если ведущий отметил
            # резкую перемену обстановки. Решение не отдано модели целиком: иначе
            # обещанные кадры то появляются, то нет.
            return outcome.new_place or bool(outcome.new_characters) or outcome.sudden
        if policy == "master":
            # Решает ведущий: он ставит блок scene только тогда, когда кадр того
            # стоит. Политика ничего не перепроверяет — иначе решение модели
            # перекрывалось бы сравнением названий мест, — но и не рисует, если
            # блока не было: проверка здесь, а не у вызывающего.
            return bool(outcome.scene_ids)
        if policy == "every_turn":
            return True
        if policy == "every_n":
            step = max(1, self.settings.image_every_n)
            assistant_turns = sum(
                1 for message in self.db.messages(session_id) if message.role == "assistant"
            )
            return assistant_turns % step == 0
        if policy == "on_scene_change":
            location = self.db.get_state(session_id, "location", "")
            last = self.db.get_state(session_id, "last_image_location", None)
            if last is None:
                return True
            return location != last
        return False

    # --- генерация ----------------------------------------------------------

    def _remember_location(
        self, session_id: int, name: str, scene: SceneSpec, outcome: TurnOutcome
    ) -> int | None:
        """Запоминает место сцены, чтобы оно выглядело одинаково при возвращении.

        Новое место заводится с каноническим описанием и стилем. Уже известное
        только отмечается посещением: описание берётся из текущей сцены, а облик
        держит кадр-образец, который подкладывается генератору как референс.

        @param session_id: партия, из которой пришла сцена.
        @param name: имя места от ведущего; пустое означает, что место не названо.
        @param scene: разобранный блок сцены.
        @param outcome: итог хода; в нём отмечается, что место новое.
        @returns: идентификатор места либо ``None``, если имя пустое.
        """
        cleaned = " ".join((name or "").split())
        if not cleaned:
            return None
        session = self.db.session(session_id)
        if session is None:
            return None
        world = self.db.world(session.world_id)
        known = self.db.location_by_name(session.world_id, cleaned)
        if known is not None:
            self.db.touch_location(known.id)
            self.note(f"знакомое место «{known.name}», посещение {known.visits + 1}")
            return known.id
        location_id = self.db.add_location(
            session.world_id,
            cleaned,
            prompt=scene.image_prompt,
            style=scene.style or (world.style if world else ""),
            seed=scene.seed,
        )
        outcome.new_place = True
        self.note(f"новое место «{cleaned}» — кадр станет для него образцом")
        return location_id

    def _mark_new_characters(
        self, session_id: int, scene: SceneSpec, outcome: TurnOutcome
    ) -> None:
        """Отмечает, какие лица в кадре появляются впервые.

        Список уже показанных хранится в состоянии партии: он нужен политике
        «минимум картинок», где новый персонаж в кадре — одно из трёх оснований
        нарисовать кадр.

        @param session_id: партия.
        @param scene: разобранный блок сцены.
        @param outcome: итог хода; в нём отмечаются новые лица.
        """
        saved = self.db.get_state(session_id, "shown_characters", []) or []
        # Сравнение без учёта регистра: ведущий пишет имя то с заглавной, то со
        # строчной, а «Старый Грог» и «старый грог» — один человек.
        seen = {str(item).strip().casefold() for item in saved}
        fresh = [
            name.strip() for name in scene.npc
            if name.strip() and name.strip().casefold() not in seen
        ]
        outcome.new_characters = fresh
        outcome.sudden = bool(scene.sudden)
        if fresh:
            self.note("впервые в кадре: " + ", ".join(fresh))

    def _note_shown(self, session_id: int, outcome: TurnOutcome) -> None:
        """Запоминает показанных персонажей, чтобы не считать их новыми снова."""
        saved = list(self.db.get_state(session_id, "shown_characters", []) or [])
        known = {str(item).strip().casefold() for item in saved}
        for name in outcome.new_characters:
            if name.strip().casefold() not in known:
                saved.append(name.strip())
                known.add(name.strip().casefold())
        if saved:
            self.db.set_state(session_id, "shown_characters", saved)

    def _pending_scenes(
        self, session_id: int, scene_ids: list[int] | None = None,
        quality: str | None = None,
    ) -> list[dict[str, Any]]:
        """Сцены, ожидающие генерации.

        Образец места здесь намеренно не вычисляется: за один заход рисуется
        несколько сцен, и кадр, нарисованный первым, становится образцом для
        следующих. Если определить образцы заранее, весь заход получил бы
        состояние до генерации, и второе место в том же ходу осталось бы без
        образца.

        @param session_id: партия.
        @param scene_ids: рисовать только эти сцены; ``None`` — все ожидающие.
        @param quality: ступень качества на этот заход; ``None`` — из настроек.
        @returns: сцены в виде словарей для генератора.
        """
        rows = self.db.scenes(session_id)
        wanted = set(scene_ids) if scene_ids else None
        return [
            {
                "id": scene.id,
                "prompt": scene.prompt,
                "seed": scene.seed,
                "location_id": scene.location_id,
                "quality": quality,
            }
            for scene in rows
            if scene.status == "pending" and (wanted is None or scene.id in wanted)
        ]

    def _location_reference(self, location_id: int | None) -> str | None:
        """Имя файла-образца места в каталоге входов ComfyUI.

        @param location_id: место сцены; ``None`` для сцен без места.
        @returns: имя для узла ``LoadImage`` либо ``None``.
        """
        if location_id is None:
            return None
        location = self.db.location(location_id)
        if location is None or not location.reference_path:
            return None
        source = Path(location.reference_path)
        if not source.is_file():
            return None
        try:
            return stage_reference(source)
        except OSError as exc:
            self.note(f"образец места не подготовлен: {exc}")
            return None

    def _image_phase(
        self, session_id: int, outcome: TurnOutcome, quality: str | None = None
    ) -> list[dict[str, Any]]:
        """Генерирует накопившиеся сцены за одно переключение VRAM.

        @param session_id: партия.
        @param outcome: копилка таймингов и ошибок.
        @param quality: ступень качества на этот заход; ``None`` — из настроек.
        @returns: отчёты по кадрам.
        """
        settings = self.settings
        results: list[dict[str, Any]] = []

        stop_report = self.stop_engine()
        outcome.timings["engine_stop"] = float(stop_report.get("stop_seconds") or 0.0)

        if not self.comfy.is_alive():
            self.ensure_comfy()

        if not self.comfy.is_alive():
            message = (
                "ComfyUI не отвечает на 127.0.0.1:8188, и поднять его не удалось — "
                f"смотри {config.LOGS_DIR / 'comfyui_server.log'}"
            )
            self.note(message)
            for scene in self._pending_scenes(session_id, quality=quality):
                self.db.finish_scene(scene["id"], None, "failed")
            outcome.errors.append(message)
        else:
            self._set_state(State.IMAGE_GEN)
            waiting = self._pending_scenes(session_id, quality=quality)
            self._scene_queue = [scene["id"] for scene in waiting]
            for scene in waiting:
                results.append(self._generate_one(session_id, scene, settings, outcome.timings))
            self._scene_queue = []
            try:
                started = time.time()
                self.comfy.free()
                self.comfy_models_loaded = False
                outcome.timings["comfy_free"] = round(time.time() - started, 2)
                self.note(f"модели ComfyUI выгружены за {outcome.timings['comfy_free']} c")
            except ComfyError as exc:
                self.note(f"не удалось выгрузить модели ComfyUI: {exc}")

        if self.settings.engine_autostart:
            self._set_state(State.SWITCH_TO_LLM)
            started = time.time()
            report = self.ensure_engine()
            outcome.timings["engine_return"] = round(time.time() - started, 2)
            if not report.get("ready"):
                outcome.errors.append("движок FreeToken не удалось вернуть после генерации")
        return results

    def _reference_for_scene(
        self, session_id: int, scene: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Решает, нужен ли этой сцене образец места.

        Образец — это картинка целиком, вместе с моментом: позой, светом и
        действием. Если подкладывать его в каждой сцене одного места, лечение,
        подъём и разговор выйдут одним и тем же кадром. Поэтому образец нужен при
        **возвращении** в место, а подряд идущие кадры того же места рисуются
        свободно по новому описанию.

        @param session_id: партия.
        @param scene: сцена из очереди.
        @returns: ``(имя места, имя файла-образца либо None)``.
        """
        location_id = scene.get("location_id")
        place = self.db.location(location_id) if location_id else None
        place_name = place.name if place else ""
        if not place_name:
            return "", None
        previous = str(self.db.get_state(session_id, "last_image_location", "") or "")
        if place_name.strip().casefold() == previous.strip().casefold():
            return place_name, None
        return place_name, self._location_reference(location_id)

    def _with_world_suffix(self, session_id: int, prompt: str) -> str:
        """Приписывает к промпту кадра дополнение мира.

        Дополнение уходит генератору дословно. Ведущий его не видит и переписать
        не может: так мир задаёт постоянную часть промпта кадра, не тратя на неё
        бюджет ответа и не завися от того, вспомнит ли о ней модель.

        @param session_id: партия.
        @param prompt: собранный промпт кадра.
        @returns: промпт с дополнением либо без него.
        """
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session else None
        suffix = (world.image_suffix if world else "").strip()
        return f"{prompt}, {suffix}" if suffix else prompt

    def _pick_seed(self, session_id: int, scene: dict[str, Any]) -> int:
        """Выбирает зерно кадра.

        Модель почти всегда присылает одно и то же круглое число: в примере
        промпта стоит ``"seed": 12345``, и ведущий повторяет его из кадра в кадр.
        Одинаковое зерно при похожем описании даёт похожую картинку, поэтому
        повтор пропускается и берётся случайное зерно. Осознанно новое число
        ведущего уважается.

        @param session_id: партия.
        @param scene: сцена из очереди.
        @returns: зерно для генератора.
        """
        given = scene.get("seed")
        previous = self.db.get_state(session_id, "last_image_seed", None)
        if given and int(given) != previous:
            return int(given)
        return random.randint(1, 2**31 - 1)

    def _look_names(self, session_id: int) -> dict[str, str]:
        """Сопоставляет написание имени каноническому.

        Ведущий пишет имена как придётся: «игрок» и «Игрок» для него одно и то
        же, а для словаря — две разные записи. Тогда персонаж попадает в слой
        внешности дважды, с разными описаниями, и ведущий путается, кто есть кто.

        @param session_id: партия.
        @returns: ``{имя в нижнем регистре: каноническое написание}``.
        """
        canonical = {"игрок": "игрок"}
        session = self.db.session(session_id)
        if session is not None:
            for character in self.db.characters(session.world_id):
                canonical[character.name.strip().casefold()] = character.name
        return canonical

    def _merge_looks(self, session_id: int, fresh: dict[str, str]) -> None:
        """Дописывает изменения внешности в состояние партии.

        Имена приводятся к каноническому написанию, а прежние записи
        переписываются заново: так разбираются пары вроде «игрок» и «Игрок»,
        накопившиеся до этой правки.

        @param session_id: партия.
        @param fresh: новые описания «имя — как выглядит».
        """
        canonical = self._look_names(session_id)
        known = dict(self.db.get_state(session_id, "looks", {}) or {})
        merged: dict[str, str] = {}
        strangers: list[str] = []
        for name, look in {**known, **fresh}.items():
            clean = name.strip()
            key = canonical.get(clean.casefold())
            if key is None:
                # Имени нет ни в карточках мира, ни среди слов об игроке. Ведущий
                # его выдумал, и в слое внешности такой персонаж становится для
                # него настоящим: дальше он путает выдуманного с остальными.
                strangers.append(clean)
                continue
            merged[key] = look
        self.db.set_state(session_id, "looks", merged)
        if fresh:
            self.note("внешность обновилась: " + ", ".join(
                canonical.get(name.strip().casefold(), name) for name in fresh
                if name.strip().casefold() in canonical
            ) or "внешность: новых записей нет")
        if strangers:
            self.note("внешность: пропущены незнакомые имена — " + ", ".join(strangers))

    def _track_looks(
        self,
        session_id: int,
        user_input: str,
        prose: str,
        settings: Any,
        timings: dict[str, float],
    ) -> dict[str, str]:
        """Спрашивает модель напрямую, изменилась ли чья-то внешность.

        Отдельный запрос нужен потому, что в основном ответе модель про блок
        ``looks`` забывает: инструкция в промпте есть, а блока нет. Прямой вопрос
        короткий и не портит основной ответ.

        @param session_id: партия.
        @param user_input: реплика игрока.
        @param prose: ответ ведущего.
        @param settings: настройки.
        @param timings: тайминги хода, куда пишется время проверки.
        @returns: найденные изменения внешности, возможно пустой словарь.
        """
        if not getattr(settings, "looks_tracking", True) or not prose.strip():
            return {}
        session = self.db.session(session_id)
        if session is None:
            return {}
        names = [c.name for c in self.db.characters(session.world_id)]
        known = dict(self.db.get_state(session_id, "looks", {}) or {})
        instruction = (
            "Ты следишь только за внешностью персонажей. Прочитай реплику игрока и "
            "ответ ведущего и ответь ТОЛЬКО объектом JSON без пояснений:\n"
            '{"Имя": "как выглядит теперь"}\n'
            "Включай только тех, чья внешность изменилась прямо сейчас: "
            "переоделся, снял или надел доспехи, ранен, перевязан, загримирован, "
            "скрыл лицо. Если внешность ни у кого не менялась, ответь {}.\n"
            f"Известные имена: {', '.join(names)}, игрок — это сам читатель."
        )
        if known:
            current = "; ".join(f"{name}: {look}" for name, look in known.items())
            instruction += f"\nСейчас записано: {current}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Игрок: {user_input}\n\nВедущий: {prose}"},
        ]
        started = time.time()
        try:
            # Размышления выключаем и здесь, а бюджета даём с запасом: с 220
            # токенами модель уходила в размышления целиком и не отвечала вовсе
            # (finish_reason='length') — проверка падала на каждом ходу.
            self._configure_reasoning()
            result = self.ft.chat_stream(
                messages, max_tokens=800, temperature=0.0,
                timeout_s=settings.request_timeout_s,
            )
        except (FreeTokenError, OSError) as exc:
            # Проверка внешности не стоит сорванного хода: основной ответ уже
            # записан и показан игроку.
            self.note(f"проверка внешности не удалась: {exc}")
            return {}
        timings["looks_check"] = time.time() - started
        found = looks_from_text(result.text)
        if found:
            self.note(
                f"проверка внешности: изменения за {timings['looks_check']:.1f} c "
                f"({', '.join(found)})"
            )
        return found

    def invent_notes(self, session_id: int, horizon: int = 5) -> dict[str, Any]:
        """Просит ведущего придумать, куда может пойти история.

        Задумки ведущего — предположения, а не приказ: они ложатся отдельным
        разделом слоя, и ведущему сказано отбросить их, если игрок свернул в
        сторону. Потолок отдельный от игроцкого, чтобы выдумки не вытесняли то,
        что задумал человек.

        @param session_id: партия.
        @param horizon: на сколько ходов вперёд задумывать.
        @returns: отчёт с добавленными задумками.
        """
        settings = self.settings
        session = self.db.session(session_id)
        if session is None:
            raise RuntimeError("партия не найдена")
        world = self.db.world(session.world_id)
        if world is None:
            raise RuntimeError("мир не найден")

        free = MODEL_NOTE_LIMIT - self.db.count_notes(session_id, "model")
        if free <= 0:
            return {
                "added": [],
                "note": f"у ведущего уже {MODEL_NOTE_LIMIT} своих задумок — "
                        "пусть сначала сбудутся или отпадут",
            }

        horizon = max(1, min(int(horizon or 5), 20))
        # Движок мог быть выгружен ради кадров: поднимаем, иначе запрос некуда слать.
        if self.controller.port_pid() is None:
            if not settings.engine_autostart:
                raise RuntimeError("движок остановлен, а автозапуск выключен")
            report = self.ensure_engine()
            if not report.get("ready"):
                raise RuntimeError("движок FreeToken не удалось запустить")

        recent = self.db.messages(session_id)[-8:]
        history = "\n".join(
            f"{'Игрок' if m.role == 'user' else 'Ведущий'}: {m.content[:300]}"
            for m in recent
        ) or "(история только начинается)"
        existing = "\n".join(
            f"- {item.text}" for item in self.db.notes(session_id, limit=10)
        ) or "(ничего не задумано)"
        rules = "\n".join(
            f"- {rule.title}: {rule.body}" for rule in self.db.rules(world.id, enabled_only=True)
        ) or "(правил нет)"

        body = prompts.render(
            prompts.INVENT_NOTES_PROMPT,
            count=str(free),
            horizon=str(horizon),
            world=(world.brief or world.name)[:1200],
            rules=rules[:1500],
            history=history[:2500],
            notes=existing[:800],
        )
        self._configure_reasoning()
        try:
            result = self.ft.chat(
                [{"role": "user", "content": body}],
                max_tokens=700, temperature=0.8,
                timeout_s=settings.request_timeout_s,
            )
        except (FreeTokenError, OSError) as exc:
            raise RuntimeError(f"ведущий не ответил: {exc}") from exc

        lines = [
            " ".join(line.strip().lstrip("-•*0123456789. ").split())
            for line in (result.text or "").splitlines()
        ]
        lines = [line for line in lines if len(line) > 12][:free]
        if not lines:
            return {"added": [], "note": "ведущий не придумал ничего путного"}

        anchor = self.db.count_messages(session_id)
        added: list[dict[str, Any]] = []
        for line in lines:
            note_id = self.db.add_note(
                session_id, line, source="model", horizon=horizon,
                anchor_message_id=anchor,
            )
            added.append({"id": note_id, "text": line})
        self.note(f"ведущий задумал на {horizon} ходов: {len(added)}")
        return {
            "added": added,
            "horizon": horizon,
            "note": f"добавлено задумок ведущего: {len(added)}",
        }

    def _generate_one(
        self, session_id: int, scene: dict[str, Any], settings: Any, timings: dict[str, float]
    ) -> dict[str, Any]:
        """Генерирует одну сцену и записывает результат."""
        # Образец места нужен при возвращении, а не на каждом кадре подряд.
        # Образец — это картинка целиком, вместе с моментом: позой, светом и
        # действием. Если подкладывать его в каждой сцене одного места, лечение,
        # подъём и разговор выйдут одним и тем же кадром. Поэтому подряд идущие
        # кадры одного места рисуются свободно по новому описанию, а образец
        # возвращается, когда действие приходит сюда из другого места.
        place_name, reference = self._reference_for_scene(session_id, scene)
        # Что рисуется сейчас — это показывает интерфейс, чтобы долгая генерация
        # не выглядела зависанием.
        self._current_scene = {
            "id": scene["id"],
            "prompt": scene["prompt"],
            "reference": bool(reference),
            "queue": list(getattr(self, "_scene_queue", [])),
        }
        self.comfy_models_loaded = True
        seed = self._pick_seed(session_id, scene)
        # Дополнение мира приписывается дословно и в самом конце: ведущий его не
        # видит, переписать не может, а генератор получает.
        final_prompt = self._with_world_suffix(session_id, scene["prompt"])
        # Галочка «призрачные очертания»: мужские фигуры заменяются на месте,
        # уже после дополнения мира, чтобы заменить и то, что написано в нём.
        if getattr(settings, "male_silhouette", False):
            final_prompt = prompts.shape_males(final_prompt)
        # Ступень качества можно задать на один кадр — так работает
        # «перерисовать крупнее» на карточке кадра.
        size, steps = settings.image_dimensions(scene.get("quality"))
        graph = build_t2i_graph(
            final_prompt,
            width=size,
            height=size,
            seed=seed,
            steps=steps,
            cfg=settings.image_cfg,
            filename_prefix=f"novel_{scene['id']:05d}",
            reference=reference,
        )
        if reference:
            self.note(f"сцена #{scene['id']} рисуется по образцу места")
        elif place_name:
            self.note(f"сцена #{scene['id']}: знакомое место, но новый момент — рисуется свободно")
        sampler = metrics.Sampler()
        sampler.start()
        try:
            result = self.comfy.generate(graph, timeout_s=settings.image_timeout_s)
            summary = sampler.summary().as_dict()
            path = result.first_image
            self.db.finish_scene(
                scene["id"], None if path is None else str(path), "done", result.elapsed_s,
                used_reference=bool(reference),
            )
            timings[f"image_{scene['id']}"] = result.elapsed_s
            if path is not None:
                self._adopt_reference(scene.get("location_id"), path)
            # Запоминаем место именно этого кадра: по нему следующий кадр решает,
            # продолжает он знакомое место или вернулся в него.
            self.db.set_state(session_id, "last_image_location", place_name)
            self.db.set_state(session_id, "last_image_seed", seed)
            self.db.set_scene_seed(scene["id"], seed)
            self.note(
                f"сцена #{scene['id']} готова за {result.elapsed_s:.1f} c, "
                f"пик VRAM {summary['gpu_used_mb']['max']:.0f} MB"
            )
            return {
                "scene_id": scene["id"],
                "path": None if path is None else str(path),
                "elapsed_s": round(result.elapsed_s, 2),
                "peak_vram_mb": summary["gpu_used_mb"]["max"],
                "prompt": scene["prompt"],
                "used_reference": bool(reference),
            }
        except ComfyError as exc:
            self.db.finish_scene(scene["id"], None, "failed")
            self.note(f"сцена #{scene['id']} не сгенерирована: {exc}")
            return {"scene_id": scene["id"], "error": str(exc), "prompt": scene["prompt"]}
        finally:
            sampler.stop()
            self._current_scene = None

    def _adopt_reference(self, location_id: int | None, path: Path) -> None:
        """Назначает первый удачный кадр места его образцом.

        Образец задаётся один раз: если обновлять его каждым кадром, внешний вид
        поедет от кадра к кадру — ровно то, от чего образец и должен спасать.
        Пользователь может назначить образцом любой другой кадр вручную.

        @param location_id: место сцены.
        @param path: путь к готовому кадру.
        """
        if location_id is None:
            return
        location = self.db.location(location_id)
        if location is None or location.reference_path:
            return
        self.db.set_location_reference(location_id, str(path))
        self.note(f"кадр места «{location.name}» стал образцом для следующих")

    def generate_pending_now(
        self, session_id: int, scene_ids: list[int] | None = None,
        quality: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ручной запуск генерации сцен из интерфейса.

        @param session_id: партия.
        @param scene_ids: рисовать только эти сцены; ``None`` — все ожидающие.
        @param quality: ступень качества на этот заход.
        @returns: отчёты по нарисованным кадрам.
        """
        with self._busy:
            outcome = TurnOutcome()
            try:
                if not self._pending_scenes(session_id, scene_ids, quality):
                    return []
                return self._image_phase(session_id, outcome, quality=quality)
            except (ComfyError, FreeTokenError, RuntimeError) as exc:
                self.note(f"ошибка генерации: {exc}")
                self._set_state(State.ERROR)
                return [{"error": str(exc)}]
            finally:
                if self.state != State.ERROR:
                    self._set_state(State.IDLE)
                outcome.total_s = 0.0
                self.last_outcome = outcome

    def generate_draft(self, prompt: str, seed: int | None = None) -> dict[str, Any]:
        """Черновая генерация малым разрешением — заготовка для спекуляции."""
        settings = self.settings
        graph = build_t2i_graph(
            prompt,
            width=settings.draft_size,
            height=settings.draft_size,
            seed=seed or 0,
            steps=settings.draft_steps,
            cfg=settings.image_cfg,
            filename_prefix="novel_draft",
        )
        result = self.comfy.generate(graph, timeout_s=settings.image_timeout_s)
        return {
            "path": None if result.first_image is None else str(result.first_image),
            "elapsed_s": round(result.elapsed_s, 2),
        }
