"""Клиент текстового движка FreeToken.

Оркестратор общается с движком только по HTTP. CLI ``ft`` существует, но в
PowerShell ``ft`` — это алиас ``Format-Table``, а сам CLI всё равно ходит в те
же самые эндпоинты, поэтому единственный источник истины — HTTP.

Управление памятью идёт через два эндпоинта:

* ``GET /v1/cache/status`` — текущая геометрия пулов и лимиты;
* ``POST /v1/cache/rebuild`` — пересборка пулов под новую геометрию.

Сервер сам разрешает запрошенные размеры к ближайшей допустимой геометрии, а
ответ содержит то, что получилось на самом деле, — поэтому результат всегда
читается обратно из ``/v1/cache/status``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from novel import config

MB = 1024 * 1024


class FreeTokenError(RuntimeError):
    """Движок недоступен или отказал в запросе."""


@dataclass
class CacheGeometry:
    """Разобранная геометрия пулов из ``/v1/cache/status``."""

    num_pages: int
    page_size: int
    moe_cache_size: int
    num_mamba_slots: int
    num_swa_pages: int
    swa_page_size: int
    num_experts: int
    num_moe_layers: int
    moe_cache_policy: str
    kv_per_token: int
    moe_per_expert: int
    mamba_per_slot: int
    swa_per_token: int
    cache_budget_bytes: int
    limits: dict[str, Any]

    @classmethod
    def from_document(cls, geometry: dict[str, Any]) -> "CacheGeometry":
        """Собирает геометрию из поля ``geometry`` ответа сервера."""
        units = geometry.get("unit_bytes") or {}
        return cls(
            num_pages=int(geometry.get("num_pages", 0)),
            page_size=int(geometry.get("page_size", 1) or 1),
            moe_cache_size=int(geometry.get("moe_cache_size", 0)),
            num_mamba_slots=int(geometry.get("num_mamba_slots", 0)),
            num_swa_pages=int(geometry.get("num_swa_pages", 0)),
            swa_page_size=int(geometry.get("swa_page_size", 1) or 1),
            num_experts=int(geometry.get("num_experts", 0)),
            num_moe_layers=int(geometry.get("num_moe_layers", 0)),
            moe_cache_policy=str(geometry.get("moe_cache_policy", "")),
            kv_per_token=int(units.get("kv_per_token", 0)),
            moe_per_expert=int(units.get("moe_per_expert", 0)),
            mamba_per_slot=int(units.get("mamba_per_slot", 0)),
            swa_per_token=int(units.get("swa_per_token", 0)),
            cache_budget_bytes=int(geometry.get("cache_budget_bytes", 0)),
            limits=geometry.get("limits") or {},
        )

    def bytes_for(
        self,
        *,
        moe: int | None = None,
        kv_tokens: int | None = None,
        swa_tokens: int | None = None,
        mamba_slots: int | None = None,
    ) -> dict[str, int]:
        """Считает объём пулов для заданной геометрии, байты.

        Пропущенные пулы берутся в текущем размере.
        @returns: ``kv``, ``swa``, ``moe``, ``mamba``, ``total``.
        """
        kv = self.num_pages if kv_tokens is None else kv_tokens
        swa = self.num_swa_pages if swa_tokens is None else swa_tokens
        moe = self.moe_cache_size if moe is None else moe
        mamba = self.num_mamba_slots if mamba_slots is None else mamba_slots
        parts = {
            "kv": kv * self.kv_per_token,
            "swa": swa * self.swa_per_token,
            "moe": moe * self.moe_per_expert,
            "mamba": mamba * self.mamba_per_slot,
        }
        parts["total"] = sum(parts.values())
        return parts

    def describe(self) -> str:
        """Человекочитаемая строка текущей геометрии."""
        sizes = self.bytes_for()
        return (
            f"moe={self.moe_cache_size} слотов ({sizes['moe'] / MB:.0f} MB), "
            f"kv={self.num_pages} ток. ({sizes['kv'] / MB:.0f} MB), "
            f"swa={self.num_swa_pages} ток. ({sizes['swa'] / MB:.0f} MB)"
        )


@dataclass
class ChatResult:
    """Ответ движка на запрос чата вместе с таймингами."""

    text: str
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any]
    ttft_s: float | None = None

    @property
    def tokens_per_second(self) -> float:
        """Скорость генерации, токенов в секунду."""
        if self.elapsed_s <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_s

    @property
    def decode_tokens_per_second(self) -> float:
        """Скорость генерации без учёта задержки до первого токена."""
        if self.ttft_s is None:
            return self.tokens_per_second
        span = self.elapsed_s - self.ttft_s
        if span <= 0 or self.completion_tokens <= 1:
            return 0.0
        return (self.completion_tokens - 1) / span


class FreeTokenClient:
    """HTTP-клиент одного запущенного сервера FreeToken.

    Клиент помнит, как для текущей модели выключаются размышления. Это нужно
    потому, что reasoning-модели (Qwen3.5/3.6, gpt-oss) по умолчанию тратят
    бюджет ответа на длинную цепочку рассуждений и до текста не доходят:
    снаружи это выглядит как пустой ответ при ненулевых токенах. Движок сам
    сообщает точные аргументы в ``/v1/cache/status``, поэтому они не
    зашиваются в код.
    """

    def __init__(self, base_url: str = config.FREETOKEN_BASE_URL, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        #: Дополнительные поля запроса, выключающие размышления, или ``None``.
        self.reasoning_kwargs: dict[str, Any] | None = None

    # --- транспорт ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_s or self.timeout_s
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FreeTokenError(f"HTTP {exc.code} на {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FreeTokenError(f"{self.base_url} недоступен: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FreeTokenError(f"таймаут запроса к {self.base_url}{path}") from exc
        except OSError as exc:
            # Движок выгружают штатно: перед генерацией картинки, при смене модели
            # и кнопкой «Стоп всё». Оборванное на середине соединение даёт
            # ConnectionResetError, который не является URLError, и без этой
            # ветки исключение уходило бы мимо обработки.
            raise FreeTokenError(f"соединение с {self.base_url} оборвано: {exc}") from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # --- наблюдение ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /health`` — жив ли сервер и в какой фазе."""
        return self._request("GET", "/health", timeout_s=5.0)

    def stats(self) -> dict[str, Any]:
        """``GET /v1/stats`` — занятая VRAM, счётчики запросов, пропускная способность."""
        return self._request("GET", "/v1/stats")

    def models(self) -> list[str]:
        """Идентификаторы моделей, которые отдаёт ``GET /v1/models``."""
        doc = self._request("GET", "/v1/models")
        return [entry["id"] for entry in doc.get("data", [])]

    def cache_status(self) -> dict[str, Any]:
        """``GET /v1/cache/status`` целиком."""
        return self._request("GET", "/v1/cache/status")

    def geometry(self) -> CacheGeometry:
        """Текущая геометрия пулов."""
        doc = self.cache_status()
        return CacheGeometry.from_document(doc.get("geometry") or {})

    def is_idle(self) -> bool:
        """Нет ли прямо сейчас активных запросов."""
        return int(self.stats().get("requests", {}).get("active", 0)) == 0

    # --- управление кэшем ---------------------------------------------------

    def cache_rebuild(
        self,
        *,
        moe_slots: int | None = None,
        kv_tokens: int | None = None,
        swa_tokens: int | None = None,
        mamba_slots: int | None = None,
        wait_s: float = 300.0,
        mode: str = "if_idle",
    ) -> tuple[dict[str, Any], float]:
        """Пересобирает пулы кэша под новую геометрию.

        Переданные размеры — это то, что мы *просим*; сервер разрешает их сам и
        возвращает фактическую геометрию в ``last_rebuild``. Токены KV и SWA
        передаются в токенах, как их и показывает ``cache status``.

        @param mode: ``if_idle`` пересобирает только при отсутствии активных запросов.
        @returns: ``(ответ сервера, длительность операции в секундах)``.
        @raises FreeTokenError: если сервер отказал или вернул статус не ``ok``.
        """
        body: dict[str, Any] = {"mode": mode, "timeout": wait_s}
        if moe_slots is not None:
            body["moe_cache_size"] = moe_slots
        if kv_tokens is not None:
            body["num_pages"] = kv_tokens
        if swa_tokens is not None:
            body["num_swa_pages"] = swa_tokens
        if mamba_slots is not None:
            body["num_mamba_slots"] = mamba_slots

        started = time.time()
        doc = self._request("POST", "/v1/cache/rebuild", body, timeout_s=wait_s + 30.0)
        elapsed = time.time() - started
        if doc.get("status") != "ok":
            raise FreeTokenError(f"пересборка кэша не удалась: {doc}")
        return doc, elapsed

    def cache_limits(self) -> dict[str, Any]:
        """Допустимые диапазоны размеров пулов из ``geometry.limits``."""
        return self.cache_status().get("geometry", {}).get("limits", {})

    def input_modalities(self) -> list[str]:
        """Что текущая модель принимает на входе.

        @returns: список вида ``["text"]`` или ``["text", "image"]``; пустой,
            если движок не сообщил.
        """
        try:
            model = (self.stats().get("model") or {})
        except FreeTokenError:
            return []
        modalities = model.get("input_modalities")
        return [str(item) for item in modalities] if isinstance(modalities, list) else []

    def supports_images(self) -> bool:
        """Принимает ли текущая модель изображения."""
        return "image" in self.input_modalities()

    def reasoning_control(self) -> dict[str, Any]:
        """Что движок сообщает об управлении размышлениями этой модели.

        @returns: ``default`` (``on``/``off``), ``gears`` и ``kwargs`` для
            каждого положения, либо пустой словарь, если модель не умеет.
        """
        geometry = self.cache_status().get("geometry") or {}
        control = geometry.get("reasoning") or {}
        return control if isinstance(control, dict) else {}

    def configure_reasoning(self, mode: str = "off") -> dict[str, Any]:
        """Настраивает клиент так, чтобы размышления были выключены или включены.

        Для роли ведущего размышления вредны: они съедают бюджет ответа и время,
        а игрок их всё равно не видит.

        @param mode: ``off``, ``on`` или ``auto`` (оставить как у модели).
        @returns: отчёт о том, что удалось выяснить и применить.
        """
        try:
            control = self.reasoning_control()
        except FreeTokenError as exc:
            self.reasoning_kwargs = None
            return {"mode": mode, "applied": False, "error": str(exc)}

        if not control or mode == "auto":
            self.reasoning_kwargs = None
            return {
                "mode": mode,
                "model_default": control.get("default"),
                "applied": False,
                "reason": "у модели нет рычагов размышлений" if not control else "режим auto",
            }

        gear = "off" if mode == "off" else "on"
        by_gear = control.get("kwargs") or {}
        gears = [str(name) for name in (control.get("gears") or [])]
        kwargs = by_gear.get(gear)
        chosen = gear
        if kwargs is None and gears:
            # У модели может не быть положения «выключено»: gpt-oss знает только
            # слабое, среднее и сильное. Тогда берётся крайнее доступное — самое
            # слабое для «off» и самое сильное для «on». Иначе размышления
            # остаются на умолчании модели и съедают весь бюджет ответа.
            chosen = gears[0] if mode == "off" else gears[-1]
            kwargs = by_gear.get(chosen)
        self.reasoning_kwargs = dict(kwargs) if kwargs else None
        return {
            "mode": mode,
            "model_default": control.get("default"),
            "gear": chosen,
            "applied": self.reasoning_kwargs is not None,
            "kwargs": kwargs,
        }

    def _with_reasoning(self, body: dict[str, Any]) -> dict[str, Any]:
        """Добавляет в запрос управление размышлениями, если оно настроено.

        Поля раскладываются по двум местам, и это не прихоть: ``reasoning_effort``
        серверы читают полем верхнего уровня, а всё, что касается шаблона чата
        (``enable_thinking``, ``thinking_mode``), обязано лежать внутри
        ``chat_template_kwargs``. Если положить шаблонное поле наверх, оно
        пропадает молча, и модель продолжает размышлять.
        """
        if not self.reasoning_kwargs:
            return body
        template = dict(body.get("chat_template_kwargs") or {})
        template.update(self.reasoning_kwargs.get("chat_template_kwargs") or {})
        for key, value in self.reasoning_kwargs.items():
            if key == "chat_template_kwargs":
                continue
            if key == "reasoning_effort":
                body["reasoning_effort"] = value
            else:
                template[key] = value
        if template:
            body["chat_template_kwargs"] = template
        return body

    def count_tokens(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> int:
        """Считает токены запроса, не генерируя ответ.

        Счёт идёт через ``/v1/messages/count_tokens`` — это Anthropic-формат,
        поэтому системное сообщение передаётся отдельным полем, а не первым
        элементом списка.

        @param messages: сообщения диалога без системного.
        @param system: системная инструкция, если есть.
        @returns: число токенов входа.
        @raises FreeTokenError: если сервер недоступен.
        """
        body: dict[str, Any] = {
            "model": model or config.FREETOKEN_SERVED_NAME,
            "messages": [
                {"role": item["role"], "content": item.get("content", "")} for item in messages
            ],
        }
        if system:
            body["system"] = system
        doc = self._request("POST", "/v1/messages/count_tokens", body, timeout_s=60.0)
        for key in ("input_tokens", "tokens", "total_tokens", "count"):
            if key in doc:
                return int(doc[key])
        return 0

    # --- генерация ----------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        timeout_s: float = 600.0,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Обычный (не потоковый) запрос к ``/v1/chat/completions``.

        @param messages: список сообщений в формате OpenAI.
        @param extra: дополнительные поля запроса, например ``chat_template_kwargs``.
        @returns: текст ответа, тайминги и счётчики токенов.
        @raises FreeTokenError: если сервер недоступен или вернул пустой выбор.
        """
        body: dict[str, Any] = {
            "model": model or config.FREETOKEN_SERVED_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop:
            body["stop"] = stop
        if extra:
            body.update(extra)
        self._with_reasoning(body)

        started = time.time()
        doc = self._request("POST", "/v1/chat/completions", body, timeout_s=timeout_s)
        elapsed = time.time() - started

        choices = doc.get("choices") or []
        if not choices:
            raise FreeTokenError(f"пустой ответ движка: {doc}")
        usage = doc.get("usage") or {}
        return ChatResult(
            text=choices[0].get("message", {}).get("content", "") or "",
            elapsed_s=elapsed,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=doc,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.8,
        model: str | None = None,
        timeout_s: float = 600.0,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Потоковый запрос: тот же чат, но с замером задержки до первого токена.

        Задержка до первого токена — главная величина, по которой видно цену
        холодного кэша экспертов после пересборки пулов.

        @returns: текст ответа, время до первого токена и общая длительность.
        @raises FreeTokenError: если сервер недоступен или поток оборвался.
        """
        body: dict[str, Any] = {
            "model": model or config.FREETOKEN_SERVED_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra:
            body.update(extra)
        self._with_reasoning(body)

        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
            method="POST",
        )

        started = time.time()
        ttft: float | None = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        errors: list[str] = []
        finish_reason: str | None = None
        reasoning_chars = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # Сервер сообщает об отказе отдельным событием и закрывает поток,
                    # не присылая ни одного токена; без этой ветки отказ выглядел бы
                    # как успешный пустой ответ.
                    if event.get("error"):
                        errors.append(json.dumps(event["error"], ensure_ascii=False))
                    if event.get("detail"):
                        errors.append(str(event["detail"]))
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        for key in ("reasoning_content", "reasoning", "thinking"):
                            piece = delta.get(key)
                            if piece:
                                reasoning_chars += len(piece)
                        piece = delta.get("content")
                        if piece:
                            if ttft is None:
                                ttft = time.time() - started
                            chunks.append(piece)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FreeTokenError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FreeTokenError(f"{self.base_url} недоступен: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FreeTokenError(f"таймаут потока от {self.base_url}") from exc
        except OSError as exc:
            raise FreeTokenError(f"соединение с {self.base_url} оборвано: {exc}") from exc

        if errors:
            raise FreeTokenError("движок отказал: " + "; ".join(errors))
        if not chunks and reasoning_chars:
            raise FreeTokenError(
                f"модель израсходовала бюджет на размышления ({reasoning_chars} символов) "
                f"и не начала ответ (finish_reason={finish_reason!r}). "
                "Выключи размышления в настройках модели или подними max_tokens."
            )
        if not chunks and not usage:
            raise FreeTokenError(
                "поток закрылся без токенов и без usage "
                f"(finish_reason={finish_reason!r}) — вероятно, не хватило кэша"
            )

        return ChatResult(
            text="".join(chunks),
            elapsed_s=time.time() - started,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw={"usage": usage, "finish_reason": finish_reason, "reasoning_chars": reasoning_chars},
            ttft_s=ttft,
        )
