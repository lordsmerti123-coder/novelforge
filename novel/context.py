"""Сборка контекста: слои, бюджет токенов, скользящее окно, суммаризация.

Контекст собирается слоями, и каждый слой можно посмотреть по отдельности
вместе с его стоимостью в токенах. Это и есть основной инструмент отладки:
видно, что именно съедает окно — правила, персонажи или разросшаяся история.

Порядок слоёв выбран так, чтобы неизменная часть шла первой: движок кэширует
префикс запроса, и на каждом ходу пересчитывается только хвост.

    формат -> мир -> правила -> персонажи -> стиль -> состояние -> память -> окно

Окно — единственный слой, который меняется от хода к ходу. Когда истории
становится больше, чем помещается, старые сообщения сворачиваются в
суммаризацию: она поглощает предыдущую, поэтому память не растёт бесконечно.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from novel import prompts
from novel.db import Memory, Message, NovelDB
from novel.formats import DialogueFormat, get_format
from novel.freetoken import FreeTokenClient, FreeTokenError
from novel.vision import user_message

#: Сколько символов приходится на один токен при грубой оценке.
#: Русский текст даёт примерно 2.5-3.5 символа на токен; берём осторожное значение,
#: чтобы оценка скорее завышала, чем занижала.
CHARS_PER_TOKEN = 2.8


def _item_rows(items: list[Any], limit: int) -> list[tuple[str, str, int]]:
    """Готовит вещи владельца к показу в промпте.

    @param items: вещи одного владельца.
    @param limit: сколько показывать целиком.
    @returns: строки ``(имя, свойства, количество)``; лишние сворачиваются.
    """
    rows = [(item.name, item.properties, item.quantity) for item in items[:limit]]
    hidden = len(items) - len(rows)
    if hidden > 0:
        rows.append((f"и ещё {hidden}", "", 1))
    return rows


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов по длине текста.

    Точный подсчёт стоит HTTP-запроса, поэтому окно сначала подбирается оценкой,
    а точное число проверяется один раз на готовой сборке.

    @param text: любой текст.
    @returns: оценка числа токенов.
    """
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Layer:
    """Один слой промпта."""

    key: str
    title: str
    text: str
    tokens: int
    stable: bool

    def as_dict(self) -> dict[str, Any]:
        """Представление для отладочной панели."""
        return {
            "key": self.key,
            "title": self.title,
            "text": self.text,
            "tokens": self.tokens,
            "stable": self.stable,
            "chars": len(self.text),
        }


@dataclass
class AssembledPrompt:
    """Готовый запрос вместе с разбором по слоям."""

    messages: list[dict[str, str]]
    layers: list[Layer]
    system_text: str
    total_tokens: int
    budget_tokens: int
    response_reserve: int
    included_message_ids: list[int] = field(default_factory=list)
    dropped_message_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exact_tokens: bool = False

    @property
    def available_tokens(self) -> int:
        """Сколько токенов остаётся под ответ модели."""
        return max(0, self.budget_tokens - self.total_tokens)

    @property
    def transcript_tokens(self) -> int:
        """Сколько токенов занимает дословная история."""
        return sum(layer.tokens for layer in self.layers if layer.key == "window")

    def chat_messages(self) -> list[dict[str, str]]:
        """Сообщения для запроса: системная инструкция идёт первой.

        ``/v1/chat/completions`` ждёт системную инструкцию элементом списка, а не
        отдельным полем, поэтому здесь она подставляется в начало. Отдельно
        ``system_text`` хранится для подсчёта токенов: там формат Anthropic, и
        системная инструкция передаётся своим полем.

        @returns: список сообщений в формате OpenAI.
        """
        if self.system_text.strip():
            return [{"role": "system", "content": self.system_text}, *self.messages]
        return list(self.messages)

    def as_dict(self, include_text: bool = True) -> dict[str, Any]:
        """Разбор сборки для отладочной панели.

        @param include_text: включать ли полный текст слоёв.
        @returns: словарь с послойной разбивкой и предупреждениями.
        """
        layers = [
            layer.as_dict() if include_text else {**layer.as_dict(), "text": ""}
            for layer in self.layers
        ]
        return {
            "layers": layers,
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "response_reserve": self.response_reserve,
            "available_tokens": self.available_tokens,
            "stable_tokens": sum(layer.tokens for layer in self.layers if layer.stable),
            "included_messages": len(self.included_message_ids),
            "dropped_messages": len(self.dropped_message_ids),
            "warnings": self.warnings,
            "exact_tokens": self.exact_tokens,
        }


class ContextBuilder:
    """Собирает контекст для одного хода и следит за бюджетом токенов."""

    def __init__(
        self,
        db: NovelDB,
        ft: FreeTokenClient,
        settings_provider: Callable[[], Any],
        attachment_loader: Callable[[list[str]], list[str]] | None = None,
    ) -> None:
        self.db = db
        self.ft = ft
        self.settings_provider = settings_provider
        self.attachment_loader = attachment_loader

    # --- слои ---------------------------------------------------------------

    def stable_layers(self, session_id: int) -> tuple[list[Layer], DialogueFormat]:
        """Слои, не зависящие от текущего хода.

        @param session_id: партия.
        @returns: список слоёв и формат диалога мира.
        """
        session = self.db.session(session_id)
        world = self.db.world(session.world_id) if session else None
        fmt = get_format(world.format if world else "story")

        blocks: list[tuple[str, str, str]] = [
            # Политика картинок влияет на инструкцию: при «каждом ходу» ведущий
            # обязан ставить блок сцены в каждом ответе, иначе политике нечего
            # рисовать. В переписке кадр — портрет собеседника, если так решено.
            ("format", "Формат и протокол", prompts.base_instruction(
                fmt,
                getattr(self.settings_provider(), "image_policy", "minimal"),
                self._portrait_mode(fmt),
            )),
        ]
        if world is not None:
            blocks.append(("world", "Мир", prompts.world_block(world)))
            blocks.append(("rules", "Правила", prompts.rules_block(self.db.rules(world.id, enabled_only=True))))
            blocks.append(
                ("characters", "Персонажи", prompts.characters_block(self.db.characters(world.id, enabled_only=True)))
            )
            blocks.append(("style", "Стиль изображений", prompts.style_block(world, fmt)))
            if getattr(self.settings_provider(), "location_memory", True):
                names = [item.name for item in self.db.locations(world.id, limit=12)]
                blocks.append(("places", "Известные места", prompts.places_block(names)))
            if getattr(self.settings_provider(), "items_memory", True):
                blocks.append(("items", "Вещи", self._items_layer(world)))
            # Внешность «сейчас» живёт в состоянии партии, а не в истории:
            # сводка сжимает подробности, и после неё ведущий путает, кто есть кто.
            looks = self.db.get_state(session_id, "looks", {}) or {}
            if isinstance(looks, dict) and looks:
                blocks.append((
                    "looks", "Внешность сейчас",
                    prompts.looks_block([(str(k), str(v)) for k, v in looks.items()]),
                ))
            if world.hidden_rules.strip():
                blocks.append(("hidden", "Скрытые правила", world.hidden_rules.strip()))
        blocks.append(("state", "Состояние", prompts.state_block(self.db.world_state(session_id))))
        # Задумки игрока и ведущего идут одним слоем, но разными разделами:
        # первые — приказ, вторые — предположение. Возраст считается по числу
        # сообщений, прошедших с момента задумки.
        last_message = self.db.count_messages(session_id)
        planned = [
            {
                "id": item.id,
                "text": item.text,
                "source": item.source,
                "horizon": item.horizon,
                "age": max(0, last_message - (item.anchor_message_id or last_message)) // 2,
            }
            for item in self.db.notes(session_id, limit=7)
        ]
        blocks.append(("notes", "Задумано на потом", prompts.notes_block(planned)))
        blocks.append(("memory", "Память", prompts.memory_block(self.db.memories(session_id))))

        # Напоминание о стиле идёт последним среди постоянных слоёв: оно должно
        # стоять вплотную к истории, иначе длинная проза в окне перебивает
        # инструкцию формата — модель просто продолжает то, что видит.
        reminder = prompts.style_reminder(fmt, self._persona_lines(session_id))
        if reminder:
            blocks.append(("reminder", "Напоминание о стиле", reminder))

        layers = [
            Layer(key=key, title=title, text=text, tokens=estimate_tokens(text), stable=True)
            for key, title, text in blocks
            if text.strip()
        ]
        return layers, fmt

    def _persona_lines(self, session_id: int, limit: int = 3) -> list[str]:
        """Короткие подсказки о характере собеседников.

        Карточки персонажей идут в начале промпта, а инструкция формата — в
        конце, и модель держится последнего, что видит. На практике из-за
        этого ведущий соблюдал формат переписки, но говорил общими словами:
        «Сядь удобно. Закрой глаза», хотя персонаж задуман суровым и властным. Поэтому
        характер повторяется рядом с напоминанием о стиле.

        @param session_id: партия.
        @param limit: сколько собеседников упомянуть.
        @returns: строки вида «Имя — чем живёт».
        """
        session = self.db.session(session_id)
        if session is None:
            return []
        lines: list[str] = []
        for character in self.db.characters(session.world_id, enabled_only=True)[:limit]:
            traits = " ".join(
                part.strip() for part in (character.speech, character.description) if part.strip()
            )
            traits = " ".join(traits.split())
            if len(traits) > 160:
                traits = traits[:157].rstrip() + "…"
            lines.append(f"{character.name.strip()}: {traits}" if traits else character.name.strip())
        return lines

    def window(
        self,
        session_id: int,
        budget_tokens: int,
        fmt: DialogueFormat,
        *,
        hard_limit: int | None = None,
        extra: str = "",
        exclude_ids: set[int] | None = None,
    ) -> tuple[list[Message], list[Message]]:
        """Подбирает дословное окно истории под бюджет.

        Идём от свежих сообщений к старым и набираем, пока помещается. Свежая
        пара «реплика игрока — ответ ведущего» ценнее старой, поэтому окно
        строится с конца.

        @param session_id: партия.
        @param budget_tokens: сколько токенов доступно окну и текущей реплике.
        @param fmt: формат диалога.
        @param hard_limit: жёсткий предел числа сообщений из настроек.
        @param extra: текст текущей реплики, который тоже занимает место.
        @param exclude_ids: сообщения, которые в окно не попадают. Текущая
            реплика уже записана в хранилище, но в промпт добавляется отдельно,
            поэтому без исключения она оказалась бы там дважды.
        @returns: ``(включённые сообщения, выброшенные сообщения)``.
        """
        all_messages = self.db.messages(session_id)
        skipped = exclude_ids or set()
        # Всё, что уже покрыто суммаризацией, в окно не попадает: иначе старые
        # сообщения одновременно и лежали бы в памяти, и считались выброшенными,
        # а признак «пора сворачивать» срабатывал бы на каждом ходу заново.
        memory = self.db.latest_memory(session_id)
        covered = memory.through_message_id if memory else 0
        all_messages = [
            message
            for message in all_messages
            if message.id > covered and message.id not in skipped
        ]
        if not all_messages:
            return [], []

        remaining = budget_tokens - estimate_tokens(extra)
        included: list[Message] = []
        for message in reversed(all_messages):
            cost = estimate_tokens(message.content) + 4
            if cost > remaining:
                break
            if hard_limit is not None and len(included) >= hard_limit:
                break
            remaining -= cost
            included.append(message)
        included.reverse()
        dropped = all_messages[: len(all_messages) - len(included)]
        return included, dropped

    # --- сборка -------------------------------------------------------------

    def build(
        self,
        session_id: int,
        user_input: str,
        *,
        interject: str = "",
        solo: bool = False,
        count_exactly: bool = True,
        exclude_ids: set[int] | None = None,
    ) -> AssembledPrompt:
        """Собирает запрос к модели целиком.

        @param session_id: партия.
        @param user_input: текущая реплика игрока; пустая, если игрок не действует.
        @param interject: врезка пользователя, действующая только на этот ход.
        @param solo: врезка без реплики игрока.
        @param count_exactly: уточнять ли число токенов запросом к движку.
        @param exclude_ids: уже записанные сообщения, которые не должны попасть
            в окно — прежде всего сама текущая реплика.
        @returns: сообщения, послойный разбор и предупреждения.
        """
        settings = self.settings_provider()
        warnings: list[str] = []
        budget = int(getattr(settings, "context_budget_tokens", 16384))
        reserve = int(getattr(settings, "max_tokens", 900))

        stable, fmt = self.stable_layers(session_id)
        stable_tokens = sum(layer.tokens for layer in stable)
        window_budget = budget - reserve - stable_tokens
        if window_budget <= 0:
            warnings.append(
                f"неизменяемая часть промпта ({stable_tokens} токенов) не оставляет места "
                f"под историю: бюджет {budget}, резерв под ответ {reserve}"
            )
            window_budget = 0

        injected = prompts.interject_block(interject, solo=solo)
        included, dropped = self.window(
            session_id,
            window_budget,
            fmt,
            hard_limit=int(getattr(settings, "context_turns", 12)) or None,
            extra=user_input,
            exclude_ids=exclude_ids,
        )
        if dropped:
            warnings.append(
                f"{len(dropped)} старых сообщений не поместились в окно — "
                f"их стоит свернуть в суммаризацию"
            )

        window_text = prompts.transcript(included) if included else ""
        window_layer = Layer(
            key="window",
            title=f"История, {len(included)} сообщений",
            text=window_text,
            tokens=estimate_tokens(window_text) if window_text else 0,
            stable=False,
        )

        system_text = "\n\n".join(layer.text for layer in stable if layer.text.strip())
        messages = self._render_window(included, settings)
        user_text = user_input
        if injected:
            # Реплики может не быть вовсе: тогда уходит только врезка, без
            # висящих переводов строки.
            user_text = f"{injected}\n\n{user_input}" if user_input.strip() else injected
        messages.append({"role": "user", "content": user_text})

        layers = [*stable, window_layer]
        total = sum(layer.tokens for layer in layers) + estimate_tokens(user_input)
        exact = False
        if count_exactly:
            total, exact = self._exact_count(messages, system_text, total, warnings)
            if exact and getattr(settings, "debug_mode", False):
                self._refine_layer_tokens(layers, messages, warnings)

        return AssembledPrompt(
            messages=messages,
            layers=layers,
            system_text=system_text,
            total_tokens=total,
            budget_tokens=budget,
            response_reserve=reserve,
            included_message_ids=[message.id for message in included],
            dropped_message_ids=[message.id for message in dropped],
            warnings=warnings,
            exact_tokens=exact,
        )

    def _portrait_mode(self, fmt: Any) -> bool:
        """Рисовать ли в этом формате портрет собеседника вместо сцены.

        Портрет имеет смысл только в переписке, где игрок говорит с одним
        человеком: там кадр работает карточкой собеседника. В прозе идёт
        история, и портрет в ней был бы неуместен.

        @param fmt: формат диалога.
        @returns: True, если ведущий должен описывать портрет.
        """
        if not fmt.wants_scene or fmt.ui not in ("chat", "photo_chat"):
            return False
        return getattr(self.settings_provider(), "chat_frame", "portrait") == "portrait"

    def _items_layer(self, world: Any) -> str:
        """Собирает список вещей по владельцам.

        Вещей может быть много, а промпт не резиновый: на каждого владельца
        показываются первые несколько, остальные сворачиваются в «и ещё N».

        @param world: мир.
        @returns: текстовый блок либо пустая строка.
        """
        limit = 8
        groups: list[tuple[str, list[tuple[str, str, int]]]] = []

        player_items = self.db.items(world.id, character_id=None)
        # Имена не склоняются: «У Барон Готтхольд» читалось бы как ошибка, а
        # угадывать падежи русских имён программа не умеет.
        groups.append(("Игрок", _item_rows(player_items, limit)))

        for character in self.db.characters(world.id):
            owned = self.db.items(world.id, character_id=character.id)
            if owned:
                groups.append((character.name, _item_rows(owned, limit)))
        return prompts.items_block(groups)

    def _render_window(self, included: list[Message], settings: Any) -> list[dict[str, Any]]:
        """Превращает сообщения окна в элементы запроса.

        Изображения из прошлых реплик пересылаются только если это разрешено
        настройкой: каждое стоит около 258 токенов входа и удлиняет ответ, а
        польза от старых кадров невелика. По умолчанию пересылается только то,
        что приложено к текущей реплике, — этим занимается оркестратор.

        @param included: сообщения, попавшие в окно.
        @param settings: настройки с числом пересылаемых вложений.
        @returns: список сообщений в формате OpenAI; у части может быть
            многочастное содержимое.
        """
        limit = int(getattr(settings, "history_images", 0) or 0)
        if getattr(settings, "vision_mode", "on_attach") == "off":
            # Зрение выключено целиком: старые вложения тоже не пересылаются.
            limit = 0
        if not limit or self.attachment_loader is None:
            return [{"role": m.role, "content": m.content} for m in included]

        with_images = [m for m in included if m.attachment_paths]
        selected = {m.id for m in with_images[-limit:]}
        rendered: list[dict[str, Any]] = []
        for message in included:
            if message.id in selected:
                urls = self.attachment_loader(message.attachment_paths)
                rendered.append(user_message(message.content, urls))
            else:
                rendered.append({"role": message.role, "content": message.content})
        return rendered

    def _exact_count(
        self,
        messages: list[dict[str, str]],
        system_text: str,
        fallback: int,
        warnings: list[str],
    ) -> tuple[int, bool]:
        """Уточняет число токенов запроса обращением к движку.

        @returns: ``(токены, удалось ли посчитать точно)``.
        """
        try:
            return self.ft.count_tokens(messages, system=system_text), True
        except (FreeTokenError, ValueError) as exc:
            warnings.append(f"точный подсчёт токенов недоступен ({exc}); показана оценка")
            return fallback, False

    def _refine_layer_tokens(
        self,
        layers: list[Layer],
        messages: list[dict[str, str]],
        warnings: list[str],
    ) -> None:
        """Считает стоимость каждого слоя тем же токенизатором, что и запрос.

        Оценка по длине текста нужна, чтобы подобрать окно до обращения к
        серверу, но в отладочной панели она вводит в заблуждение: сумма оценок
        не сходится с точным итогом, и слой, который на деле съедает половину
        бюджета, выглядит безобидно.

        ``count_tokens`` отказывается работать с пустым списком сообщений,
        поэтому каждый замер идёт с одним пробным сообщением, а его стоимость
        вычитается как базовая линия.

        Сумма послойных замеров может слегка расходиться с общим итогом:
        токенизатор считает границы по-разному в зависимости от окружения. Итог
        остаётся главным числом, послойные значения — ориентиром.

        @param layers: слои, у которых поле ``tokens`` заменяется точным числом.
        @param messages: сообщения окна.
        @param warnings: куда записать причину, если точный подсчёт недоступен.
        """
        probe = [{"role": "user", "content": "."}]
        try:
            baseline = self.ft.count_tokens(probe)
        except (FreeTokenError, ValueError) as exc:
            warnings.append(f"послойный подсчёт токенов недоступен ({exc}); показана оценка")
            return

        for layer in layers:
            try:
                if layer.key == "window":
                    measured = self.ft.count_tokens(messages) if messages else 0
                elif layer.text.strip():
                    measured = self.ft.count_tokens(probe, system=layer.text)
                else:
                    measured = baseline
                layer.tokens = max(0, measured - baseline)
            except (FreeTokenError, ValueError) as exc:
                warnings.append(f"слой «{layer.title}»: точный подсчёт недоступен ({exc})")
                return

    # --- суммаризация -------------------------------------------------------

    def needs_summary(self, session_id: int, dropped_count: int) -> bool:
        """Пора ли сворачивать историю.

        @param session_id: партия.
        @param dropped_count: сколько сообщений не поместилось в окно.
        @returns: ``True``, если суммаризация нужна.
        """
        settings = self.settings_provider()
        threshold = int(getattr(settings, "summarize_after_dropped", 6))
        if dropped_count < threshold:
            return False
        total = self.db.count_messages(session_id)
        return total > int(getattr(settings, "context_turns", 12))

    def summarize(self, session_id: int, keep_recent: int | None = None) -> dict[str, Any] | None:
        """Сворачивает старую часть истории в суммаризацию.

        Новая суммаризация поглощает предыдущие: хранится одна запись памяти на
        партию, иначе память растёт теми же темпами, что и история.

        @param session_id: партия.
        @param keep_recent: сколько последних сообщений оставить дословно.
        @returns: отчёт о суммаризации либо ``None``, если сворачивать нечего.
        """
        settings = self.settings_provider()
        keep = keep_recent if keep_recent is not None else int(getattr(settings, "context_turns", 12))
        previous = self.db.memories(session_id)
        covered = previous[-1].through_message_id if previous else 0
        pending = [message for message in self.db.messages(session_id) if message.id > covered]
        if len(pending) <= keep:
            return None

        older = pending[: len(pending) - keep]
        transcript = prompts.transcript(older)
        if previous:
            transcript = (
                "Раньше уже было сжато:\n"
                + "\n".join(memory.summary for memory in previous)
                + "\n\n"
                + transcript
            )

        started = time.time()
        result = self.ft.chat(
            [{"role": "user", "content": prompts.render(prompts.SUMMARIZE_PROMPT, transcript=transcript)}],
            max_tokens=int(getattr(settings, "summary_max_tokens", 400)),
            temperature=0.3,
        )
        summary = result.text.strip()
        if not summary:
            return None

        through = older[-1].id
        self.db.replace_memories(session_id, through, summary, result.completion_tokens)
        return {
            "through_message_id": through,
            "summarized_messages": len(older),
            "summary": summary,
            "tokens": result.completion_tokens,
            "elapsed_s": round(time.time() - started, 2),
        }
