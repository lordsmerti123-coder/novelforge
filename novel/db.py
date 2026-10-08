"""Хранилище миров, правил, персонажей, партий и сообщений.

Схема разделена на две части:

* **Мир** — то, что сочиняет пользователь и что переиспользуется: описание,
  правила, персонажи, стиль иллюстраций. Правится редко.
* **Партия** (сессия) — конкретное прохождение: сообщения, память, состояние
  мира, сгенерированные сцены. У одного мира может быть много партий, и они не
  влияют друг на друга.

Правило контекста: в модель уходит только состояние мира, суммаризация старых
ходов и последние сообщения. Таблицы ``scenes`` и ``speculative`` — односторонний
поток наружу, их содержимое в запрос не возвращается.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel import config
from novel.formats import DEFAULT_FORMAT

SCHEMA_VERSION = 10

#: Таблицы по отдельности: миграции пересоздают их по одной, а не весь файл.
TABLES: dict[str, str] = {
    "meta": """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
    "worlds": """
CREATE TABLE IF NOT EXISTS worlds (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    format TEXT NOT NULL,
    brief TEXT NOT NULL DEFAULT '',
    genre TEXT NOT NULL DEFAULT '',
    tone TEXT NOT NULL DEFAULT '',
    style TEXT NOT NULL DEFAULT '',
    narrator TEXT NOT NULL DEFAULT '',
    hidden_rules TEXT NOT NULL DEFAULT '',
    image_suffix TEXT NOT NULL DEFAULT '',
    image_policy TEXT NOT NULL DEFAULT '',
    image_frame TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)""",
    "world_rules": """
CREATE TABLE IF NOT EXISTS world_rules (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'rule',
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 100,
    created_at INTEGER NOT NULL
)""",
    "characters": """
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    appearance TEXT NOT NULL DEFAULT '',
    speech TEXT NOT NULL DEFAULT '',
    voice TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL
)""",
    "sessions": """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    format TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)""",
    "messages": """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',
    tokens INTEGER,
    ts INTEGER NOT NULL
)""",
    "memories": """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    through_message_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    tokens INTEGER,
    created_at INTEGER NOT NULL
)""",
    "locations": """
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    style TEXT NOT NULL DEFAULT '',
    seed INTEGER,
    reference_path TEXT,
    visits INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)""",
    "plot_notes": """CREATE TABLE IF NOT EXISTS plot_notes (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'player',
    horizon INTEGER NOT NULL DEFAULT 0,
    anchor_message_id INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)""",
    #: Вещи: что у кого есть. character_id = NULL означает игрока — отдельной
    #: записи персонажа для него нет, он сам игрок.
    "items": """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    properties TEXT NOT NULL DEFAULT '',
    quantity INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)""",
    "scenes": """
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    message_id INTEGER,
    prompt TEXT NOT NULL,
    raw_description TEXT NOT NULL DEFAULT '',
    seed INTEGER,
    path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    elapsed_s REAL,
    created_at INTEGER NOT NULL
)""",
    "speculative": """
CREATE TABLE IF NOT EXISTS speculative (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    scene_id INTEGER,
    trigger TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL,
    seed INTEGER,
    path TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
)""",
    "world_state": """
CREATE TABLE IF NOT EXISTS world_state (
    session_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (session_id, key)
)""",
}

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_scenes_session ON scenes(session_id, id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id, id);
CREATE INDEX IF NOT EXISTS idx_rules_world ON world_rules(world_id, priority);
"""

#: Колонки, без которых таблица считается старой и подлежит пересозданию,
#: и колонки, которые при этом переносятся как есть.
SHAPE_REQUIREMENTS: dict[str, tuple[set[str], list[str]]] = {
    # Дополнение к промпту кадров появилось позже остальных полей мира.
    "worlds": (
        {"image_suffix", "image_policy", "image_frame"},
        ["id", "name", "format", "brief", "genre", "tone", "style", "narrator",
         "hidden_rules", "image_suffix", "image_policy", "image_frame",
         "created_at", "updated_at"],
    ),
    "scenes": (
        {"session_id", "message_id", "raw_description", "elapsed_s"},
        ["id", "session_id", "message_id", "prompt", "raw_description", "seed", "path",
         "status", "elapsed_s", "created_at"],
    ),
    # Задумки на потом получили автора и срок, когда их стало можно поручать
    # ведущему: прежние строки достаются игроку.
    "plot_notes": (
        {"source", "horizon"},
        ["id", "session_id", "text", "status", "source", "horizon",
         "anchor_message_id", "created_at", "updated_at"],
    ),
    "speculative": (
        {"session_id"},
        ["id", "session_id", "scene_id", "trigger", "prompt", "seed", "path", "status"],
    ),
    # Состояние мира в старой схеме не было привязано к партии, а новая колонка
    # обязательна — переносить такие строки некуда, поэтому они отбрасываются.
    "world_state": ({"session_id"}, []),
}

#: Выражения для отдельных колонок при переносе. Нужны там, где значение может
#: нарушить новое ограничение: ссылка на удалённую партию не переживёт внешний
#: ключ, поэтому такие значения обнуляются.
CARRY_EXPRESSIONS: dict[str, dict[str, str]] = {
    "scenes": {
        "session_id": "CASE WHEN session_id IN (SELECT id FROM sessions) THEN session_id ELSE NULL END",
    },
    "speculative": {
        "session_id": "CASE WHEN session_id IN (SELECT id FROM sessions) THEN session_id ELSE NULL END",
    },
}

#: Колонки, добавленные после первой версии схемы. В отличие от таблиц их не
#: нужно пересоздавать: ``ALTER TABLE ADD COLUMN`` дописывает их на месте.
EXTRA_COLUMNS: dict[str, dict[str, str]] = {
    "messages": {"attachments": "TEXT NOT NULL DEFAULT ''"},
    "characters": {"voice": "TEXT NOT NULL DEFAULT ''"},
    "sessions": {"format": "TEXT NOT NULL DEFAULT ''"},
    "worlds": {
        "image_policy": "TEXT NOT NULL DEFAULT ''",
        "image_frame": "TEXT NOT NULL DEFAULT ''",
    },
    "scenes": {
        "location_id": "INTEGER",
        "used_reference": "INTEGER NOT NULL DEFAULT 0",
    },
}

#: Таблицы, которые нужно пересобрать, если сохранённая версия схемы меньше
#: указанной. Нужно там, где меняется не набор колонок, а ограничения: у
#: ``scenes`` и ``speculative`` появился внешний ключ на партии, и без пересборки
#: сцены удалённых миров оставались бы в базе.
REBUILD_WHEN_BELOW: dict[str, int] = {"scenes": 3, "speculative": 3}

SCHEMA = ";\n".join(TABLES.values()) + ";"


@dataclass
class World:
    """Мир: то, что сочинил пользователь."""

    id: int
    name: str
    format: str
    brief: str
    genre: str
    tone: str
    style: str
    narrator: str
    hidden_rules: str
    #: Дополнение, которое приписывается к каждому промпту кадра дословно.
    #: Ведущий его не видит и переписать не может: постоянная часть промпта.
    image_suffix: str
    #: Когда рисовать кадры в этом мире. Пустая строка означает «как в настройках».
    image_policy: str = ""
    #: Что показывать в кадре. Пустая строка означает «как в настройках».
    image_frame: str = ""
    created_at: int = 0
    updated_at: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Поля мира для интерфейса.

        Одно место на все ответы: словарь, собранный руками, отстаёт от таблицы
        при каждом новом поле, и добавленное поле просто не доезжает до
        страницы. Поле, которое сохраняется в базу, но не возвращается отсюда,
        стирается при каждом обновлении.

        @returns: словарь полей мира.
        """
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format,
            "image_policy": self.image_policy,
            "image_frame": self.image_frame,
            "brief": self.brief,
            "genre": self.genre,
            "tone": self.tone,
            "style": self.style,
            "narrator": self.narrator,
            "hidden_rules": self.hidden_rules,
            "image_suffix": self.image_suffix,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Rule:
    """Правило мира."""

    id: int
    world_id: int
    title: str
    body: str
    kind: str
    enabled: int
    priority: int
    created_at: int


@dataclass
class Character:
    """Карточка персонажа."""

    id: int
    world_id: int
    name: str
    role: str
    description: str
    appearance: str
    speech: str
    enabled: int
    created_at: int
    #: Образцы настоящих сообщений этого человека: по ним ведущий подстраивается
    #: под манеру, когда игрок принёс переписку из жизни. Поле стоит последним
    #: и со значением по умолчанию: карточки собираются и без него.
    voice: str = ""


@dataclass
class Session:
    """Партия: одно прохождение мира."""

    id: int
    world_id: int
    title: str
    created_at: int
    updated_at: int
    #: Формат диалога этой партии. Пустая строка означает «как у мира»: так
    #: ведут себя партии, заведённые до того, как формат переехал в партию.
    format: str = ""


@dataclass
class Message:
    """Сообщение в партии."""

    id: int
    session_id: int
    role: str
    content: str
    kind: str
    tokens: int | None
    ts: int
    #: Пути к приложенным изображениям в виде JSON-массива.
    attachments: str = ""

    @property
    def attachment_paths(self) -> list[str]:
        """Пути приложенных изображений."""
        if not self.attachments:
            return []
        try:
            value = json.loads(self.attachments)
        except json.JSONDecodeError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []


@dataclass
class Memory:
    """Суммаризация участка переписки."""

    id: int
    session_id: int
    through_message_id: int
    summary: str
    tokens: int | None
    created_at: int


@dataclass
class Item:
    """Вещь: что у кого есть и какими свойствами обладает.

    ``character_id`` равный ``None`` означает игрока: отдельной записи персонажа
    для него нет, потому что игрок — это читатель.
    """

    id: int
    world_id: int
    character_id: int | None
    name: str
    description: str
    properties: str
    quantity: int
    created_at: int
    updated_at: int

    @property
    def is_player(self) -> bool:
        """Принадлежит ли вещь игроку."""
        return self.character_id is None


@dataclass
class PlotNote:
    """Отложенная заметка для ведущего: что должно случиться позже.

    Врезка действует один ход, правило мира — всегда, а заметка живёт, пока не
    сбудется: ведущий видит её каждый ход и сам отмечает исполненную.
    """

    id: int
    session_id: int
    text: str
    status: str
    #: Кто задумал: ``player`` — игрок, ``model`` — ведущий сам.
    source: str
    #: На сколько ходов задумано; 0 — без срока.
    horizon: int
    #: Номер последнего сообщения в момент задумки: по нему считается возраст.
    anchor_message_id: int | None
    created_at: int
    updated_at: int


@dataclass
class Location:
    """Постоянное место мира.

    Хранит каноническое описание и кадр-образец: по ним одно и то же место
    выглядит одинаково при каждом возвращении.
    """

    id: int
    world_id: int
    name: str
    prompt: str
    style: str
    seed: int | None
    reference_path: str | None
    visits: int
    created_at: int
    updated_at: int


@dataclass
class Scene:
    """Запись о сгенерированной или запланированной картинке."""

    id: int
    session_id: int | None
    message_id: int | None
    prompt: str
    raw_description: str
    seed: int | None
    path: str | None
    status: str
    elapsed_s: float | None
    created_at: int
    #: Место, к которому относится кадр; ``None`` для старых записей.
    location_id: int | None = None
    #: Рисовался ли кадр по образцу места.
    used_reference: int = 0


def _row_to(model: type, row: sqlite3.Row) -> Any:
    """Собирает dataclass из строки запроса."""
    return model(**{key: row[key] for key in row.keys()})


class NovelDB:
    """Обёртка над SQLite с методами уровня предметной области.

    Соединение заводится **на каждый поток**. Одно общее соединение с
    ``check_same_thread=False`` выглядит рабочим, но под параллельными запросами
    разваливается: интерфейс опрашивает ``/api/status`` из разных потоков, и
    запрос может вернуть пустую строку вместо счётчика — ``TypeError:
    'NoneType' object is not subscriptable`` в ``stats()``, из-за которого
    перестаёт отвечать весь ``/api/status``.

    У каждого потока своё соединение, а запись разводится таймаутом ожидания:
    в режиме WAL читатели писателю не мешают, а два писателя иначе получили бы
    «database is locked».
    """

    def __init__(self, path: Path = config.DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        #: Все открытые соединения. Нужны, чтобы ``close`` закрывал и чужие
        #: потоки: иначе файл остаётся заблокированным на Windows, и временный
        #: каталог с базой не удаляется.
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        conn = self._connect()
        stored_version = self._stored_schema_version()
        conn.executescript(SCHEMA)
        forced = {
            table for table, version in REBUILD_WHEN_BELOW.items() if stored_version < version
        }
        # Сначала лечим ссылки на удалённые ``*_legacy``: такая таблица читается,
        # но не пишется, и без починки миграция ниже сломалась бы на ней же.
        self._repair_legacy_links()
        self._migrate_table_shapes(force=forced)
        self._ensure_columns()
        # Индексы создаются после миграции: на старой форме таблицы колонки,
        # по которой строится индекс, ещё нет.
        conn.executescript(INDEXES)
        self._migrate_legacy_turns()
        self._purge_orphans()
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Открывает соединение для текущего потока.

        @returns: соединение с настроенными режимами журнала и ожидания.
        """
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Без этого второй писатель сразу получает «database is locked» вместо
        # того, чтобы подождать первого.
        conn.execute("PRAGMA busy_timeout=10000")
        with self._connections_lock:
            self._connections.append(conn)
        self._local.conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Соединение текущего потока; создаётся при первом обращении."""
        conn = getattr(self._local, "conn", None)
        return conn if conn is not None else self._connect()

    def close(self) -> None:
        """Закрывает соединения всех потоков."""
        with self._connections_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    # Уже закрытое соединение — не повод падать при уборке.
                    continue
            self._connections.clear()
        self._local.conn = None

    # --- миграции -----------------------------------------------------------

    def _has_table(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _columns(self, table: str) -> set[str]:
        """Имена колонок существующей таблицы."""
        return {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}

    def _stored_schema_version(self) -> int:
        """Версия схемы, записанная в базе; 0 для базы без таблицы ``meta``."""
        if not self._has_table("meta"):
            return 0
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def _purge_orphans(self) -> int:
        """Убирает сцены, оставшиеся от удалённых партий.

        До появления внешнего ключа удаление мира не трогало сцены, а SQLite
        переиспользует освободившиеся номера партий. В результате чужие кадры
        могли появиться в новой партии под тем же номером. Признак подделки —
        сообщение, привязанное к другой партии.

        @returns: сколько записей удалено.
        """
        removed = 0
        for table in ("scenes", "speculative"):
            if not self._has_table(table):
                continue
            cursor = self.conn.execute(
                f"DELETE FROM {table} WHERE session_id IS NOT NULL AND session_id NOT IN"
                " (SELECT id FROM sessions)"
            )
            removed += cursor.rowcount
        # Предгенерация без партии недостижима: выборка всегда идёт по партии,
        # поэтому такие строки остаются мусором от старой схемы.
        cursor = self.conn.execute("DELETE FROM speculative WHERE session_id IS NULL")
        removed += cursor.rowcount
        cursor = self.conn.execute(
            "DELETE FROM scenes WHERE message_id IS NOT NULL AND session_id IS NOT NULL"
            " AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.id = scenes.message_id"
            " AND m.session_id = scenes.session_id)"
        )
        removed += cursor.rowcount
        if removed:
            self.conn.commit()
        return removed

    def check_integrity(self) -> list[dict[str, Any]]:
        """Ищет записи, потерявшие владельца.

        @returns: список найденных расхождений с пояснением.
        """
        issues: list[dict[str, Any]] = []
        checks = [
            ("сцены без партии", "SELECT COUNT(*) AS n FROM scenes WHERE session_id IS NULL"),
            ("сцены чужой партии", "SELECT COUNT(*) AS n FROM scenes WHERE message_id IS NOT NULL"
             " AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.id = scenes.message_id"
             " AND m.session_id = scenes.session_id)"),
            ("предгенерации без партии",
             "SELECT COUNT(*) AS n FROM speculative WHERE session_id IS NULL"
             " OR session_id NOT IN (SELECT id FROM sessions)"),
            ("сообщения без партии", "SELECT COUNT(*) AS n FROM messages WHERE session_id NOT IN"
             " (SELECT id FROM sessions)"),
            ("партии без мира", "SELECT COUNT(*) AS n FROM sessions WHERE world_id NOT IN"
             " (SELECT id FROM worlds)"),
            ("правила без мира", "SELECT COUNT(*) AS n FROM world_rules WHERE world_id NOT IN"
             " (SELECT id FROM worlds)"),
        ]
        for title, query in checks:
            row = self.conn.execute(query).fetchone()
            if row and row["n"]:
                issues.append({"check": title, "count": int(row["n"])})
        return issues

    def _ensure_columns(self) -> None:
        """Дописывает колонки, появившиеся после первой версии схемы.

        В отличие от смены формы таблицы, добавление колонки не требует
        пересоздания: ``ALTER TABLE ADD COLUMN`` сохраняет все строки на месте.
        """
        for table, columns in EXTRA_COLUMNS.items():
            if not self._has_table(table):
                continue
            existing = self._columns(table)
            for name, ddl in columns.items():
                if name in existing:
                    continue
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        self.conn.commit()

    def _migrate_table_shapes(self, force: set[str] | None = None) -> None:
        """Пересоздаёт таблицы, оставшиеся от прежних версий схемы.

        ``CREATE TABLE IF NOT EXISTS`` не меняет уже существующую таблицу,
        поэтому старую форму приходится перестраивать: переименовать, создать
        новую и перенести совпадающие колонки.

        Внешние ключи на время выключаются: при переименовании таблицы SQLite
        переписывает ссылки в других таблицах на новое имя, и последующий
        ``DROP`` упирается в ограничение. Прагма действует только вне
        транзакции, поэтому коммит делается заранее.

        @param force: таблицы, которые нужно пересобрать независимо от набора
            колонок — например, когда изменились ограничения.
        """
        # Хвост от прерванной миграции: таблица уже пересоздана, а старая копия
        # осталась. Данные из неё уже перенесены, поэтому копия удаляется.
        for table in SHAPE_REQUIREMENTS:
            legacy = f"{table}_legacy"
            if self._has_table(legacy):
                self.conn.execute(f"DROP TABLE {legacy}")
        self.conn.commit()

        forced = force or set()
        pending = [
            (table, required, carried)
            for table, (required, carried) in SHAPE_REQUIREMENTS.items()
            if self._has_table(table)
            and (table in forced or not required.issubset(self._columns(table)))
        ]
        if not pending:
            return

        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=OFF")
        # legacy_alter_table не даёт SQLite переписать ссылки на переименованную
        # таблицу в других таблицах. Без него дочерние таблицы начинают
        # ссылаться на «worlds_legacy», которую мы тут же удаляем, и любая запись
        # падает с «no such table: main.worlds_legacy» — чтение при этом работает,
        # поэтому поломка всплывает не сразу.
        self.conn.execute("PRAGMA legacy_alter_table=ON")
        try:
            for table, _required, carried in pending:
                self._rebuild_table(table, carried)
            self.conn.commit()
        finally:
            self.conn.execute("PRAGMA legacy_alter_table=OFF")
            self.conn.execute("PRAGMA foreign_keys=ON")

    def _rebuild_table(self, table: str, carried: list[str]) -> None:
        """Пересоздаёт таблицу по текущему описанию, перенося совпадающие колонки.

        Вызывается только при выключенных внешних ключах и включённом
        ``legacy_alter_table``: иначе SQLite испортит ссылки в дочерних таблицах.

        @param table: имя таблицы.
        @param carried: колонки-кандидаты на перенос.
        """
        legacy = f"{table}_legacy"
        available = self._columns(table)
        columns = [column for column in carried if column in available]
        overrides = CARRY_EXPRESSIONS.get(table, {})
        self.conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
        self.conn.execute(TABLES[table])
        if columns:
            names = ", ".join(columns)
            expressions = ", ".join(overrides.get(column, column) for column in columns)
            self.conn.execute(
                f"INSERT INTO {table} ({names}) SELECT {expressions} FROM {legacy}"
            )
        self.conn.execute(f"DROP TABLE {legacy}")

    def _repair_legacy_links(self) -> list[str]:
        """Пересобирает таблицы, оставшиеся со ссылкой на удалённую ``*_legacy``.

        SQLite при переименовании таблицы переписывает ссылки на неё в других
        таблицах. Если старая копия потом удаляется, дочерние таблицы начинают
        ссылаться на несуществующую таблицу: чтение работает, а любая запись
        падает. Такие таблицы пересобираются из текущего описания схемы.

        @returns: имена пересобранных таблиц.
        """
        broken = [
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%_legacy%'"
            ).fetchall()
            if row[0] in TABLES
        ]
        if not broken:
            return []
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute("PRAGMA legacy_alter_table=ON")
        try:
            for table in broken:
                self._rebuild_table(table, self._columns(table))
            self.conn.commit()
        finally:
            self.conn.execute("PRAGMA legacy_alter_table=OFF")
            self.conn.execute("PRAGMA foreign_keys=ON")
        return broken

    def _migrate_legacy_turns(self) -> None:
        """Переносит ходы из старой таблицы ``turns`` в первый мир и партию.

        Старая схема знала только один мир. Чтобы наработки не пропали, при
        первом запуске новой версии они переезжают в мир по умолчанию, а сама
        таблица переименовывается — повторный перенос исключён.
        """
        if not self._has_table("turns"):
            return
        rows = self.conn.execute("SELECT role, content, ts FROM turns ORDER BY id").fetchall()
        if rows:
            world_id = self.create_world(
                name="Перенесённый мир",
                format=DEFAULT_FORMAT,
                brief="Ходы из версии без миров.",
            )
            session_id = self.create_session(world_id, title="Первая партия")
            for row in rows:
                self.add_message(
                    session_id, row["role"], row["content"], ts=int(row["ts"])
                )
        self.conn.execute("ALTER TABLE turns RENAME TO turns_migrated_v2")
        self.conn.commit()

    # --- миры ---------------------------------------------------------------

    def create_world(
        self,
        *,
        name: str,
        format: str = DEFAULT_FORMAT,
        brief: str = "",
        genre: str = "",
        tone: str = "",
        style: str = "",
        narrator: str = "",
        hidden_rules: str = "",
        image_suffix: str = "",
        image_policy: str = "",
        image_frame: str = "",
    ) -> int:
        """Создаёт мир и возвращает его идентификатор.

        @param image_policy: когда рисовать кадры; пустая строка — как в настройках.
        @param image_frame: что показывать в кадре; пустая строка — как в настройках.
        @returns: идентификатор мира.
        """
        now = int(time.time())
        cursor = self.conn.execute(
            "INSERT INTO worlds (name, format, brief, genre, tone, style, narrator, hidden_rules,"
            " image_suffix, image_policy, image_frame, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, format, brief, genre, tone, style, narrator, hidden_rules, image_suffix,
             image_policy, image_frame, now, now),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def worlds(self) -> list[World]:
        """Все миры, свежие сверху."""
        rows = self.conn.execute("SELECT * FROM worlds ORDER BY updated_at DESC").fetchall()
        return [_row_to(World, row) for row in rows]

    def world(self, world_id: int) -> World | None:
        """Мир по идентификатору."""
        row = self.conn.execute("SELECT * FROM worlds WHERE id = ?", (world_id,)).fetchone()
        return None if row is None else _row_to(World, row)

    def update_world(self, world_id: int, changes: dict[str, Any]) -> list[str]:
        """Правит поля мира.

        @param changes: пары «поле — значение».
        @returns: имена отклонённых полей.
        """
        allowed = {"name", "format", "brief", "genre", "tone", "style", "narrator",
                   "hidden_rules", "image_suffix", "image_policy", "image_frame"}
        rejected = [key for key in changes if key not in allowed]
        fields = {key: value for key, value in changes.items() if key in allowed}
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.conn.execute(
                f"UPDATE worlds SET {assignments}, updated_at = ? WHERE id = ?",
                (*fields.values(), int(time.time()), world_id),
            )
            self.conn.commit()
        return rejected

    def delete_world(self, world_id: int) -> None:
        """Удаляет мир со всем содержимым."""
        self.conn.execute("DELETE FROM worlds WHERE id = ?", (world_id,))
        self.conn.commit()

    # --- правила ------------------------------------------------------------

    def add_rule(
        self,
        world_id: int,
        body: str,
        *,
        title: str = "",
        kind: str = "rule",
        priority: int = 100,
        enabled: bool = True,
    ) -> int:
        """Добавляет правило мира."""
        cursor = self.conn.execute(
            "INSERT INTO world_rules (world_id, title, body, kind, enabled, priority, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (world_id, title, body, kind, int(enabled), priority, int(time.time())),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def rules(self, world_id: int, enabled_only: bool = False) -> list[Rule]:
        """Правила мира по возрастанию приоритета."""
        query = "SELECT * FROM world_rules WHERE world_id = ?"
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY priority, id"
        rows = self.conn.execute(query, (world_id,)).fetchall()
        return [_row_to(Rule, row) for row in rows]

    def update_rule(self, rule_id: int, changes: dict[str, Any]) -> list[str]:
        """Правит правило."""
        allowed = {"title", "body", "kind", "enabled", "priority"}
        rejected = [key for key in changes if key not in allowed]
        fields = {key: value for key, value in changes.items() if key in allowed}
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.conn.execute(
                f"UPDATE world_rules SET {assignments} WHERE id = ?", (*fields.values(), rule_id)
            )
            self.conn.commit()
        return rejected

    def delete_rule(self, rule_id: int) -> None:
        """Удаляет правило."""
        self.conn.execute("DELETE FROM world_rules WHERE id = ?", (rule_id,))
        self.conn.commit()

    # --- персонажи ----------------------------------------------------------

    def add_character(
        self,
        world_id: int,
        name: str,
        *,
        role: str = "",
        description: str = "",
        appearance: str = "",
        speech: str = "",
        voice: str = "",
        enabled: bool = True,
    ) -> int:
        """Добавляет персонажа.

        @param voice: образцы настоящих сообщений человека, по строке на образец.
        @returns: идентификатор персонажа.
        """
        # Имя обрезается по краям: «Хозяйка » и «Хозяйка» — для словаря внешности
        # два разных человека, и ведущий начинает их путать.
        cursor = self.conn.execute(
            "INSERT INTO characters (world_id, name, role, description, appearance, speech,"
            " voice, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (world_id, name.strip(), role, description, appearance, speech,
             voice, int(enabled), int(time.time())),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def characters(self, world_id: int, enabled_only: bool = False) -> list[Character]:
        """Персонажи мира."""
        query = "SELECT * FROM characters WHERE world_id = ?"
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY id"
        rows = self.conn.execute(query, (world_id,)).fetchall()
        return [_row_to(Character, row) for row in rows]

    def character(self, character_id: int) -> Character | None:
        """Персонаж по идентификатору."""
        row = self.conn.execute(
            "SELECT * FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
        return None if row is None else _row_to(Character, row)

    def update_character(self, character_id: int, changes: dict[str, Any]) -> list[str]:
        """Правит карточку персонажа."""
        allowed = {"name", "role", "description", "appearance", "speech", "voice", "enabled"}
        rejected = [key for key in changes if key not in allowed]
        fields = {key: value for key, value in changes.items() if key in allowed}
        if "name" in fields:
            # Имя обрезается по краям: «Хозяйка » и «Хозяйка» для словаря
            # внешности два разных человека, и ведущий начинает их путать.
            fields["name"] = str(fields["name"]).strip()
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.conn.execute(
                f"UPDATE characters SET {assignments} WHERE id = ?",
                (*fields.values(), character_id),
            )
            self.conn.commit()
        return rejected

    def delete_character(self, character_id: int) -> None:
        """Удаляет персонажа."""
        self.conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))
        self.conn.commit()

    # --- партии -------------------------------------------------------------

    def create_session(self, world_id: int, title: str = "", format: str = "") -> int:
        """Создаёт партию в мире.

        @param world_id: мир.
        @param title: название партии.
        @param format: формат диалога; пустая строка означает «как у мира».
        @returns: идентификатор партии.
        """
        now = int(time.time())
        cursor = self.conn.execute(
            "INSERT INTO sessions (world_id, title, format, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (world_id, title, format, now, now),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def sessions(self, world_id: int | None = None) -> list[Session]:
        """Партии, свежие сверху."""
        if world_id is None:
            rows = self.conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE world_id = ? ORDER BY updated_at DESC", (world_id,)
            ).fetchall()
        return [_row_to(Session, row) for row in rows]

    def session(self, session_id: int) -> Session | None:
        """Партия по идентификатору."""
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else _row_to(Session, row)

    def touch_session(self, session_id: int) -> None:
        """Обновляет время последней активности партии."""
        self.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (int(time.time()), session_id)
        )
        self.conn.commit()

    def delete_session(self, session_id: int) -> None:
        """Удаляет партию со всем содержимым."""
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def rename_session(self, session_id: int, title: str) -> None:
        """Переименовывает партию."""
        """Переименовывает партию."""
        self.conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title.strip() or "Без названия", int(time.time()), session_id),
        )
        self.conn.commit()

    def set_session_format(self, session_id: int, fmt: str) -> None:
        """Задаёт формат диалога у партии.

        Пустая строка означает «как у мира»: тогда формат берётся у мира, как
        было до того, как он переехал в партию.

        @param session_id: партия.
        @param fmt: ключ формата или пустая строка.
        """
        self.conn.execute(
            "UPDATE sessions SET format = ?, updated_at = ? WHERE id = ?",
            (fmt.strip(), int(time.time()), session_id),
        )
        self.conn.commit()

    def session_stats(self, session_id: int) -> dict[str, Any]:
        """Сводка по партии: объём, время, расход.

        @param session_id: партия.
        @returns: счётчики сообщений, сцен, памяти и суммарные токены.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts,"
            " COALESCE(SUM(tokens), 0) AS tokens FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        scenes = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(elapsed_s), 0) AS seconds FROM scenes"
            " WHERE session_id = ? AND status = 'done'",
            (session_id,),
        ).fetchone()
        pending = self.conn.execute(
            "SELECT COUNT(*) AS n FROM scenes WHERE session_id = ? AND status = 'pending'",
            (session_id,),
        ).fetchone()
        memories = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS tokens FROM memories WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return {
            "messages": int(row["n"]),
            "first_ts": row["first_ts"],
            "last_ts": row["last_ts"],
            "completion_tokens": int(row["tokens"]),
            "scenes_done": int(scenes["n"]),
            "scenes_pending": int(pending["n"]),
            "scene_seconds": round(float(scenes["seconds"]), 1),
            "memories": int(memories["n"]),
            "memory_tokens": int(memories["tokens"]),
            "state_keys": len(self.world_state(session_id)),
        }

    # --- перенос данных -----------------------------------------------------

    def export_session(self, session_id: int) -> dict[str, Any]:
        """Выгружает партию в переносимый словарь.

        Мир в выгрузку не входит: он переиспользуется и живёт отдельно. Сцены
        выгружаются без файлов — только описания и пути, потому что картинки
        лежат вне базы.

        @param session_id: партия.
        @returns: словарь, пригодный для записи в JSON.
        """
        session = self.session(session_id)
        if session is None:
            raise ValueError(f"партия {session_id} не найдена")
        return {
            "kind": "novelforge.session",
            "version": 1,
            "title": session.title,
            "messages": [
                {"role": m.role, "content": m.content, "kind": m.kind, "tokens": m.tokens,
                 "ts": m.ts, "attachments": m.attachment_paths}
                for m in self.messages(session_id)
            ],
            "memories": [
                {"through_message_id": m.through_message_id, "summary": m.summary, "tokens": m.tokens}
                for m in self.memories(session_id)
            ],
            "scenes": [
                {
                    "prompt": s.prompt,
                    "raw_description": s.raw_description,
                    "seed": s.seed,
                    "path": s.path,
                    "status": s.status,
                    "elapsed_s": s.elapsed_s,
                }
                for s in self.scenes(session_id)
            ],
            "state": self.world_state(session_id),
        }

    def import_session(self, world_id: int, payload: dict[str, Any]) -> int:
        """Загружает партию из выгрузки в указанный мир.

        @param world_id: мир, куда загружается партия.
        @param payload: словарь из :meth:`export_session`.
        @returns: идентификатор новой партии.
        """
        if payload.get("kind") != "novelforge.session":
            raise ValueError("это не выгрузка партии NovelForge")
        session_id = self.create_session(world_id, str(payload.get("title") or "Импорт"))
        for message in payload.get("messages") or []:
            self.add_message(
                session_id,
                str(message.get("role", "user")),
                str(message.get("content", "")),
                kind=str(message.get("kind", "text")),
                tokens=message.get("tokens"),
                ts=message.get("ts"),
                attachments=[str(item) for item in (message.get("attachments") or [])],
            )
        for memory in payload.get("memories") or []:
            self.add_memory(
                session_id,
                int(memory.get("through_message_id", 0)),
                str(memory.get("summary", "")),
                memory.get("tokens"),
            )
        for scene in payload.get("scenes") or []:
            scene_id = self.add_scene(
                session_id,
                str(scene.get("prompt", "")),
                raw_description=str(scene.get("raw_description", "")),
                seed=scene.get("seed"),
                status=str(scene.get("status", "pending")),
            )
            if scene.get("path"):
                self.finish_scene(scene_id, str(scene["path"]), str(scene.get("status", "done")),
                                  scene.get("elapsed_s"))
        for key, value in (payload.get("state") or {}).items():
            self.set_state(session_id, key, value)
        return session_id

    def export_world(self, world_id: int) -> dict[str, Any]:
        """Выгружает мир целиком: описание, правила, персонажей и все партии.

        @param world_id: мир.
        @returns: словарь, пригодный для записи в JSON.
        """
        world = self.world(world_id)
        if world is None:
            raise ValueError(f"мир {world_id} не найден")
        return {
            "kind": "novelforge.world",
            "version": 1,
            "world": {
                "name": world.name,
                "format": world.format,
                "brief": world.brief,
                "genre": world.genre,
                "tone": world.tone,
                "style": world.style,
                "narrator": world.narrator,
                "hidden_rules": world.hidden_rules,
                "image_suffix": world.image_suffix,
            },
            "rules": [
                {"title": r.title, "body": r.body, "kind": r.kind, "priority": r.priority,
                 "enabled": bool(r.enabled)}
                for r in self.rules(world_id)
            ],
            "characters": [
                {"name": c.name, "role": c.role, "description": c.description,
                 "appearance": c.appearance, "speech": c.speech, "enabled": bool(c.enabled)}
                for c in self.characters(world_id)
            ],
            "sessions": [self.export_session(s.id) for s in self.sessions(world_id)],
            # Владелец-персонаж записывается именем: номера при переносе в другой
            # мир другие, а имя остаётся тем же.
            "items": [
                {
                    "name": item.name,
                    "description": item.description,
                    "properties": item.properties,
                    "quantity": item.quantity,
                    "owner": next(
                        (c.name for c in self.characters(world_id) if c.id == item.character_id),
                        None,
                    ),
                }
                for item in self.items(world_id)
            ],
        }

    def import_world(self, payload: dict[str, Any]) -> tuple[int, list[int]]:
        """Загружает мир из выгрузки.

        @param payload: словарь из :meth:`export_world`.
        @returns: ``(идентификатор мира, идентификаторы партий)``.
        """
        if payload.get("kind") != "novelforge.world":
            raise ValueError("это не выгрузка мира NovelForge")
        data = payload.get("world") or {}
        world_id = self.create_world(
            name=str(data.get("name") or "Импортированный мир"),
            format=str(data.get("format") or DEFAULT_FORMAT),
            brief=str(data.get("brief") or ""),
            genre=str(data.get("genre") or ""),
            tone=str(data.get("tone") or ""),
            style=str(data.get("style") or ""),
            narrator=str(data.get("narrator") or ""),
            hidden_rules=str(data.get("hidden_rules") or ""),
            image_suffix=str(data.get("image_suffix") or ""),
        )
        for rule in payload.get("rules") or []:
            self.add_rule(
                world_id,
                str(rule.get("body", "")),
                title=str(rule.get("title", "")),
                kind=str(rule.get("kind", "rule")),
                priority=int(rule.get("priority", 100)),
                enabled=bool(rule.get("enabled", True)),
            )
        for character in payload.get("characters") or []:
            self.add_character(
                world_id,
                str(character.get("name", "")),
                role=str(character.get("role", "")),
                description=str(character.get("description", "")),
                appearance=str(character.get("appearance", "")),
                speech=str(character.get("speech", "")),
                enabled=bool(character.get("enabled", True)),
            )
        sessions = [self.import_session(world_id, item) for item in payload.get("sessions") or []]
        # Вещи кладутся после персонажей: владелец ищется по имени.
        owners = {c.name: c.id for c in self.characters(world_id)}
        for item in payload.get("items") or []:
            self.add_item(
                world_id,
                str(item.get("name", "")),
                character_id=owners.get(str(item.get("owner") or "")),
                description=str(item.get("description", "")),
                properties=str(item.get("properties", "")),
                quantity=int(item.get("quantity", 1) or 1),
            )
        if not sessions:
            sessions = [self.create_session(world_id, "Первая партия")]
        return world_id, sessions

    def duplicate_world(self, world_id: int, *, with_sessions: bool = False) -> tuple[int, list[int]]:
        """Копирует мир целиком.

        Нужно для отладки: копию можно ломать, менять настройки и проверять
        гипотезы, не трогая рабочий мир.

        Файлы изображений не копируются — оба мира ссылаются на одни и те же
        файлы. Это безопасно: уборка мусора считает файл нужным, пока на него
        ссылается хоть одна запись.

        @param world_id: мир-образец.
        @param with_sessions: копировать ли партии с историей, кадрами и памятью.
            Без них копия получает одну пустую партию — чтобы играть заново.
        @returns: ``(идентификатор копии, идентификаторы партий)``.
        @raises ValueError: если мир не найден.
        """
        source = self.world(world_id)
        if source is None:
            raise ValueError(f"мир {world_id} не найден")

        copy_id = self.create_world(
            name=f"{source.name} (копия)",
            format=source.format,
            brief=source.brief,
            genre=source.genre,
            tone=source.tone,
            style=source.style,
            narrator=source.narrator,
            hidden_rules=source.hidden_rules,
            image_suffix=source.image_suffix,
        )

        for rule in self.rules(world_id):
            self.add_rule(
                copy_id, rule.body, title=rule.title, kind=rule.kind,
                priority=rule.priority, enabled=bool(rule.enabled),
            )

        # Персонажей копируем с сохранением номеров: на них ссылаются вещи.
        owners: dict[int, int] = {}
        for character in self.characters(world_id):
            owners[character.id] = self.add_character(
                copy_id, character.name, role=character.role,
                description=character.description, appearance=character.appearance,
                speech=character.speech, enabled=bool(character.enabled),
            )
        for item in self.items(world_id):
            self.add_item(
                copy_id, item.name,
                character_id=owners.get(item.character_id) if item.character_id else None,
                description=item.description, properties=item.properties,
                quantity=item.quantity,
            )

        places: dict[int, int] = {}
        for location in self.locations(world_id, limit=500):
            new_place = self.add_location(
                copy_id, location.name, prompt=location.prompt,
                style=location.style, seed=location.seed,
            )
            if location.reference_path:
                self.set_location_reference(new_place, location.reference_path)
            places[location.id] = new_place

        if not with_sessions:
            return copy_id, [self.create_session(copy_id, "Первая партия")]

        copies: list[int] = []
        for session in self.sessions(world_id):
            new_session = self.create_session(copy_id, session.title)
            # Номера сообщений в копии другие: сцены и память ссылаются на старые,
            # поэтому переносим их через таблицу соответствия.
            message_ids: dict[int, int] = {}
            for message in self.messages(session.id):
                message_ids[message.id] = self.add_message(
                    new_session, message.role, message.content, kind=message.kind,
                    tokens=message.tokens, ts=message.ts,
                    attachments=message.attachment_paths,
                )
            for scene in self.scenes(session.id):
                new_scene = self.add_scene(
                    new_session, scene.prompt,
                    message_id=message_ids.get(scene.message_id) if scene.message_id else None,
                    raw_description=scene.raw_description, seed=scene.seed,
                    status=scene.status,
                    location_id=places.get(scene.location_id) if scene.location_id else None,
                )
                if scene.path:
                    self.finish_scene(
                        new_scene, scene.path, scene.status or "done",
                        scene.elapsed_s, used_reference=bool(scene.used_reference),
                    )
            for memory in self.memories(session.id):
                self.add_memory(
                    new_session,
                    message_ids.get(memory.through_message_id, memory.through_message_id),
                    memory.summary, memory.tokens,
                )
            copies.append(new_session)
        if not copies:
            copies = [self.create_session(copy_id, "Первая партия")]
        return copy_id, copies

    def clear_session(self, session_id: int) -> None:
        """Очищает партию, оставляя мир: сообщения, память, сцены, состояние.

        Это кнопка «начать заново»: мир и правила остаются, история пропадает.
        """
        for table in ("messages", "memories", "world_state"):
            self.conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM scenes WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM speculative WHERE session_id = ?", (session_id,))
        self.conn.commit()

    # --- сообщения ----------------------------------------------------------

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        *,
        kind: str = "text",
        tokens: int | None = None,
        ts: int | None = None,
        attachments: list[str] | None = None,
    ) -> int:
        """Добавляет сообщение и возвращает его идентификатор.

        @param attachments: пути к изображениям, приложенным к сообщению.
        """
        cursor = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, kind, tokens, ts, attachments)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                kind,
                tokens,
                ts if ts is not None else int(time.time()),
                json.dumps(attachments or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()
        self.touch_session(session_id)
        return int(cursor.lastrowid)

    def messages(self, session_id: int, limit: int | None = None) -> list[Message]:
        """Сообщения партии в хронологическом порядке."""
        if limit is None:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)"
                " ORDER BY id",
                (session_id, limit),
            ).fetchall()
        return [_row_to(Message, row) for row in rows]

    def update_message(self, message_id: int, content: str) -> None:
        """Правит текст сообщения — ручная починка ответа модели."""
        self.conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
        self.conn.commit()

    def delete_message(self, message_id: int) -> None:
        """Удаляет одно сообщение."""
        self.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self.conn.commit()

    def rewind_to(self, session_id: int, message_id: int) -> dict[str, int]:
        """Откатывает партию к сообщению, удаляя всё после него.

        Вместе с сообщениями убирается и то, что на них ссылалось: сцены этих
        ходов и сводки памяти, которые их описывали. Иначе после отката модель
        получала бы пересказ событий, которых в истории уже нет, а справа
        висели бы кадры удалённых сцен.

        Файлы изображений при этом остаются на диске: база их не хранит, и
        удалять чужие файлы молча нельзя. Их находит уборка мусора.

        @param session_id: партия.
        @param message_id: сообщение, к которому откатываемся; оно остаётся.
        @returns: сколько чего удалено: ``messages``, ``scenes``, ``memories``.
        """
        messages = self.conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id > ?", (session_id, message_id)
        ).rowcount
        scenes = self.conn.execute(
            "DELETE FROM scenes WHERE session_id = ? AND message_id > ?", (session_id, message_id)
        ).rowcount
        memories = self.conn.execute(
            "DELETE FROM memories WHERE session_id = ? AND through_message_id > ?",
            (session_id, message_id),
        ).rowcount
        # Показанные лица относились к удалённым кадрам: в новой ветке читатель
        # их ещё не видел, и политика «минимум картинок» должна считать их новыми.
        self.conn.execute(
            "DELETE FROM world_state WHERE session_id = ? AND key = 'shown_characters'",
            (session_id,),
        )
        self.conn.commit()
        return {"messages": messages, "scenes": scenes, "memories": memories}

    def count_messages(self, session_id: int) -> int:
        """Сколько сообщений в партии."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])

    # --- память -------------------------------------------------------------

    def add_memory(self, session_id: int, through_message_id: int, summary: str, tokens: int | None = None) -> int:
        """Сохраняет суммаризацию участка переписки."""
        cursor = self.conn.execute(
            "INSERT INTO memories (session_id, through_message_id, summary, tokens, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, through_message_id, summary, tokens, int(time.time())),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def memories(self, session_id: int) -> list[Memory]:
        """Суммаризации партии по возрастанию охвата."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE session_id = ? ORDER BY through_message_id, id",
            (session_id,),
        ).fetchall()
        return [_row_to(Memory, row) for row in rows]

    def latest_memory(self, session_id: int) -> Memory | None:
        """Последняя суммаризация: с её конца и начинается дословное окно."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE session_id = ? ORDER BY through_message_id DESC, id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return None if row is None else _row_to(Memory, row)

    def replace_memories(self, session_id: int, through_message_id: int, summary: str, tokens: int | None) -> None:
        """Сворачивает все прежние суммаризации в одну новую.

        Иначе память растёт теми же темпами, что и история: каждая новая
        суммаризация поглощает предыдущие.
        """
        self.conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        self.add_memory(session_id, through_message_id, summary, tokens)

    # --- вещи ---------------------------------------------------------------

    def add_item(
        self,
        world_id: int,
        name: str,
        *,
        character_id: int | None = None,
        description: str = "",
        properties: str = "",
        quantity: int = 1,
    ) -> int:
        """Кладёт вещь в мир.

        @param world_id: мир.
        @param name: название, как его называет ведущий.
        @param character_id: владелец-персонаж; ``None`` — вещь у игрока.
        @param description: что это такое.
        @param properties: чем вещь полезна или опасна.
        @param quantity: сколько штук.
        @returns: идентификатор вещи.
        """
        now = int(time.time())
        cursor = self.conn.execute(
            "INSERT INTO items (world_id, character_id, name, description, properties,"
            " quantity, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (world_id, character_id, name.strip(), description.strip(),
             properties.strip(), max(1, int(quantity)), now, now),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def items(self, world_id: int, character_id: int | None = "any") -> list[Item]:
        """Вещи мира.

        @param world_id: мир.
        @param character_id: конкретный владелец; ``None`` — только вещи игрока;
            ``"any"`` — все подряд.
        @returns: список вещей в порядке добавления.
        """
        if character_id == "any":
            rows = self.conn.execute(
                "SELECT * FROM items WHERE world_id = ? ORDER BY id", (world_id,)
            ).fetchall()
        elif character_id is None:
            rows = self.conn.execute(
                "SELECT * FROM items WHERE world_id = ? AND character_id IS NULL ORDER BY id",
                (world_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM items WHERE world_id = ? AND character_id = ? ORDER BY id",
                (world_id, character_id),
            ).fetchall()
        return [_row_to(Item, row) for row in rows]

    def item(self, item_id: int) -> Item | None:
        """Вещь по идентификатору."""
        row = self.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return None if row is None else _row_to(Item, row)

    def update_item(self, item_id: int, changes: dict[str, Any]) -> list[str]:
        """Правит вещь.

        @param changes: пары «поле — значение»; ``character_id`` можно сменить,
            чтобы передать вещь другому.
        @returns: имена отклонённых полей.
        """
        allowed = {"name", "description", "properties", "quantity", "character_id"}
        rejected = [key for key in changes if key not in allowed]
        fields = {key: value for key, value in changes.items() if key in allowed}
        if "quantity" in fields:
            fields["quantity"] = max(1, int(fields["quantity"]))
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.conn.execute(
                f"UPDATE items SET {assignments}, updated_at = ? WHERE id = ?",
                (*fields.values(), int(time.time()), item_id),
            )
            self.conn.commit()
        return rejected

    def delete_item(self, item_id: int) -> None:
        """Убирает вещь из мира."""
        self.conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self.conn.commit()

    # --- заметки на потом ---------------------------------------------------

    def add_note(
        self, session_id: int, text: str, *, source: str = "player",
        horizon: int = 0, anchor_message_id: int | None = None,
    ) -> int:
        """Заводит заметку на потом.

        @param session_id: партия, к которой относится заметка.
        @param text: что должно случиться позже.
        @param source: ``player`` или ``model`` — от этого зависит и срочность,
            и то, какой потолок считается.
        @param horizon: на сколько ходов задумано; 0 — без срока.
        @param anchor_message_id: последнее сообщение в момент задумки.
        @returns: идентификатор заметки.
        """
        now = int(time.time())
        cursor = self.conn.execute(
            "INSERT INTO plot_notes (session_id, text, status, source, horizon,"
            " anchor_message_id, created_at, updated_at)"
            " VALUES (?, ?, 'active', ?, ?, ?, ?, ?)",
            (session_id, text.strip(), source, int(horizon), anchor_message_id, now, now),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def notes(
        self, session_id: int, status: str = "active", limit: int = 20,
        source: str | None = None,
    ) -> list[PlotNote]:
        """Заметки партии: по умолчанию только несбывшиеся, свежие сверху.

        @param session_id: партия.
        @param status: статус заметок.
        @param limit: сколько вернуть.
        @param source: только от этого автора; ``None`` — от всех.
        @returns: список заметок.
        """
        query = "SELECT * FROM plot_notes WHERE session_id = ? AND status = ?"
        params: list[Any] = [session_id, status]
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [_row_to(PlotNote, row) for row in rows]

    def count_notes(self, session_id: int, source: str, status: str = "active") -> int:
        """Сколько активных заметок этого автора.

        @param session_id: партия.
        @param source: ``player`` или ``model``.
        @param status: статус.
        @returns: количество.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM plot_notes WHERE session_id = ? AND status = ? AND source = ?",
            (session_id, status, source),
        ).fetchone()
        return int(row[0])

    def set_note_text(self, note_id: int, text: str) -> bool:
        """Переписывает текст заметки.

        @param note_id: заметка.
        @param text: новый текст.
        @returns: True, если заметка найдена и обновлена.
        """
        clean = " ".join(text.split())
        if not clean:
            return False
        cursor = self.conn.execute(
            "UPDATE plot_notes SET text = ?, updated_at = ? WHERE id = ?",
            (clean, int(time.time()), note_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def note(self, note_id: int) -> PlotNote | None:
        """Заметка по идентификатору."""
        row = self.conn.execute("SELECT * FROM plot_notes WHERE id = ?", (note_id,)).fetchone()
        return None if row is None else _row_to(PlotNote, row)

    def close_note(self, note_id: int, status: str = "done") -> bool:
        """Отмечает заметку сбывшейся или снятой.

        @param note_id: заметка.
        @param status: ``done`` — сбылась, ``dropped`` — снята без исполнения.
        @returns: ``True``, если запись была найдена и обновлена.
        """
        cursor = self.conn.execute(
            "UPDATE plot_notes SET status = ?, updated_at = ? WHERE id = ?",
            (status, int(time.time()), note_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close_notes(self, note_ids: list[int]) -> int:
        """Отмечает сбывшимися сразу несколько заметок.

        @param note_ids: идентификаторы.
        @returns: сколько записей действительно закрыто.
        """
        closed = 0
        for note_id in note_ids:
            if self.close_note(note_id):
                closed += 1
        return closed

    def delete_note(self, note_id: int) -> None:
        """Удаляет заметку насовсем."""
        self.conn.execute("DELETE FROM plot_notes WHERE id = ?", (note_id,))
        self.conn.commit()

    def delete_notes(self, session_id: int, source: str | None = None) -> int:
        """Удаляет заметки партии разом.

        Нужно, чтобы расчистить стол: задумки копятся, старые мешают, а крестик
        по одному — это десять нажатий и десять подтверждений.

        @param session_id: партия.
        @param source: ``player``, ``model`` или ``None`` для всех.
        @returns: сколько удалено.
        """
        query = "DELETE FROM plot_notes WHERE session_id = ?"
        params: list[Any] = [session_id]
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        cursor = self.conn.execute(query, tuple(params))
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def delete_finished_notes(self, session_id: int) -> int:
        """Удаляет закрытые заметки: сбывшиеся и отброшенные.

        Активные не трогает — они ещё работают.

        @param session_id: партия.
        @returns: сколько удалено.
        """
        cursor = self.conn.execute(
            "DELETE FROM plot_notes WHERE session_id = ? AND status != 'active'",
            (session_id,),
        )
        self.conn.commit()
        return int(cursor.rowcount or 0)

    # --- места --------------------------------------------------------------

    def add_location(
        self,
        world_id: int,
        name: str,
        *,
        prompt: str = "",
        style: str = "",
        seed: int | None = None,
    ) -> int:
        """Заводит место в мире.

        @param world_id: мир.
        @param name: имя места так, как его называет ведущий.
        @returns: идентификатор места.
        """
        now = int(time.time())
        cursor = self.conn.execute(
            "INSERT INTO locations (world_id, name, prompt, style, seed, visits,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (world_id, name.strip(), prompt, style, seed, now, now),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def locations(self, world_id: int, limit: int = 50) -> list[Location]:
        """Места мира, недавние сверху."""
        rows = self.conn.execute(
            "SELECT * FROM locations WHERE world_id = ? ORDER BY updated_at DESC LIMIT ?",
            (world_id, limit),
        ).fetchall()
        return [_row_to(Location, row) for row in rows]

    def location(self, location_id: int) -> Location | None:
        """Место по идентификатору."""
        row = self.conn.execute("SELECT * FROM locations WHERE id = ?", (location_id,)).fetchone()
        return None if row is None else _row_to(Location, row)

    def location_by_name(self, world_id: int, name: str) -> Location | None:
        """Ищет место по имени без учёта регистра и лишних пробелов."""
        cleaned = " ".join(name.split()).casefold()
        if not cleaned:
            return None
        for item in self.locations(world_id, limit=200):
            if " ".join(item.name.split()).casefold() == cleaned:
                return item
        return None

    def update_location(self, location_id: int, changes: dict[str, Any]) -> list[str]:
        """Правит место.

        @param changes: пары «поле — значение».
        @returns: имена отклонённых полей.
        """
        allowed = {"name", "prompt", "style", "seed", "reference_path", "visits"}
        rejected = [key for key in changes if key not in allowed]
        fields = {key: value for key, value in changes.items() if key in allowed}
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            self.conn.execute(
                f"UPDATE locations SET {assignments}, updated_at = ? WHERE id = ?",
                (*fields.values(), int(time.time()), location_id),
            )
            self.conn.commit()
        return rejected

    def touch_location(self, location_id: int) -> None:
        """Отмечает ещё одно посещение места."""
        self.conn.execute(
            "UPDATE locations SET visits = visits + 1, updated_at = ? WHERE id = ?",
            (int(time.time()), location_id),
        )
        self.conn.commit()

    def set_location_reference(self, location_id: int, path: str | None) -> None:
        """Назначает кадр-образец места.

        Образец задаёт внешний вид: следующие кадры этого места рисуются с ним
        как с референсом.

        @param location_id: место.
        @param path: путь к изображению; ``None`` снимает образец.
        """
        self.conn.execute(
            "UPDATE locations SET reference_path = ?, updated_at = ? WHERE id = ?",
            (path, int(time.time()), location_id),
        )
        self.conn.commit()

    def delete_location(self, location_id: int) -> None:
        """Удаляет место."""
        self.conn.execute("DELETE FROM locations WHERE id = ?", (location_id,))
        self.conn.commit()

    # --- сцены --------------------------------------------------------------

    def add_scene(
        self,
        session_id: int | None,
        prompt: str,
        *,
        message_id: int | None = None,
        raw_description: str = "",
        seed: int | None = None,
        status: str = "pending",
        location_id: int | None = None,
    ) -> int:
        """Регистрирует сцену."""
        cursor = self.conn.execute(
            "INSERT INTO scenes (session_id, message_id, prompt, raw_description, seed,"
            " status, location_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, message_id, prompt, raw_description, seed, status, location_id,
             int(time.time())),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def set_scene_seed(self, scene_id: int, seed: int) -> None:
        """Записывает зерно, которым кадр нарисован на самом деле.

        Модель присылает зерно, но часто повторяет одно и то же число, и тогда
        берётся случайное. В записи должно лежать то, что действительно
        использовано: иначе кадр не воспроизвести.

        @param scene_id: сцена.
        @param seed: фактическое зерно.
        """
        self.conn.execute("UPDATE scenes SET seed = ? WHERE id = ?", (int(seed), scene_id))
        self.conn.commit()

    def set_scene_prompt(self, scene_id: int, prompt: str) -> None:
        """Меняет описание кадра и ставит его в очередь на перерисовку.

        Готовый файл при этом остаётся на месте: пока новый кадр не нарисован,
        читателю есть на что смотреть. Перерисовка затрёт его, когда закончится.

        @param scene_id: сцена.
        @param prompt: новое описание.
        """
        self.conn.execute(
            "UPDATE scenes SET prompt = ?, status = 'pending' WHERE id = ?",
            (prompt, scene_id),
        )
        self.conn.commit()

    def finish_scene(
        self,
        scene_id: int,
        path: str | None,
        status: str = "done",
        elapsed_s: float | None = None,
        used_reference: bool = False,
    ) -> None:
        """Отмечает сцену готовой или упавшей.

        @param used_reference: рисовался ли кадр по образцу места — это видно в
            интерфейсе и объясняет, почему кадр похож на предыдущий.
        """
        self.conn.execute(
            "UPDATE scenes SET path = ?, status = ?, elapsed_s = ?, used_reference = ?"
            " WHERE id = ?",
            (path, status, elapsed_s, int(used_reference), scene_id),
        )
        self.conn.commit()

    def scenes(self, session_id: int | None = None, limit: int = 200) -> list[Scene]:
        """Сцены партии в порядке появления."""
        if session_id is None:
            rows = self.conn.execute("SELECT * FROM scenes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM scenes WHERE session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [_row_to(Scene, row) for row in rows]

    def scene(self, scene_id: int) -> Scene | None:
        """Сцена по идентификатору."""
        row = self.conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
        return None if row is None else _row_to(Scene, row)

    def delete_scene(self, scene_id: int) -> None:
        """Удаляет запись о сцене."""
        self.conn.execute("DELETE FROM scenes WHERE id = ?", (scene_id,))
        self.conn.commit()

    # --- предгенерация ------------------------------------------------------

    def add_speculative(
        self,
        session_id: int | None,
        prompt: str,
        *,
        trigger: str = "",
        seed: int | None = None,
        scene_id: int | None = None,
    ) -> int:
        """Кладёт вариант предгенерации в очередь."""
        cursor = self.conn.execute(
            "INSERT INTO speculative (session_id, scene_id, trigger, prompt, seed) VALUES (?, ?, ?, ?, ?)",
            (session_id, scene_id, trigger, prompt, seed),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def pending_speculative(self, session_id: int | None = None, limit: int = 8) -> list[dict[str, Any]]:
        """Варианты, ждущие генерации."""
        if session_id is None:
            rows = self.conn.execute(
                "SELECT * FROM speculative WHERE status = 'pending' ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM speculative WHERE status = 'pending' AND session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_speculative(self, spec_id: int, path: str | None, status: str = "done") -> None:
        """Отмечает вариант готовым или упавшим."""
        self.conn.execute(
            "UPDATE speculative SET path = ?, status = ? WHERE id = ?", (path, status, spec_id)
        )
        self.conn.commit()

    # --- состояние мира -----------------------------------------------------

    def set_state(self, session_id: int, key: str, value: Any) -> None:
        """Записывает значение состояния мира для партии."""
        self.conn.execute(
            "INSERT INTO world_state (session_id, key, value, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (session_id, key, json.dumps(value, ensure_ascii=False), int(time.time())),
        )
        self.conn.commit()

    def get_state(self, session_id: int, key: str, default: Any = None) -> Any:
        """Читает значение состояния мира."""
        row = self.conn.execute(
            "SELECT value FROM world_state WHERE session_id = ? AND key = ?", (session_id, key)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def delete_state(self, session_id: int, key: str) -> None:
        """Удаляет ключ состояния мира."""
        self.conn.execute(
            "DELETE FROM world_state WHERE session_id = ? AND key = ?", (session_id, key)
        )
        self.conn.commit()

    def world_state(self, session_id: int) -> dict[str, Any]:
        """Всё состояние мира партии — оно и уходит в контекст модели."""
        rows = self.conn.execute(
            "SELECT key, value FROM world_state WHERE session_id = ?", (session_id,)
        ).fetchall()
        state: dict[str, Any] = {}
        for row in rows:
            try:
                state[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                state[row["key"]] = row["value"]
        return state

    # --- обслуживание -------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Счётчики записей по таблицам."""
        counts: dict[str, int] = {}
        for table in (
            "worlds", "world_rules", "characters", "sessions",
            "messages", "memories", "scenes", "speculative",
        ):
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = int(row["n"])
        return counts
