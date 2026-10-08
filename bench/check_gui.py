"""Проверка разметки интерфейса без браузера.

Ловит два класса ошибок, которые иначе видны только руками:

* в скрипте есть обращение к элементу, которого нет в разметке;
* функция перерисовки пишет во внутренность контейнера, внутри которого живут
  другие элементы с идентификаторами, — такая перерисовка стирает их, и
  следующий же обработчик падает на исчезнувшем элементе.

Второй случай уже был: список сцен писался во вкладку целиком и стирал кнопку и
счётчик; со второго обновления интерфейс ломался.

Запуск::

    python bench\\check_gui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "gui" / "static" / "index.html"
PASSED = 0
FAILED = 0

ELEMENT_IDS = re.compile(r'id="([^"]+)"')
JS_LOOKUP = re.compile(r'\$\("([^"]+)"\)')
#: Начало функции, которая что-то перерисовывает.
RENDER_FUNC = re.compile(r'function\s+(\w+)\s*\([^)]*\)\s*\{')
INNER_WRITE = re.compile(r'(\w+)\.innerHTML\s*=')


def check(name: str, condition: bool, detail: str = "") -> None:
    """Печатает результат одной проверки."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  OK   {name}")
    else:
        FAILED += 1
        print(f"  ФЕЙЛ {name}{'' if not detail else ' — ' + detail}")


def container_spans(text: str) -> dict[str, tuple[int, int]]:
    """Границы элементов с идентификаторами по смещениям в тексте разметки.

    Считается вложенность тегов ``div``: элемент заканчивается там, где
    закрывается открытый им ``div``.

    @param text: содержимое страницы.
    @returns: ``{идентификатор: (начало, конец)}``.
    """
    spans: dict[str, tuple[int, int]] = {}
    for match in re.finditer(r'<div\b[^>]*id="([^"]+)"[^>]*>', text):
        start = match.end()
        depth = 1
        position = start
        while depth > 0 and position < len(text):
            opened = text.find("<div", position)
            closed = text.find("</div", position)
            if closed == -1:
                break
            if opened != -1 and opened < closed:
                depth += 1
                position = opened + 4
            else:
                depth -= 1
                position = closed + 5
        spans[match.group(1)] = (match.start(), position)
    return spans


def main() -> int:
    setup_console()
    print("\n=== Разметка интерфейса ===\n")

    text = PAGE.read_text(encoding="utf-8")
    ids = set(ELEMENT_IDS.findall(text))
    lookups = set(JS_LOOKUP.findall(text))
    check("страница не пустая", len(text) > 5000, f"{len(text)} символов")
    check("идентификаторы найдены", len(ids) >= 20, f"{len(ids)} штук")
    check("обращения из скрипта найдены", len(lookups) >= 20, f"{len(lookups)} штук")

    missing = sorted(lookups - ids)
    check("все обращения ведут к существующим элементам", not missing, ", ".join(missing))

    # Вкладки и их кнопки: каждая кнопка должна открывать существующую панель.
    panels = {f"tab-{name}" for name in re.findall(r'data-tab="([^"]+)"', text)}
    check("у каждой кнопки вкладки есть панель",
          panels <= ids, ", ".join(sorted(panels - ids)))

    # Перерисовка не должна затирать контейнеры, внутри которых есть элементы с
    # идентификаторами: их сотрёт вместе с содержимым.
    spans = container_spans(text)
    nested: dict[str, list[str]] = {}
    for outer, (start, end) in spans.items():
        inside = [
            match.group(1) for match in ELEMENT_IDS.finditer(text[start:end])
            if match.group(1) != outer
        ]
        if inside:
            nested[outer] = sorted(set(inside))

    writes: dict[str, list[str]] = {}
    for match in RENDER_FUNC.finditer(text):
        name = match.group(1)
        body_start = match.end()
        depth = 1
        position = body_start
        while depth > 0 and position < len(text):
            opened = text.find("{", position)
            closed = text.find("}", position)
            if closed == -1:
                break
            if opened != -1 and opened < closed:
                depth += 1
                position = opened + 1
            else:
                depth -= 1
                position = closed + 1
        for target in INNER_WRITE.findall(text[body_start:position]):
            writes.setdefault(target, []).append(name)

    offenders: list[str] = []
    for target, functions in writes.items():
        # Ищем переменную-получателя: const box = $("scenes") внутри функции.
        for function in functions:
            body = text.split(f"function {function}(", 1)[-1][:2000]
            found = re.search(rf'{re.escape(target)}\s*=\s*\$\("([^"]+)"\)', body)
            if not found:
                continue
            element = found.group(1)
            if element in nested:
                offenders.append(f"{function}: {element} содержит {', '.join(nested[element])}")

    check("перерисовка не затирает вложенные элементы", not offenders, "; ".join(offenders))

    # Сам детектор: на подделанной разметке он обязан сработать, иначе проверка
    # выше ничего не значит.
    sample = (
        '<div id="tab-x"><span id="note"></span><div id="list"></div></div>'
        'function render(){ const box = $("tab-x"); box.innerHTML = "x"; }'
    )
    sample_spans = container_spans(sample)
    start, end = sample_spans["tab-x"]
    sample_nested = sorted(
        match.group(1) for match in ELEMENT_IDS.finditer(sample[start:end])
        if match.group(1) != "tab-x"
    )
    check("детектор находит затирание на подделанной разметке",
          sample_nested == ["list", "note"], str(sample_nested))

    # Кнопки должны лежать в своей вкладке. Правило простое: вкладка отвечает за
    # то, что названо в её заголовке. «Выгрузить мир» во вкладке про партию —
    # именно та ошибка, ради которой эта проверка написана.
    expected = {
        "chat-tools": ["btn-rename", "btn-summarize", "btn-export-session",
                       "btn-clear", "btn-drop-session", "state-json", "timings"],
        "world": ["btn-import", "btn-export-world", "btn-export-bundle",
                  "btn-drop-world", "btn-clean-world", "btn-duplicate-world",
                  "btn-agent-run", "btn-invent", "invent-horizon",
                  "notes-list-world", "btn-drop-my-notes", "btn-drop-his-notes",
                  "btn-drop-all-notes", "items", "rules", "characters"],
        "scenes": ["btn-generate", "locations", "scenes"],
        "model": ["btn-comfy", "btn-start-engine", "model-select", "f-image_policy",
                  "f-male_silhouette", "btn-refresh-models", "f-engine_kind",
                  "f-external_url", "f-external_server_path", "f-external_manage",
                  "f-external_context", "f-external_gpu_layers",
                  "f-external_disable_thinking", "external-settings",
                  "f-external_reasoning_budget", "f-agent_max_steps"],
        "debug": ["btn-orphans", "btn-compact", "btn-dump", "layers"],
    }
    misplaced: list[str] = []
    for tab, wanted in expected.items():
        if f"tab-{tab}" not in spans:
            misplaced.append(f"нет панели {tab}")
            continue
        start, end = spans[f"tab-{tab}"]
        inside = set(ELEMENT_IDS.findall(text[start:end]))
        for element in wanted:
            if element not in inside:
                # Мог оказаться в другой вкладке — это и есть переезд не туда.
                elsewhere = [name for name in expected if name != tab
                             and f"tab-{name}" in spans
                             and element in ELEMENT_IDS.findall(
                                 text[spans[f"tab-{name}"][0]:spans[f"tab-{name}"][1]])]
                where = f" (лежит в «{elsewhere[0]}»)" if elsewhere else " (нет вовсе)"
                misplaced.append(f"{element} не в «{tab}»{where}")
    check("кнопки лежат в своих вкладках", not misplaced, "; ".join(misplaced))

    print(f"\nПройдено: {PASSED}, провалено: {FAILED}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
