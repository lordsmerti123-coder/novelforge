"""Показывает, на каких ходах срабатывает каждая политика генерации картинок.

Картинки не рисуются: проверяется только решение «рисовать сейчас или нет».
Решение принимает `NovelMachine._should_generate_now`, и оно зависит не только
от настройки, но и от состояния партии — от места, показанных лиц и числа
ответов ведущего. Поэтому политики прогоняются по одному и тому же сценарию, а
решение «нарисовано» возвращается в состояние: иначе сымитировать цикл нельзя.

Запуск::

    python bench\\policies.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel.console import setup_console  # noqa: E402
from novel.db import NovelDB  # noqa: E402
from novel.machine import NovelMachine  # noqa: E402
from novel.models import ModelRegistry  # noqa: E402
from novel.settings import SettingsStore  # noqa: E402


@dataclass
class Step:
    """Один ход сценария."""

    turn: int
    place: str
    #: Место названо впервые в этой партии.
    new_place: bool = False
    #: В кадре есть лицо, которого ещё не было.
    new_faces: list[str] = field(default_factory=list)
    #: Ведущий отметил резкую перемену обстановки.
    sudden: bool = False
    #: Ведущий вообще не поставил блок сцены.
    silent: bool = False

    def describe(self) -> str:
        """Короткое описание хода для таблицы."""
        bits = []
        if self.silent:
            bits.append("без сцены")
        else:
            bits.append(self.place)
            if self.new_place:
                bits.append("новое место")
            if self.new_faces:
                bits.append("новое лицо: " + ", ".join(self.new_faces))
            if self.sudden:
                bits.append("резкая перемена")
            if len(bits) == 1:
                bits.append("то же место, те же лица")
        return ", ".join(bits)


#: Один и тот же сценарий для всех политик.
SCRIPT: list[Step] = [
    Step(1, "таверна", new_place=True, new_faces=["Марла"]),
    Step(2, "таверна"),
    Step(3, "таверна", sudden=True),
    Step(4, "улица", new_place=True),
    Step(5, "улица", silent=True),
    Step(6, "улица", new_faces=["Сурр"]),
    Step(7, "улица"),
    Step(8, "улица"),
]

POLICIES = ["minimal", "master", "on_scene_change", "every_turn", "every_n",
            "manual", "never", "idle"]


def run_policy(policy: str, every_n: int = 3) -> list[bool]:
    """Прогоняет сценарий и возвращает решения по ходам.

    @param policy: настройка частоты картинок.
    @param every_n: шаг для политики «каждые N ходов».
    @returns: список «рисовать или нет» по числу ходов.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = NovelDB(Path(tmp) / "policy.db")
        try:
            world_id = db.create_world(name="Политики", format="story")
            session_id = db.create_session(world_id, "Партия")
            store = SettingsStore(Path(tmp) / "settings.json")
            store.settings.image_policy = policy
            store.settings.image_every_n = every_n
            machine = NovelMachine(db, store, ModelRegistry())

            decisions: list[bool] = []
            shown: set[str] = set()
            for step in SCRIPT:
                # Ход уже случился: ведущий ответил, место и лица записаны.
                db.add_message(session_id, "assistant", f"ответ на ходу {step.turn}")
                db.set_state(session_id, "location", step.place)

                outcome = _outcome(step, shown)
                decision = bool(outcome.scene_ids) and machine._should_generate_now(
                    session_id, outcome
                )
                decisions.append(decision)

                # Кадр нарисован: место становится «последним нарисованным».
                # Без этого шага цикл не сымитировать — политика «при смене
                # места» сравнивает текущее место именно с ним.
                if decision:
                    db.set_state(session_id, "last_image_location", step.place)
                    if outcome.new_characters:
                        db.set_state(session_id, "shown_characters",
                                     sorted(shown | set(outcome.new_characters)))
                shown |= set(outcome.new_characters)
            return decisions
        finally:
            db.close()


def _outcome(step: Step, shown: set[str]):
    """Собирает итог хода так, как его видит решающая функция."""
    from novel.machine import TurnOutcome

    outcome = TurnOutcome()
    if not step.silent:
        outcome.scene_ids = [step.turn]
        outcome.new_place = step.new_place
        outcome.sudden = step.sudden
        outcome.new_characters = [name for name in step.new_faces if name not in shown]
    return outcome


def main() -> int:
    setup_console()
    print("\n=== Когда срабатывает генерация картинок ===\n")
    print("Сценарий:")
    for step in SCRIPT:
        print(f"  ход {step.turn}: {step.describe()}")
    print()

    width = max(len(step.describe()) for step in SCRIPT) + 4
    header = "политика".ljust(16) + "".join(f" {step.turn} " for step in SCRIPT) + "  итого"
    print(header)
    print("-" * len(header))
    for policy in POLICIES:
        decisions = run_policy(policy)
        marks = "".join(" ✓ " if flag else " · " for flag in decisions)
        label = policy if policy != "every_n" else "every_n (3)"
        print(f"{label:<16}{marks}  {sum(decisions)} из {len(decisions)}")

    print()
    print("Пояснения:")
    print("  minimal          новое место, новое лицо или резкая перемена")
    print("  master           ведущий поставил блок сцены — рисуем")
    print("  on_scene_change  место отличается от последнего нарисованного")
    print("  every_turn       каждый ход, где есть блок сцены")
    print("  every_n          каждый N-й ответ ведущего")
    print("  manual, never, idle   сами не запускают, ждут кнопки или простоя")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
