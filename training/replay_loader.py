"""
Replay loader for Pokemon Challenge.
Responsibilities
----------------
Uses ReplayParser output and converts replay JSON files into strongly
typed Python objects.
This file DOES NOT:
    - train models
    - extract tensors
    - perform feature engineering
    - evaluate positions
It only reconstructs replay structure.
Pipeline
--------
ReplayParser
        ↓
ReplayLoader
        ↓
ReplayGame
        ↓
ReplayTurn
"""
from __future__ import annotations
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =====================================================================
# Replay Objects
# =====================================================================
@dataclass(slots=True)
class ReplayTurn:
    turn_index: int
    current: dict[str, Any] | None = None
    select: dict[str, Any] | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ReplayGame:
    replay_name: str
    source_zip: Path
    initial_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    turns: list[ReplayTurn] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

# =====================================================================
# Replay Loader
# =====================================================================
class ReplayLoader:
    def __init__(self):
        self.games: list[ReplayGame] = []

    # -----------------------------------------------------------------
    def load_from_parser(
        self,
        parser,
    ) -> list[ReplayGame]:
        self.games.clear()
        for replay_zip in parser.games:
            zip_path: Path = replay_zip["zip"]
            with zipfile.ZipFile(zip_path, "r") as archive:
                for replay_file in replay_zip["games"]:
                    try:
                        raw = archive.read(replay_file)
                        replay_json = json.loads(
                            raw.decode("utf-8")
                        )
                        game = self._parse_game(
                            replay_json,
                            replay_file,
                            zip_path,
                        )
                        if game is not None:
                            self.games.append(game)
                    except Exception as exc:
                        print(
                            f"[FAILED] {zip_path.name} :: {replay_file}"
                        )
                        print(exc)
        return self.games

    # -----------------------------------------------------------------
    def _parse_game(
        self,
        replay_json: dict,
        replay_name: str,
        source_zip: Path,
    ) -> ReplayGame:
        game = ReplayGame(
            replay_name=replay_name,
            source_zip=source_zip,
        )
        game.raw = replay_json
        game.initial_data = replay_json.get("initialData")
        for key, value in replay_json.items():
            if key in (
                "turns",
                "history",
                "states",
                "frames",
                "records",
            ):
                continue
            if key == "initialData":
                continue
            game.metadata[key] = value
        turns = self._discover_turns(replay_json)
        for index, frame in enumerate(turns):
            replay_turn = ReplayTurn(
                turn_index=index
            )
            replay_turn.raw = frame
            observation = frame["observation"]
            replay_turn.current = observation.get("current")
            replay_turn.select = observation.get("select")
            replay_turn.logs = observation.get("logs", [])
            game.turns.append(replay_turn)
        return game

    # -----------------------------------------------------------------
    def _discover_turns(
        self,
        replay_json: dict,
    ) -> list[dict]:
        steps = replay_json.get("steps")
        if not isinstance(steps, list):
            return []
        turns = []
        for step in steps:
            if not isinstance(step, list):
                continue
            if len(step) == 0:
                continue
            turns.append(step[0])
        return turns

    # -----------------------------------------------------------------
    def summary(self):
        print()
        print("=" * 70)
        print("Replay Loader Summary")
        print("=" * 70)
        print()
        print(
            f"Games Loaded : {len(self.games)}"
        )
        total_turns = sum(
            game.num_turns
            for game in self.games
        )
        print(
            f"Total Turns  : {total_turns}"
        )
        if self.games:
            average = (
                total_turns
                / len(self.games)
            )
            print(
                f"Average Turns/Game : {average:.2f}"
            )
        print()