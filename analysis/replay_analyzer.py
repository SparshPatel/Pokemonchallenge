from __future__ import annotations

import json
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Replay:
    filename: str
    winner: int | None
    turns: list
    steps: list
    rewards: list
    metadata: dict
    agents: list
    raw: dict


def _find_turns(data: dict) -> list:
    """
    Different replay formats store turn history differently.
    """

    for key in (
        "turns",
        "history",
        "actions",
        "steps",
        "frames",
    ):
        value = data.get(key)
        if isinstance(value, list):
            return value

    return []


def _find_winner(data: dict) -> int | None:

    rewards = data.get("rewards")

    if isinstance(rewards, list) and len(rewards) == 2:

        if rewards[0] > rewards[1]:
            return 0

        if rewards[1] > rewards[0]:
            return 1

    return None


def load_zip(zip_path: str | Path) -> list[Replay]:

    zip_path = Path(zip_path)

    replays = []

    with zipfile.ZipFile(zip_path, "r") as archive:

        for name in archive.namelist():

            if not name.endswith(".json"):
                continue

            try:
                with archive.open(name) as fp:
                    raw = fp.read()

                if len(raw) > 50_000_000:
                    print(f"  Skipped huge replay: {name}")
                    continue

                data = json.loads(raw)

            except MemoryError:
                print(f"  Skipped replay (MemoryError): {name}")
                continue

            except Exception:
                continue

            replay = Replay(
                filename=name,
                winner=_find_winner(data),
                turns=_find_turns(data),
                metadata=data.get("metadata", {}),
                raw=data,
                rewards=data.get("rewards", []),
                steps=data.get("steps", []),
                agents=data.get("info", {}).get("Agents", []),
            )

            replays.append(replay)

    return replays


def load_directory(folder: str | Path) -> list[Replay]:

    folder = Path(folder)

    games = []

    for zip_file in sorted(folder.glob("*.zip")):

        try:
            games.extend(load_zip(zip_file))

        except MemoryError:
            print(f"[Skipped - MemoryError] {zip_file.name}")

        except Exception as e:
            print(f"[Skipped] {zip_file.name} -> {e}")

    return games