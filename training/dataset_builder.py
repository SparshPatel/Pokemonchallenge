"""
dataset_builder.py
Builds a replay dataset from ReplayLoader.
ReplayLoader
      ↓
ReplayGame
      ↓
TrainingSample
This file only creates training samples.
It does NOT perform feature extraction or model training.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from replay_loader import ReplayGame
# =====================================================================
# Training Sample
# =====================================================================
@dataclass(slots=True)
class TrainingSample:
    game_index: int
    replay_name: str
    turn_index: int
    current: dict[str, Any] | None
    select: dict[str, Any] | None
    logs: list[dict[str, Any]]
    reward: float = 0.0
    status: Any = None
    terminal: bool = False
    raw: dict[str, Any] | None = None

# =====================================================================
# Dataset Builder
# =====================================================================
class DatasetBuilder:
    def __init__(self):
        self.samples: list[TrainingSample] = []

    # -------------------------------------------------------------
    def _winner_from_game(
        self,
        game: ReplayGame,
    ) -> int | None:
        """
        Determine winner from the terminal replay frame.
        """
        for turn in reversed(game.turns):
            if turn.raw.get("status") != "DONE":
                continue
            current = turn.current
            if current is None:
                continue
            reward = turn.raw.get("reward")
            if reward is None:
                continue
            your_index = current["yourIndex"]
            if reward > 0:
                return your_index
            if reward < 0:
                return 1 - your_index
        return None

    # -------------------------------------------------------------
    def build(
        self,
        games: list[ReplayGame],
    ) -> list[TrainingSample]:
        self.samples.clear()
        for game_index, game in enumerate(games):
            winner = self._winner_from_game(game)
            turns = [
                t
                for t in game.turns
                if t.current is not None
            ]
            if not turns:
                continue
            final_turn = turns[-1].turn_index
            for turn in turns:
                current = turn.current
                your_index = current.get("yourIndex", 0)
                reward = 0.0
                if winner is not None:
                    if your_index == winner:
                        reward = 1.0
                    else:
                        reward = -1.0
                sample = TrainingSample(
                    game_index=game_index,
                    replay_name=game.replay_name,
                    turn_index=turn.turn_index,
                    current=current,
                    select=turn.select,
                    logs=turn.logs,
                    reward=reward,
                    status=None,
                    terminal=(turn.turn_index == final_turn),
                    raw=turn.raw,
                )
                self.samples.append(sample)
        return self.samples

    # -------------------------------------------------------------
    def summary(self):
        print()
        print("=" * 70)
        print("Dataset Builder Summary")
        print("=" * 70)
        print()
        print(f"Samples : {len(self.samples)}")
        terminals = sum(
            sample.terminal
            for sample in self.samples
        )
        print(f"Terminal Samples : {terminals}")
        rewards = [s.reward for s in self.samples]
        if rewards:
            print(f"Reward Min : {min(rewards)}")
            print(f"Reward Max : {max(rewards)}")
            print(
                f"Wins  : {sum(r > 0 for r in rewards)}"
            )
            print(
                f"Losses: {sum(r < 0 for r in rewards)}"
            )
        print()