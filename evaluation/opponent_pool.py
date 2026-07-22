from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import random


# =====================================================================
# Opponent Definition
# =====================================================================

@dataclass(slots=True)
class Opponent:
    """
    Generic opponent definition.

    An opponent is treated as a black box by the tournament system.

    The tournament runner only needs:
        - a name
        - a 60-card deck
        - a function that receives an observation
        - a function that returns list[int] selections
    """

    name: str

    deck: list[int]

    choose_fn: Callable[
        [dict[str, Any]],
        list[int],
    ]

    metadata: dict[str, Any] | None = None

    def choose(
        self,
        observation: dict[str, Any],
    ) -> list[int]:
        """
        Request a selection from the opponent.
        """

        result = self.choose_fn(
            observation
        )

        if not isinstance(
            result,
            list,
        ):
            raise TypeError(
                f"Opponent '{self.name}' must return list[int]. "
                f"Received {type(result).__name__}."
            )

        if not all(
            isinstance(
                index,
                int,
            )
            for index in result
        ):
            raise TypeError(
                f"Opponent '{self.name}' returned "
                "a selection that is not list[int]."
            )

        return result


# =====================================================================
# Replay Policy
# =====================================================================

class ReplayPolicy:
    """
    Replay-backed opponent policy.

    A Kaggle replay stores actions as option indices.

    The policy extracts the action sequence belonging to one player
    from the replay and replays those selections in order.

    If the exact historical action is no longer valid in the current
    game state, the policy attempts to repair it by selecting a valid
    action with the same selection cardinality.

    This allows replay decks to be used as evaluation opponents without
    requiring the original Kaggle agent code.
    """

    def __init__(
        self,
        actions: list[list[int]],
        name: str,
    ):
        self.actions = [
            list(action)
            for action in actions
        ]

        self.name = name

        self.position = 0

    def __call__(
        self,
        observation: dict[str, Any],
    ) -> list[int]:
        """
        Return the next replayed action.

        If the historical action is valid for the current observation,
        return it directly.

        Otherwise attempt a generic validity repair.
        """

        select = observation.get(
            "select"
        )

        if not isinstance(
            select,
            dict,
        ):
            return []

        options = select.get(
            "option",
            [],
        )

        if not isinstance(
            options,
            list,
        ):
            return []

        min_count = select.get(
            "minCount",
            0,
        )

        max_count = select.get(
            "maxCount",
            len(options),
        )

        # -------------------------------------------------------------
        # Find the next historical action.
        # -------------------------------------------------------------

        historical_action: list[int] = []

        if self.position < len(
            self.actions
        ):
            historical_action = list(
                self.actions[
                    self.position
                ]
            )

            self.position += 1

        # -------------------------------------------------------------
        # Empty action.
        # -------------------------------------------------------------

        if not historical_action:
            if min_count == 0:
                return []

            return self._fallback_action(
                options=options,
                min_count=min_count,
                max_count=max_count,
            )

        # -------------------------------------------------------------
        # Check whether historical action is valid.
        # -------------------------------------------------------------

        if self._is_valid_action(
            historical_action,
            options,
            min_count,
            max_count,
        ):
            return historical_action

        # -------------------------------------------------------------
        # Attempt to repair the action.
        # -------------------------------------------------------------

        repaired = self._repair_action(
            historical_action=historical_action,
            options=options,
            min_count=min_count,
            max_count=max_count,
        )

        return repaired

    @staticmethod
    def _is_valid_action(
        action: list[int],
        options: list[Any],
        min_count: int,
        max_count: int,
    ) -> bool:
        """
        Check generic CG selection validity.
        """

        if len(action) < min_count:
            return False

        if len(action) > max_count:
            return False

        if len(
            set(action)
        ) != len(action):
            return False

        for index in action:
            if index < 0:
                return False

            if index >= len(
                options
            ):
                return False

        return True

    @staticmethod
    def _repair_action(
        historical_action: list[int],
        options: list[Any],
        min_count: int,
        max_count: int,
    ) -> list[int]:
        """
        Repair an invalid historical selection.

        The replay action may become invalid because the current match
        has diverged from the historical game.

        We preserve as much of the original selection as possible.
        """

        if not options:
            return []

        target_count = len(
            historical_action
        )

        target_count = max(
            min_count,
            target_count,
        )

        target_count = min(
            max_count,
            target_count,
        )

        if target_count <= 0:
            return []

        repaired: list[int] = []

        # -------------------------------------------------------------
        # Preserve valid historical indices first.
        # -------------------------------------------------------------

        for index in historical_action:
            if (
                0 <= index < len(options)
                and index not in repaired
            ):
                repaired.append(
                    index
                )

            if len(
                repaired
            ) >= target_count:
                break

        # -------------------------------------------------------------
        # Fill remaining selections with valid option indices.
        # -------------------------------------------------------------

        for index in range(
            len(options)
        ):
            if index in repaired:
                continue

            repaired.append(
                index
            )

            if len(
                repaired
            ) >= target_count:
                break

        # -------------------------------------------------------------
        # Ensure minimum count.
        # -------------------------------------------------------------

        if len(
            repaired
        ) < min_count:
            return ReplayPolicy._fallback_action(
                options=options,
                min_count=min_count,
                max_count=max_count,
            )

        return repaired

    @staticmethod
    def _fallback_action(
        options: list[Any],
        min_count: int,
        max_count: int,
    ) -> list[int]:
        """
        Generic fallback when no historical action can be used.
        """

        if not options:
            return []

        count = max(
            min_count,
            0,
        )

        count = min(
            count,
            max_count,
            len(options),
        )

        if count <= 0:
            return []

        return list(
            range(
                count
            )
        )


# =====================================================================
# Replay Parsing Helpers
# =====================================================================

def _iter_replay_files(
    replay_dir: Path,
) -> Iterable[Path]:
    """
    Yield replay JSON files from:
        1. Normal JSON files on disk.
        2. JSON files nested inside ZIP archives.

    ZIP-contained replay paths are returned using a synthetic
    path-like representation. The actual JSON content is loaded
    separately by `_load_replay()`.
    """

    if not replay_dir.exists():
        return

    if not replay_dir.is_dir():
        return

    # -------------------------------------------------------------
    # Normal JSON replay files.
    # -------------------------------------------------------------

    for path in sorted(
        replay_dir.rglob(
            "*.json"
        )
    ):
        yield path

    # -------------------------------------------------------------
    # ZIP archives.
    # -------------------------------------------------------------

    for zip_path in sorted(
        replay_dir.rglob(
            "*.zip"
        )
    ):
        yield from _iter_zip_replay_files(
            zip_path
        )

def _iter_zip_replay_files(
    zip_path: Path,
) -> Iterable[Path]:
    """
    Yield JSON replay entries contained inside a ZIP archive.

    The ZIP archive itself is represented by a synthetic path:

        archive.zip::path/inside/archive.json

    `_load_replay()` recognizes this representation and reads
    the JSON directly from the archive.
    """

    import zipfile

    try:
        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            for member in sorted(
                archive.namelist()
            ):

                if not member.lower().endswith(
                    ".json"
                ):
                    continue

                if member.endswith(
                    "/"
                ):
                    continue

                yield Path(
                    f"{zip_path}::{member}"
                )

    except (
        OSError,
        zipfile.BadZipFile,
    ):
        return

def _extract_player_names(
    replay: dict[str, Any],
) -> list[str]:
    """
    Extract player names from the Kaggle replay metadata.
    """

    info = replay.get(
        "info"
    )

    if not isinstance(
        info,
        dict,
    ):
        return [
            "Player 0",
            "Player 1",
        ]

    agents = info.get(
        "Agents"
    )

    if not isinstance(
        agents,
        list,
    ):
        return [
            "Player 0",
            "Player 1",
        ]

    names: list[str] = []

    for index, agent in enumerate(
        agents
    ):
        if isinstance(
            agent,
            dict,
        ):
            name = agent.get(
                "Name"
            )

            if isinstance(
                name,
                str,
            ) and name.strip():
                names.append(
                    name.strip()
                )
                continue

        names.append(
            f"Player {index}"
        )

    while len(
        names
    ) < 2:
        names.append(
            f"Player {len(names)}"
        )

    return names[:2]


def _extract_decks_from_replay(
    replay: dict[str, Any],
) -> dict[int, list[int]]:
    """
    Extract the initial 60-card decks from the replay.

    Actual replay structure:

        steps
          -> step
            -> player record
              -> visualize
                -> current
                  -> players
                    -> deck

    Each deck card is represented as:

        {
            "id": 1142,
            "name": "...",
            "playerIndex": 0,
            "serial": 42
        }

    We use card IDs only.
    """

    steps = replay.get(
        "steps"
    )

    if not isinstance(
        steps,
        list,
    ):
        return {}

    decks: dict[int, list[int]] = {}

    # -------------------------------------------------------------
    # Walk every replay step.
    # -------------------------------------------------------------

    for step in steps:

        if not isinstance(
            step,
            list,
        ):
            continue

        for player_step in step:

            if not isinstance(
                player_step,
                dict,
            ):
                continue

            visualize = player_step.get(
                "visualize"
            )

            if not isinstance(
                visualize,
                list,
            ):
                continue

            for visualization in visualize:

                if not isinstance(
                    visualization,
                    dict,
                ):
                    continue

                current = visualization.get(
                    "current"
                )

                if not isinstance(
                    current,
                    dict,
                ):
                    continue

                players = current.get(
                    "players"
                )

                if not isinstance(
                    players,
                    list,
                ):
                    continue

                # -------------------------------------------------
                # Extract each player's deck.
                # -------------------------------------------------

                for player_index, player in enumerate(
                    players
                ):

                    if not isinstance(
                        player,
                        dict,
                    ):
                        continue

                    if player_index in decks:
                        # We already found a valid initial deck.
                        continue

                    deck = player.get(
                        "deck"
                    )

                    if not isinstance(
                        deck,
                        list,
                    ):
                        continue

                    card_ids: list[int] = []

                    for card in deck:

                        if not isinstance(
                            card,
                            dict,
                        ):
                            continue

                        card_id = card.get(
                            "id"
                        )

                        if isinstance(
                            card_id,
                            int,
                        ):
                            card_ids.append(
                                card_id
                            )

                    # -------------------------------------------------
                    # A valid Pokémon TCG deck must have exactly
                    # 60 cards.
                    # -------------------------------------------------

                    if len(
                        card_ids
                    ) == 60:
                        decks[
                            player_index
                        ] = card_ids

                # -------------------------------------------------
                # Once both decks are found, stop searching.
                # -------------------------------------------------

                if len(
                    decks
                ) >= 2:
                    return decks

    return decks


def _extract_actions_from_replay(
    replay: dict[str, Any],
) -> dict[int, list[list[int]]]:
    """
    Extract action selections from a Kaggle replay.

    The replay contains per-step records like:

        {
            "action": [...],
            "observation": ...,
            ...
        }

    The player index is inferred from the observation's
    `current.yourIndex` field.

    Actions are stored separately for player 0 and player 1.
    """

    steps = replay.get(
        "steps"
    )

    if not isinstance(
        steps,
        list,
    ):
        return {
            0: [],
            1: [],
        }

    actions: dict[
        int,
        list[list[int]],
    ] = {
        0: [],
        1: [],
    }

    for step in steps:

        if not isinstance(
            step,
            list,
        ):
            continue

        for player_step in step:

            if not isinstance(
                player_step,
                dict,
            ):
                continue

            action = player_step.get(
                "action"
            )

            if not isinstance(
                action,
                list,
            ):
                continue

            # ---------------------------------------------------------
            # The action at this level is normally the action submitted
            # by the player represented by this step.
            #
            # We identify the player through the observation.
            # ---------------------------------------------------------

            observation = player_step.get(
                "observation"
            )

            player_index: int | None = None

            if isinstance(
                observation,
                dict,
            ):
                current = observation.get(
                    "current"
                )

                if isinstance(
                    current,
                    dict,
                ):
                    your_index = current.get(
                        "yourIndex"
                    )

                    if your_index in (
                        0,
                        1,
                    ):
                        player_index = your_index

            # ---------------------------------------------------------
            # If observation does not expose current.yourIndex,
            # attempt to infer from the visualize snapshot.
            # ---------------------------------------------------------

            if player_index is None:

                visualize = player_step.get(
                    "visualize"
                )

                if isinstance(
                    visualize,
                    list,
                ):

                    for visualization in visualize:

                        if not isinstance(
                            visualization,
                            dict,
                        ):
                            continue

                        current = visualization.get(
                            "current"
                        )

                        if not isinstance(
                            current,
                            dict,
                        ):
                            continue

                        your_index = current.get(
                            "yourIndex"
                        )

                        if your_index in (
                            0,
                            1,
                        ):
                            player_index = (
                                your_index
                            )
                            break

            if player_index not in (
                0,
                1,
            ):
                continue

            # ---------------------------------------------------------
            # Normalize action into list[int].
            #
            # Kaggle replay actions can occasionally be nested.
            # ---------------------------------------------------------

            normalized_action: list[int] = []

            for value in action:

                if isinstance(
                    value,
                    int,
                ):
                    normalized_action.append(
                        value
                    )

            actions[
                player_index
            ].append(
                normalized_action
            )

    return actions


def _load_replay(
    path: Path,
) -> dict[str, Any] | None:
    """
    Load one replay JSON file.

    Supports both:
        - regular JSON files
        - JSON files stored inside ZIP archives

    ZIP paths use the format:

        archive.zip::internal/file.json
    """

    import zipfile

    path_string = str(
        path
    )

    # -------------------------------------------------------------
    # ZIP-contained replay.
    # -------------------------------------------------------------

    if "::" in path_string:

        archive_path_string, member_name = (
            path_string.split(
                "::",
                1,
            )
        )

        archive_path = Path(
            archive_path_string
        )

        try:

            with zipfile.ZipFile(
                archive_path,
                "r",
            ) as archive:

                raw = archive.read(
                    member_name
                )

            replay = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except (
            OSError,
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return None

    # -------------------------------------------------------------
    # Normal filesystem replay.
    # -------------------------------------------------------------

    else:

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                replay = json.load(
                    handle
                )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

    # -------------------------------------------------------------
    # Validate top-level structure.
    # -------------------------------------------------------------

    if not isinstance(
        replay,
        dict,
    ):
        return None

    return replay


# =====================================================================
# Opponent Pool
# =====================================================================

class OpponentPool:
    """
    Collection of opponents used by the generic tournament harness.

    The pool itself has no knowledge of the candidate agent.
    """

    def __init__(
        self,
        opponents: Iterable[Opponent] | None = None,
    ):
        self._opponents: list[Opponent] = []

        if opponents is not None:

            for opponent in opponents:

                self.add(
                    opponent
                )

    def add(
        self,
        opponent: Opponent,
    ) -> None:
        """
        Add an opponent to the pool.
        """

        if not isinstance(
            opponent,
            Opponent,
        ):
            raise TypeError(
                "OpponentPool accepts only Opponent objects."
            )

        if not opponent.name:
            raise ValueError(
                "Opponent name cannot be empty."
            )

        if len(
            opponent.deck
        ) != 60:
            raise ValueError(
                f"Opponent '{opponent.name}' must have "
                "exactly 60 cards."
            )

        self._opponents.append(
            opponent
        )

    def remove(
        self,
        name: str,
    ) -> None:
        """
        Remove an opponent by name.
        """

        self._opponents = [
            opponent
            for opponent in self._opponents
            if opponent.name != name
        ]

    def get(
        self,
        name: str,
    ) -> Opponent:
        """
        Retrieve an opponent by name.
        """

        for opponent in self._opponents:

            if opponent.name == name:
                return opponent

        raise KeyError(
            f"Opponent '{name}' not found."
        )

    def all(
        self,
    ) -> list[Opponent]:
        """
        Return all registered opponents.
        """

        return list(
            self._opponents
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._opponents
        )

    def __iter__(
        self,
    ):
        return iter(
            self._opponents
        )

    # =================================================================
    # Replay Loader
    # =================================================================

    @classmethod
    def from_replays(
        cls,
        replay_dir: str | Path,
        candidate_name: str | None = None,
        max_opponents: int | None = None,
    ) -> "OpponentPool":
        """
        Build an opponent pool from Kaggle replay JSON files.

        For every replay:

            1. Load replay.
            2. Extract both 60-card decks.
            3. Identify the opponent relative to candidate_name.
            4. Extract that player's historical actions.
            5. Build a replay-backed Opponent.

        If candidate_name is None, player 1 is used as the opponent.

        This is compatible with replay files whose structure is:

            info.Agents
            steps[*][*].visualize[*].current.players[*].deck

        Args:
            replay_dir:
                Directory containing replay JSON files.

            candidate_name:
                Name of the candidate agent in replay metadata.
                If provided, the other player becomes the opponent.

            max_opponents:
                Optional maximum number of opponents to load.
        """

        replay_path = Path(
            replay_dir
        )

        pool = cls()

        files = list(
            _iter_replay_files(
                replay_path
            )
        )

        if not files:
            raise RuntimeError(
                f"No replay JSON files found in: "
                f"{replay_path}"
            )

        usable = 0

        skipped = 0

        for path in files:

            replay = _load_replay(
                path
            )

            if replay is None:
                skipped += 1
                continue

            player_names = (
                _extract_player_names(
                    replay
                )
            )

            decks = (
                _extract_decks_from_replay(
                    replay
                )
            )

            if len(
                decks
            ) < 2:
                skipped += 1
                continue

            # ---------------------------------------------------------
            # Determine candidate and opponent player indices.
            # ---------------------------------------------------------

            opponent_index = 1

            candidate_index = 0

            if candidate_name:

                normalized_candidate = (
                    candidate_name
                    .strip()
                    .lower()
                )

                matching_indices = [
                    index
                    for index, name in enumerate(
                        player_names
                    )
                    if name.strip().lower()
                    == normalized_candidate
                ]

                if matching_indices:

                    candidate_index = (
                        matching_indices[0]
                    )

                    opponent_index = (
                        1
                        if candidate_index == 0
                        else 0
                    )

            # ---------------------------------------------------------
            # Extract opponent deck.
            # ---------------------------------------------------------

            opponent_deck = decks.get(
                opponent_index
            )

            if not opponent_deck:
                skipped += 1
                continue

            if len(
                opponent_deck
            ) != 60:
                skipped += 1
                continue

            # ---------------------------------------------------------
            # Extract replay actions.
            # ---------------------------------------------------------

            actions = (
                _extract_actions_from_replay(
                    replay
                )
            )

            opponent_actions = actions.get(
                opponent_index,
                [],
            )

            # ---------------------------------------------------------
            # Construct replay policy.
            # ---------------------------------------------------------

            opponent_name = (
                player_names[
                    opponent_index
                ]
                if opponent_index
                < len(
                    player_names
                )
                else (
                    f"Player "
                    f"{opponent_index}"
                )
            )

            replay_id = replay.get(
                "id"
            )

            episode_id = (
                replay.get(
                    "info",
                    {}
                ).get(
                    "EpisodeId"
                )
                if isinstance(
                    replay.get(
                        "info"
                    ),
                    dict,
                )
                else None
            )

            policy = ReplayPolicy(
                actions=opponent_actions,
                name=opponent_name,
            )

            # ---------------------------------------------------------
            # Make opponent name unique per replay.
            # ---------------------------------------------------------

            unique_name = (
                f"{opponent_name}"
                f"__replay_"
                f"{path.stem}"
            )

            metadata = {
                "source": "kaggle_replay",
                "replay_file": str(
                    path
                ),
                "replay_id": replay_id,
                "episode_id": episode_id,
                "player_index": opponent_index,
                "candidate_index": candidate_index,
                "player_name": opponent_name,
                "actions": len(
                    opponent_actions
                ),
            }

            pool.add(
                Opponent(
                    name=unique_name,
                    deck=list(
                        opponent_deck
                    ),
                    choose_fn=policy,
                    metadata=metadata,
                )
            )

            usable += 1

            if (
                max_opponents is not None
                and usable >= max_opponents
            ):
                break

        print(
            ""
        )

        print(
            "Replay loading summary:"
        )

        print(
            f"  replay files found : "
            f"{len(files)}"
        )

        print(
            f"  usable opponents   : "
            f"{usable}"
        )

        print(
            f"  skipped replays    : "
            f"{skipped}"
        )

        if len(
            pool
        ) == 0:
            raise RuntimeError(
                "No usable replay opponents found in: "
                f"{replay_path}"
            )

        return pool


# =====================================================================
# Randomized Opponent Selection
# =====================================================================

class OpponentScheduler:
    """
    Controls the order in which opponents are presented to the candidate.
    """

    def __init__(
        self,
        pool: OpponentPool,
        seed: int = 42,
    ):
        if len(
            pool
        ) == 0:
            raise ValueError(
                "Cannot create scheduler from an empty opponent pool."
            )

        self.pool = pool

        self.rng = random.Random(
            seed
        )

    def shuffled(
        self,
    ) -> list[Opponent]:
        """
        Return opponents in randomized order.
        """

        opponents = self.pool.all()

        self.rng.shuffle(
            opponents
        )

        return opponents

    def round_robin(
        self,
        games_per_opponent: int,
    ) -> list[Opponent]:
        """
        Build a randomized round-robin schedule.

        Every registered opponent appears exactly
        `games_per_opponent` times.
        """

        if games_per_opponent <= 0:
            raise ValueError(
                "games_per_opponent must be positive."
            )

        schedule: list[Opponent] = []

        for _ in range(
            games_per_opponent
        ):

            round_opponents = (
                self.shuffled()
            )

            schedule.extend(
                round_opponents
            )

        return schedule


# =====================================================================
# Simple Opponent Factory
# =====================================================================

def create_opponent(
    name: str,
    deck: list[int],
    choose_fn: Callable[
        [dict[str, Any]],
        list[int],
    ],
    metadata: dict[str, Any] | None = None,
) -> Opponent:
    """
    Convenience factory for creating an opponent.
    """

    return Opponent(
        name=name,
        deck=list(
            deck
        ),
        choose_fn=choose_fn,
        metadata=metadata,
    )