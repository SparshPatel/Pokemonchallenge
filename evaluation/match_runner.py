from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from submission.cg import game


@dataclass(slots=True)
class MatchOutcome:
    """Generic result returned by the CG engine after a match."""

    winner_player_index: int | None
    termination_reason: int | None
    selections: int
    final_observation: dict[str, Any] | None


class MatchRunner:
    """
    Generic neutral match runner.

    This class has no knowledge of:
        - Pokémon strategy
        - decks
        - planners
        - value networks
        - heuristics
        - Kaggle-specific agent internals

    It only knows how to:
        1. Start a CG battle.
        2. Determine which player must act.
        3. Ask that player's callable for a selection.
        4. Sanitize the selection against the engine contract.
        5. Validate the selection.
        6. Submit the selection to the CG engine.
        7. Continue until the match ends.
        8. Return the generic match outcome.

    Each player is represented by a callable:

        observation -> list[int]

    The callable receives the raw CG observation and must return
    selection indices expected by the CG engine.
    """

    def __init__(
        self,
        max_selections: int = 10000,
        verbose: bool = False,
    ):
        if max_selections <= 0:
            raise ValueError("max_selections must be positive.")

        self.max_selections = max_selections
        self.verbose = verbose

    def run(
        self,
        deck0: list[int],
        deck1: list[int],
        player0: Callable[[dict[str, Any]], list[int]],
        player1: Callable[[dict[str, Any]], list[int]],
    ) -> MatchOutcome:
        """
        Run one complete battle.

        Args:
            deck0:
                Exactly 60 card IDs for player 0.

            deck1:
                Exactly 60 card IDs for player 1.

            player0:
                Callable that receives an observation and returns
                selection indices for player 0.

            player1:
                Callable that receives an observation and returns
                selection indices for player 1.

        Returns:
            MatchOutcome containing the winner, termination reason,
            number of engine selections, and final observation.
        """

        self._validate_deck(deck0, "deck0")
        self._validate_deck(deck1, "deck1")

        if not callable(player0):
            raise TypeError("player0 must be callable.")

        if not callable(player1):
            raise TypeError("player1 must be callable.")

        observation: dict[str, Any] | None = None
        selections = 0
        battle_started = False

        try:
            observation, _start_data = game.battle_start(
                list(deck0),
                list(deck1),
            )

            battle_started = True

            if observation is None:
                return MatchOutcome(
                    winner_player_index=None,
                    termination_reason=None,
                    selections=0,
                    final_observation=None,
                )

            while True:
                if selections >= self.max_selections:
                    raise RuntimeError(
                        "Match exceeded maximum number "
                        f"of selections ({self.max_selections})."
                    )

                current = observation.get("current")

                if current is None:
                    raise RuntimeError(
                        "CG engine returned an observation "
                        "without current state."
                    )

                result = current.get("result", -1)

                if result != -1:
                    return MatchOutcome(
                        winner_player_index=(
                            result
                            if result in (0, 1)
                            else None
                        ),
                        termination_reason=(
                            self._extract_termination_reason(
                                observation
                            )
                        ),
                        selections=selections,
                        final_observation=observation,
                    )

                acting_player = current.get("yourIndex")

                if acting_player not in (0, 1):
                    raise RuntimeError(
                        "Invalid acting player index: "
                        f"{acting_player}"
                    )

                agent = (
                    player0
                    if acting_player == 0
                    else player1
                )

                selection = agent(observation)

                selection = self._sanitize_selection(
                    observation,
                    selection,
                )

                self._validate_selection(
                    observation,
                    selection,
                )

                observation = game.battle_select(
                    selection
                )

                selections += 1

                if self.verbose:
                    self._print_progress(
                        observation,
                        selections,
                    )

        finally:
            if battle_started:
                try:
                    game.battle_finish()
                except Exception:
                    pass

    @staticmethod
    def _validate_deck(
        deck: list[int],
        name: str,
    ) -> None:
        """
        Validate a deck before passing it to the CG engine.

        This is intentionally generic and only checks the engine-level
        requirement that a deck contains exactly 60 integer card IDs.
        """

        if not isinstance(deck, list):
            raise TypeError(
                f"{name} must be list[int]."
            )

        if len(deck) != 60:
            raise ValueError(
                f"{name} must contain exactly 60 cards. "
                f"Received {len(deck)}."
            )

        if not all(
            isinstance(card_id, int)
            for card_id in deck
        ):
            raise TypeError(
                f"{name} must contain only integer card IDs."
            )

    @staticmethod
    def _sanitize_selection(
        observation: dict[str, Any],
        selection: list[int],
    ) -> list[int]:
        """
        Enforce the engine's selection contract immediately before submission.

        The engine's maxCount is authoritative. If the agent accidentally
        returns more selections than permitted, retain the earliest legal
        selections and truncate to maxCount.

        Invalid indices and duplicate indices are removed while preserving
        the original order.
        """

        if not isinstance(selection, list):
            return []

        select = observation.get("select")

        if not isinstance(select, dict):
            return []

        options = select.get("option", [])

        if not isinstance(options, list):
            options = []

        min_count = select.get(
            "minCount",
            0,
        )

        max_count = select.get(
            "maxCount",
            len(options),
        )

        try:
            min_count = max(
                0,
                int(min_count),
            )
        except Exception:
            min_count = 0

        try:
            max_count = max(
                0,
                int(max_count),
            )
        except Exception:
            max_count = len(options)

        max_count = min(
            max_count,
            len(options),
        )

        cleaned: list[int] = []
        seen: set[int] = set()

        for index in selection:
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(options)
                and index not in seen
            ):
                cleaned.append(index)
                seen.add(index)

        if len(cleaned) > max_count:
            cleaned = cleaned[:max_count]

        return cleaned

    @staticmethod
    def _validate_selection(
        observation: dict[str, Any],
        selection: list[int],
    ) -> None:
        """
        Validate an agent selection before passing it to CG.

        This method is completely generic.

        It does not interpret:
            - Pokémon cards
            - attacks
            - abilities
            - energy
            - retreats
            - deck strategy

        It only validates the selection contract exposed by the
        CG observation.
        """

        if not isinstance(selection, list):
            raise TypeError(
                "Agent selection must be list[int]."
            )

        if not all(
            isinstance(index, int)
            for index in selection
        ):
            raise TypeError(
                "Agent selection must contain only integers."
            )

        select = observation.get("select")

        if not isinstance(select, dict):
            raise RuntimeError(
                "Engine requested an action but "
                "observation.select is invalid."
            )

        options = select.get(
            "option",
            [],
        )

        if not isinstance(options, list):
            raise RuntimeError(
                "Engine returned an invalid selection option list."
            )

        min_count = select.get(
            "minCount",
            0,
        )

        max_count = select.get(
            "maxCount",
            len(options),
        )

        try:
            min_count = max(
                0,
                int(min_count),
            )
        except Exception:
            min_count = 0

        try:
            max_count = max(
                0,
                int(max_count),
            )
        except Exception:
            max_count = len(options)

        max_count = min(
            max_count,
            len(options),
        )

        if len(selection) < min_count:
            raise ValueError(
                "Agent returned too few selections. "
                f"Required at least {min_count}, "
                f"received {len(selection)}."
            )

        if len(selection) > max_count:
            raise ValueError(
                "Agent returned too many selections. "
                f"Allowed at most {max_count}, "
                f"received {len(selection)}."
            )

        for index in selection:
            if index < 0 or index >= len(options):
                raise IndexError(
                    "Agent selected an invalid option index: "
                    f"{index}. "
                    f"Available options: {len(options)}."
                )

        if len(set(selection)) != len(selection):
            raise ValueError(
                "Agent returned duplicate selection indices."
            )

    @staticmethod
    def _extract_termination_reason(
        observation: dict[str, Any],
    ) -> int | None:
        """
        Extract the generic termination reason from CG result logs.

        CG LogType.RESULT has numeric type 23.

        The method intentionally does not interpret the meaning of
        the reason. It simply returns the integer provided by CG.
        """

        logs = observation.get(
            "logs",
            [],
        )

        for log in reversed(logs):
            if not isinstance(log, dict):
                continue

            if log.get("type") != 23:
                continue

            reason = log.get("reason")

            if isinstance(reason, int):
                return reason

        return None

    @staticmethod
    def _print_progress(
        observation: dict[str, Any],
        selections: int,
    ) -> None:
        """
        Print minimal engine-level match progress.

        No strategy-specific or Pokémon-specific information is
        interpreted here.
        """

        current = observation.get("current")

        if not isinstance(current, dict):
            return

        turn = current.get("turn")
        acting_player = current.get("yourIndex")
        result = current.get("result", -1)

        print(
            f"[Match] "
            f"turn={turn} "
            f"player={acting_player} "
            f"selection={selections} "
            f"result={result}"
        )