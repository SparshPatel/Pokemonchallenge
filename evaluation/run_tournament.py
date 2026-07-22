from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .agent_adapter import AgentAdapter
from .config import TournamentConfig
from .opponent_pool import OpponentPool
from .tournament import Tournament


# =====================================================================
# Paths
# =====================================================================

EVALUATION_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    EVALUATION_DIR.parent
)

SUBMISSION_DIR = (
    PROJECT_ROOT
    / "submission"
)

DEFAULT_REPLAYS_DIR = (
    PROJECT_ROOT
    / "replays"
)

DEFAULT_RESULTS_DIR = (
    EVALUATION_DIR
    / "results"
)


# =====================================================================
# Candidate Agent Loading
# =====================================================================

def load_candidate_agent():
    """
    Load the candidate agent from the actual Kaggle submission entrypoint.

    The submission is expected to expose:

        submission.main.agent

    Returns:
        Callable candidate agent.
    """

    print(
        "Loading candidate agent..."
    )

    try:
        from submission.main import agent

    except Exception as exc:
        raise RuntimeError(
            "Unable to import candidate agent from "
            "submission.main.agent"
        ) from exc

    if not callable(
        agent
    ):
        raise RuntimeError(
            "submission.main.agent exists but is not callable."
        )

    print(
        "Candidate agent loaded:"
    )

    print(
        "  module   : submission.main"
    )

    print(
        "  callable : agent"
    )

    return agent


# =====================================================================
# Candidate Deck Loading
# =====================================================================

def load_candidate_deck():
    """
    Load the candidate deck using the same deck parser used by the
    Kaggle submission.

    The current submission exposes:

        submission.main.load_deck(path)

    Therefore the evaluation harness locates the submission deck file
    and passes its path to the submission's own loader.
    """

    global candidate_deck

    print(
        "Loading candidate deck..."
    )

    try:
        from submission.main import load_deck

    except Exception as exc:
        raise RuntimeError(
            "Unable to import submission.main.load_deck()."
        ) from exc

    submission_dir = (
        PROJECT_ROOT
        / "submission"
    )

    candidate_deck_path = None

    # -------------------------------------------------------------
    # Standard deck locations.
    # -------------------------------------------------------------

    possible_paths = [
        submission_dir / "deck.txt",
        submission_dir / "deck.json",
        submission_dir / "deck.csv",
        submission_dir / "deck",
        submission_dir / "data" / "deck.txt",
        submission_dir / "data" / "deck.json",
        submission_dir / "decks" / "deck.txt",
        submission_dir / "decks" / "deck.json",
    ]

    for path in possible_paths:

        if (
            path.exists()
            and path.is_file()
        ):
            candidate_deck_path = path
            break

    # -------------------------------------------------------------
    # Fallback search.
    # -------------------------------------------------------------

    if candidate_deck_path is None:

        candidates = []

        for pattern in (
            "*.txt",
            "*.json",
            "*.csv",
        ):

            candidates.extend(
                p
                for p in submission_dir.rglob(
                    pattern
                )
                if p.is_file()
            )

        ignored_names = {
            "requirements.txt",
            "config.json",
            "metadata.json",
        }

        candidates = [
            p
            for p in candidates
            if p.name.lower()
            not in ignored_names
        ]

        if len(
            candidates
        ) == 1:

            candidate_deck_path = (
                candidates[0]
            )

    # -------------------------------------------------------------
    # Nothing found.
    # -------------------------------------------------------------

    if candidate_deck_path is None:

        raise RuntimeError(
            "Unable to locate candidate deck file.\n\n"
            "submission.main.load_deck(path) requires a deck file, "
            "but the evaluation harness could not automatically "
            "identify one under:\n"
            f"  {submission_dir}\n\n"
            "Please specify the actual deck file path in "
            "evaluation/config.py."
        )

    # -------------------------------------------------------------
    # Load deck using submission's own parser.
    # -------------------------------------------------------------

    try:

        deck = load_deck(
            str(
                candidate_deck_path
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load candidate deck using "
            "submission.main.load_deck().\n\n"
            f"Deck path: {candidate_deck_path}"
        ) from exc

    # -------------------------------------------------------------
    # Validate deck.
    # -------------------------------------------------------------

    if not isinstance(
        deck,
        list,
    ):

        raise TypeError(
            "Candidate deck loader did not return list[int]. "
            f"Received: {type(deck).__name__}"
        )

    if len(
        deck
    ) != 60:

        raise ValueError(
            "Candidate deck must contain exactly 60 cards. "
            f"Received {len(deck)} cards.\n\n"
            f"Deck path: {candidate_deck_path}"
        )

    if not all(
        isinstance(
            card_id,
            int,
        )
        for card_id in deck
    ):

        raise TypeError(
            "Candidate deck must contain only integer card IDs."
        )

    candidate_deck = list(
        deck
    )

    print(
        "Candidate deck loaded:"
    )

    print(
        f"  path  : {candidate_deck_path}"
    )

    print(
        f"  cards : {len(candidate_deck)}"
    )

    return candidate_deck


# =====================================================================
# Generic Deck Validation
# =====================================================================

def validate_deck(
    deck: Any,
    source: str,
) -> list[int]:
    """
    Validate and normalize a deck.
    """

    if not isinstance(
        deck,
        (
            list,
            tuple,
        ),
    ):

        raise TypeError(
            f"{source} must return a list or tuple."
        )

    deck = [
        int(
            card_id
        )
        for card_id in deck
    ]

    if len(
        deck
    ) != 60:

        raise ValueError(
            f"{source} contains "
            f"{len(deck)} cards. "
            "A Pokémon TCG deck must contain exactly 60 cards."
        )

    return deck


# =====================================================================
# Opponent Pool
# =====================================================================

def build_opponent_pool(
    replays_dir: Path,
) -> OpponentPool:
    """
    Build the generic opponent pool from replay data.

    The opponent pool is responsible for:

        replay ZIP / JSON files
                ↓
        replay extraction
                ↓
        opponent deck reconstruction
                ↓
        opponent action replay

    The tournament itself remains unaware of replay storage.
    """

    if not replays_dir.exists():

        raise FileNotFoundError(
            "Replay directory does not exist: "
            f"{replays_dir}"
        )

    if not replays_dir.is_dir():

        raise NotADirectoryError(
            "Replay path is not a directory: "
            f"{replays_dir}"
        )

    return OpponentPool.from_replays(
        replays_dir
    )


# =====================================================================
# Tournament Factory
# =====================================================================

class CandidateTournament(
    Tournament
):
    """
    Concrete tournament that supplies the candidate deck.

    The candidate itself is always an AgentAdapter.
    """

    def __init__(
        self,
        candidate: AgentAdapter,
        opponent_pool: OpponentPool,
        candidate_deck: list[int],
        config: TournamentConfig,
    ):

        super().__init__(
            candidate=candidate,
            opponent_pool=opponent_pool,
            config=config,
        )

        self._candidate_deck_ids = list(
            candidate_deck
        )

    def _candidate_deck(
        self,
    ) -> list[int]:

        return list(
            self._candidate_deck_ids
        )


# =====================================================================
# CLI
# =====================================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the generic local "
            "Kaggle-style Pokémon tournament "
            "evaluation pipeline."
        )
    )

    parser.add_argument(
        "--replays",
        type=Path,
        default=DEFAULT_REPLAYS_DIR,
        help=(
            "Directory containing replay ZIP archives "
            "and replay JSON files."
        ),
    )

    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=1,
        help=(
            "Number of games played against "
            "each discovered opponent."
        ),
    )

    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help=(
            "Optional maximum number of matches "
            "to run."
        ),
    )

    parser.add_argument(
        "--max-selections",
        type=int,
        default=10000,
        help=(
            "Maximum number of engine selections "
            "allowed per match."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used by the tournament "
            "scheduler."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory in which tournament results "
            "are written."
        ),
    )

    parser.add_argument(
        "--no-alternate-position",
        action="store_true",
        help=(
            "Do not alternate the candidate between "
            "player 0 and player 1."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop immediately when a match produces "
            "an error."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress per-match tournament output."
        ),
    )

    return parser


# =====================================================================
# Main
# =====================================================================

def main() -> int:

    parser = (
        build_argument_parser()
    )

    args = (
        parser.parse_args()
    )

    # -------------------------------------------------------------
    # Validate CLI configuration.
    # -------------------------------------------------------------

    if (
        args.games_per_opponent
        <= 0
    ):

        parser.error(
            "--games-per-opponent must be > 0."
        )

    if (
        args.max_games is not None
        and args.max_games <= 0
    ):

        parser.error(
            "--max-games must be > 0."
        )

    if (
        args.max_selections
        <= 0
    ):

        parser.error(
            "--max-selections must be > 0."
        )

    # -------------------------------------------------------------
    # Make project root importable.
    # -------------------------------------------------------------

    project_root = str(
        PROJECT_ROOT
    )

    if project_root not in sys.path:

        sys.path.insert(
            0,
            project_root,
        )

    # -------------------------------------------------------------
    # Load candidate agent.
    # -------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "LOADING CANDIDATE"
    )

    print(
        "=" * 70
    )

    print()

    candidate = (
        load_candidate_agent()
    )

    candidate_name = getattr(
        candidate,
        "name",
        getattr(
            candidate,
            "__name__",
            "candidate",
        ),
    )

    candidate_adapter = AgentAdapter(
        name=candidate_name,
        agent_fn=candidate,
    )

    print(
        "Candidate:",
        candidate_name,
    )

    # -------------------------------------------------------------
    # Wrap candidate in AgentAdapter.
    #
    # IMPORTANT:
    # Tournament requires AgentAdapter, not raw callable.
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Load candidate deck.
    # -------------------------------------------------------------

    candidate_deck = (
        load_candidate_deck()
    )

    print(
        "Candidate deck cards:",
        len(
            candidate_deck
        ),
    )

    # -------------------------------------------------------------
    # Build opponent pool.
    # -------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "BUILDING OPPONENT POOL"
    )

    print(
        "=" * 70
    )

    print()

    opponent_pool = (
        build_opponent_pool(
            args.replays
        )
    )

    print(
        "Opponents discovered:",
        len(
            opponent_pool
        ),
    )

    if len(
        opponent_pool
    ) == 0:

        raise RuntimeError(
            "No opponents were discovered "
            "from the replay directory."
        )

    # -------------------------------------------------------------
    # Tournament configuration.
    # -------------------------------------------------------------

    config = TournamentConfig(

        games_per_opponent=(
            args.games_per_opponent
        ),

        max_games=(
            args.max_games
        ),

        max_selections_per_match=(
            args.max_selections
        ),

        seed=(
            args.seed
        ),

        alternate_agent_position=(
            not args.no_alternate_position
        ),

        stop_on_agent_error=(
            args.stop_on_error
        ),

        verbose=(
            not args.quiet
        ),

        save_results=True,

        results_dir=(
            args.results_dir
        ),
    )

    # -------------------------------------------------------------
    # Create tournament.
    # -------------------------------------------------------------

    tournament = CandidateTournament(

        candidate=(
            candidate_adapter
        ),

        opponent_pool=(
            opponent_pool
        ),

        candidate_deck=(
            candidate_deck
        ),

        config=(
            config
        ),
    )

    # -------------------------------------------------------------
    # Run tournament.
    # -------------------------------------------------------------

    result = (
        tournament.run()
    )

    # -------------------------------------------------------------
    # Final exit status.
    # -------------------------------------------------------------

    if result.errors > 0:

        return 2

    if result.total_games == 0:

        return 3

    return 0


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )