from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
# =====================================================================
# Paths
# =====================================================================
EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission"
CG_DIR = SUBMISSION_DIR / "cg"
RESULTS_DIR = EVALUATION_DIR / "results"
# =====================================================================
# Tournament Configuration
# =====================================================================
@dataclass(slots=True)
class TournamentConfig:
    """
    Configuration for a local generic tournament.
    The tournament runner does not know anything about the internal
    architecture of the evaluated agent.
    """
    games_per_opponent: int = 10
    max_games: int | None = None
    seed: int = 42
    save_results: bool = True
    results_dir: Path = field(
        default=RESULTS_DIR
    )
    verbose: bool = True
    stop_on_agent_error: bool = False
    # If True, alternate which player index is controlled by the
    # evaluated agent. This prevents first-player bias from dominating
    # the evaluation.
    alternate_agent_position: bool = True
    # Maximum number of selections allowed in a single match before
    # treating the match as stuck.
    max_selections_per_match: int = 10000

# =====================================================================
# Match Configuration
# =====================================================================
@dataclass(slots=True)
class MatchConfig:
    """
    Configuration for one match.
    """
    seed: int
    agent_player_index: int
    opponent_player_index: int
    max_selections: int = 10000
    
# =====================================================================
# Results
# =====================================================================
@dataclass(slots=True)
class MatchResult:
    """
    Result of one completed or failed match.
    """
    match_id: int
    agent_player_index: int
    opponent_name: str
    winner_player_index: int | None
    agent_won: bool
    opponent_won: bool
    draw: bool
    selections: int
    error: str | None = None
    termination_reason: int | None = None
    agent_deck_size: int = 60
    opponent_deck_size: int = 60

@dataclass(slots=True)
class OpponentResult:
    """
    Aggregate results against one opponent.
    """
    opponent_name: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    @property
    def loss_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.losses / self.games

    @property
    def draw_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.draws / self.games

@dataclass(slots=True)
class TournamentResult:
    """
    Aggregate results for the entire tournament.
    """
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    opponent_results: dict[
        str,
        OpponentResult,
    ] = field(
        default_factory=dict
    )
    matches: list[MatchResult] = field(
        default_factory=list
    )

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def loss_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.losses / self.total_games

    @property
    def draw_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games

    def record(
        self,
        result: MatchResult,
    ) -> None:
        """
        Add one match result to the aggregate tournament result.
        """
        self.matches.append(
            result
        )
        self.total_games += 1
        opponent_result = (
            self.opponent_results.setdefault(
                result.opponent_name,
                OpponentResult(
                    opponent_name=(
                        result.opponent_name
                    )
                ),
            )
        )
        opponent_result.games += 1
        if result.error is not None:
            self.errors += 1
            opponent_result.errors += 1
            return
        if result.agent_won:
            self.wins += 1
            opponent_result.wins += 1
        elif result.opponent_won:
            self.losses += 1
            opponent_result.losses += 1
        elif result.draw:
            self.draws += 1
            opponent_result.draws += 1