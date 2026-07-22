from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
from .agent_adapter import AgentAdapter
from .config import (
    MatchConfig,
    MatchResult,
    OpponentResult,
    TournamentConfig,
    TournamentResult,
)
from .match_runner import MatchOutcome
from .match_runner import MatchRunner
from .opponent_pool import Opponent
from .opponent_pool import OpponentPool
from .opponent_pool import OpponentScheduler
# =====================================================================
# Tournament
# =====================================================================
class Tournament:
    """
    Generic local tournament harness.
    The tournament treats:
        - the evaluated agent
        - every opponent
        - every deck
    as external inputs.
    It does not inspect the internals of the evaluated agent.
    Conceptually:
        Candidate Agent
              |
              v
        +-------------+
        | MatchRunner |
        +-------------+
              |
              v
        Opponent Pool
        /     |     \
       A      B      C
    Every opponent is independently evaluated.
    """
    def __init__(
        self,
        candidate: AgentAdapter,
        opponent_pool: OpponentPool,
        config: TournamentConfig | None = None,
    ):
        if not isinstance(
            candidate,
            AgentAdapter,
        ):
            raise TypeError(
                "candidate must be an AgentAdapter."
            )
        if not isinstance(
            opponent_pool,
            OpponentPool,
        ):
            raise TypeError(
                "opponent_pool must be an OpponentPool."
            )
        if len(
            opponent_pool
        ) == 0:
            raise ValueError(
                "Opponent pool cannot be empty."
            )
        self.candidate = candidate
        self.opponent_pool = (
            opponent_pool
        )
        self.config = (
            config
            if config is not None
            else TournamentConfig()
        )
        self.scheduler = (
            OpponentScheduler(
                opponent_pool,
                seed=self.config.seed,
            )
        )
        self.match_runner = (
            MatchRunner(
                max_selections=(
                    self.config
                    .max_selections_per_match
                ),
                verbose=False,
            )
        )
        self.rng = random.Random(
            self.config.seed
        )

    # =================================================================
    # Run Tournament
    # =================================================================
    def run(
        self,
    ) -> TournamentResult:
        """
        Execute the complete tournament.
        The candidate is repeatedly exposed to opponents from the pool.
        If alternate_agent_position is enabled, the candidate alternates
        between player 0 and player 1.
        The resulting evaluation is therefore less sensitive to
        first-player advantage.
        """
        result = TournamentResult()
        schedule = (
            self.scheduler.round_robin(
                games_per_opponent=(
                    self.config
                    .games_per_opponent
                )
            )
        )
        if self.config.max_games is not None:
            schedule = schedule[
                : self.config.max_games
            ]
        if not schedule:
            raise RuntimeError(
                "Tournament schedule is empty."
            )
        if self.config.verbose:
            self._print_header(
                len(schedule)
            )
        for match_id, opponent in enumerate(
            schedule,
            start=1,
        ):
            if self.config.alternate_agent_position:
                agent_player_index = (
                    (match_id - 1) % 2
                )
            else:
                agent_player_index = 0
            opponent_player_index = (
                1
                - agent_player_index
            )
            match_config = MatchConfig(
                seed=(
                    self.config.seed
                    + match_id
                ),
                agent_player_index=(
                    agent_player_index
                ),
                opponent_player_index=(
                    opponent_player_index
                ),
                max_selections=(
                    self.config
                    .max_selections_per_match
                ),
            )
            if self.config.verbose:
                self._print_match_start(
                    match_id,
                    len(schedule),
                    opponent,
                    match_config,
                )
            match_result = (
                self._run_match(
                    match_id=match_id,
                    opponent=opponent,
                    match_config=match_config,
                )
            )
            result.record(
                match_result
            )
            if self.config.verbose:
                self._print_match_result(
                    match_result
                )
            if (
                match_result.error is not None
                and self.config.stop_on_agent_error
            ):
                raise RuntimeError(
                    "Tournament stopped because "
                    "a match produced an error."
                )
        if self.config.verbose:
            self._print_summary(
                result
            )
        if self.config.save_results:
            self._save_results(
                result
            )
        return result

    # =================================================================
    # Match Execution
    # =================================================================
    def _run_match(
        self,
        match_id: int,
        opponent: Opponent,
        match_config: MatchConfig,
    ) -> MatchResult:
        """
        Execute one candidate-vs-opponent match.
        """
        if match_config.agent_player_index == 0:
            deck0 = self._candidate_deck()
            deck1 = opponent.deck
            player0 = (
                self.candidate.choose
            )
            player1 = (
                opponent.choose
            )
        else:
            deck0 = opponent.deck
            deck1 = self._candidate_deck()
            player0 = (
                opponent.choose
            )
            player1 = (
                self.candidate.choose
            )
        try:
            outcome = (
                self.match_runner.run(
                    deck0=deck0,
                    deck1=deck1,
                    player0=player0,
                    player1=player1,
                )
            )
            return self._build_match_result(
                match_id=match_id,
                opponent=opponent,
                agent_player_index=(
                    match_config
                    .agent_player_index
                ),
                outcome=outcome,
            )
        except Exception as exc:
            return MatchResult(
                match_id=match_id,
                agent_player_index=(
                    match_config
                    .agent_player_index
                ),
                opponent_name=(
                    opponent.name
                ),
                winner_player_index=None,
                agent_won=False,
                opponent_won=False,
                draw=False,
                selections=0,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
            
    # =================================================================
    # Match Result
    # =================================================================
    @staticmethod
    def _build_match_result(
        match_id: int,
        opponent: Opponent,
        agent_player_index: int,
        outcome: MatchOutcome,
    ) -> MatchResult:
        """
        Convert a generic MatchOutcome into a MatchResult.
        """
        winner = (
            outcome
            .winner_player_index
        )
        agent_won = (
            winner is not None
            and winner
            == agent_player_index
        )
        opponent_won = (
            winner is not None
            and winner
            != agent_player_index
        )
        draw = (
            winner is None
            and outcome.termination_reason
            == 3
        )
        return MatchResult(
            match_id=match_id,
            agent_player_index=(
                agent_player_index
            ),
            opponent_name=(
                opponent.name
            ),
            winner_player_index=winner,
            agent_won=agent_won,
            opponent_won=opponent_won,
            draw=draw,
            selections=(
                outcome.selections
            ),
            termination_reason=(
                outcome.termination_reason
            ),
        )

    # =================================================================
    # Candidate Deck
    # =================================================================
    def _candidate_deck(
        self,
    ) -> list[int]:
        """
        Return the evaluated candidate's deck.
        This method is intentionally isolated because the tournament
        harness should not know how the candidate deck is configured.
        Override this method or subclass Tournament to provide the
        candidate deck.
        """
        raise NotImplementedError(
            "Tournament._candidate_deck() must be implemented "
            "by the concrete tournament runner."
        )

    # =================================================================
    # Output
    # =================================================================
    def _save_results(
        self,
        result: TournamentResult,
    ) -> Path:
        """
        Save tournament results as JSON.
        """
        output_dir = Path(
            self.config.results_dir
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path = (
            output_dir
            / "tournament_results.json"
        )
        payload = {
            "candidate": (
                self.candidate.name
            ),
            "total_games": (
                result.total_games
            ),
            "wins": (
                result.wins
            ),
            "losses": (
                result.losses
            ),
            "draws": (
                result.draws
            ),
            "errors": (
                result.errors
            ),
            "win_rate": (
                result.win_rate
            ),
            "loss_rate": (
                result.loss_rate
            ),
            "draw_rate": (
                result.draw_rate
            ),
            "opponents": {
                name: {
                    "games": (
                        opponent.games
                    ),
                    "wins": (
                        opponent.wins
                    ),
                    "losses": (
                        opponent.losses
                    ),
                    "draws": (
                        opponent.draws
                    ),
                    "errors": (
                        opponent.errors
                    ),
                    "win_rate": (
                        opponent.win_rate
                    ),
                    "loss_rate": (
                        opponent.loss_rate
                    ),
                    "draw_rate": (
                        opponent.draw_rate
                    ),
                }
                for name, opponent
                in result.opponent_results.items()
            },
            "matches": [
                {
                    "match_id": (
                        match.match_id
                    ),
                    "agent_player_index": (
                        match.agent_player_index
                    ),
                    "opponent_name": (
                        match.opponent_name
                    ),
                    "winner_player_index": (
                        match.winner_player_index
                    ),
                    "agent_won": (
                        match.agent_won
                    ),
                    "opponent_won": (
                        match.opponent_won
                    ),
                    "draw": (
                        match.draw
                    ),
                    "selections": (
                        match.selections
                    ),
                    "error": (
                        match.error
                    ),
                    "termination_reason": (
                        match.termination_reason
                    ),
                }
                for match in result.matches
            ],
        }
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
            )
        if self.config.verbose:
            print()
            print(
                "Results saved to:"
            )
            print(
                output_path
            )
        return output_path

    # =================================================================
    # Console Output
    # =================================================================
    @staticmethod
    def _print_header(
        total_matches: int,
    ) -> None:
        print()
        print(
            "=" * 70
        )
        print(
            "GENERIC LOCAL TOURNAMENT"
        )
        print(
            "=" * 70
        )
        print()
        print(
            "Scheduled matches:",
            total_matches,
        )
        print()

    @staticmethod
    def _print_match_start(
        match_id: int,
        total_matches: int,
        opponent: Opponent,
        config: MatchConfig,
    ) -> None:
        print(
            f"[{match_id}/{total_matches}] "
            f"{opponent.name} | "
            f"Candidate=P{config.agent_player_index}"
        )

    @staticmethod
    def _print_match_result(
        result: MatchResult,
    ) -> None:
        if result.error is not None:
            print(
                "  ERROR:",
                result.error,
            )
            return
        if result.agent_won:
            outcome = "WIN"
        elif result.opponent_won:
            outcome = "LOSS"
        elif result.draw:
            outcome = "DRAW"
        else:
            outcome = "UNKNOWN"
        print(
            f"  Result: {outcome} | "
            f"Selections: {result.selections}"
        )

    @staticmethod
    def _print_summary(
        result: TournamentResult,
    ) -> None:
        print()
        print(
            "=" * 70
        )
        print(
            "TOURNAMENT SUMMARY"
        )
        print(
            "=" * 70
        )
        print()

        print(
            "Games  :",
            result.total_games,
        )
        print(
            "Wins   :",
            result.wins,
        )
        print(
            "Losses :",
            result.losses,
        )
        print(
            "Draws  :",
            result.draws,
        )
        print(
            "Errors :",
            result.errors,
        )
        print()
        print(
            f"Win Rate : "
            f"{result.win_rate * 100:.2f}%"
        )
        print(
            f"Loss Rate : "
            f"{result.loss_rate * 100:.2f}%"
        )
        print()
        print(
            "Per Opponent"
        )
        print(
            "-" * 70
        )
        for (
            name,
            opponent,
        ) in result.opponent_results.items():
            print(
                f"{name:30s} "
                f"Games={opponent.games:4d} "
                f"W={opponent.wins:4d} "
                f"L={opponent.losses:4d} "
                f"D={opponent.draws:4d} "
                f"WinRate={opponent.win_rate * 100:6.2f}%"
            )
        print()