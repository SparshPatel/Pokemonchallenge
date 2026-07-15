"""
strategic.py
High-level strategic reasoning.
Responsibilities
----------------
• Determine current game plan.
• Estimate whether we are ahead or behind.
• Choose strategic priorities.
Contains NO search.
Contains NO engine calls.
Contains NO action scoring.
Produces a StrategyState consumed by:
    PredictiveBrain
    AdaptiveBrain
    SupervisorBrain
"""
from __future__ import annotations
from dataclasses import dataclass
# ----------------------------------------------------------------------
# Strategy state
# ----------------------------------------------------------------------
@dataclass(slots=True)
class StrategyState:
    aggression: float = 0.0
    setup_priority: float = 0.0
    survival_priority: float = 0.0
    tempo_priority: float = 0.0
    resource_priority: float = 0.0
    bench_priority: float = 0.0
    gust_priority: float = 0.0
    sacrifice_allowed: bool = False
    comeback_mode: bool = False
    prize_trade_target: int = 1
    preferred_attacker: int | None = None
    preferred_target: int | None = None

# ----------------------------------------------------------------------
# Strategic Brain
# ----------------------------------------------------------------------
class StrategicBrain:
    """
    Determines the long-term plan.
    Never evaluates moves.
    Never performs search.
    Never modifies the board.
    It only determines
        "What kind of game should we play?"
    """
    def analyse(
        self,
        brain_state,
    ) -> StrategyState:
        strategy = StrategyState()
        self._prize_state(
            brain_state,
            strategy,
        )
        self._board_state(
            brain_state,
            strategy,
        )
        self._resource_state(
            brain_state,
            strategy,
        )
        self._tempo_state(
            brain_state,
            strategy,
        )
        self._late_game(
            brain_state,
            strategy,
        )
        return strategy

    # ---------------------------------------------------------
    def _prize_state(
        self,
        state,
        strategy,
    ):
        my_prizes = state.me.prizes_remaining
        opp_prizes = state.opponent.prizes_remaining
        difference = my_prizes - opp_prizes
        if difference > 1:
            strategy.comeback_mode = True
            strategy.setup_priority = 1.0
            strategy.survival_priority = 1.0
            strategy.aggression = 0.0
            strategy.prize_trade_target = 2
        elif difference < -1:
            strategy.aggression = 1.0
            strategy.tempo_priority = 1.0
            strategy.prize_trade_target = 1
        else:
            strategy.tempo_priority = 1.0

    # ---------------------------------------------------------
    def _board_state(
        self,
        state,
        strategy,
    ):
        my_attackers = sum(
            p.can_attack
            for p in state.me.bench
        )
        if state.me.active is not None:
            my_attackers += int(
                state.me.active.can_attack
            )
        if my_attackers <= 1:
            strategy.bench_priority = 1.0
            strategy.setup_priority = max(
                strategy.setup_priority,
                1.0,
            )

    # ---------------------------------------------------------
    def _resource_state(
        self,
        state,
        strategy,
    ):
        if state.me.deck_count < 10:
            strategy.resource_priority = 1.0
        if state.me.hand_count <= 2:
            strategy.resource_priority = max(
                strategy.resource_priority,
                1.0,
            )

    # ---------------------------------------------------------
    def _tempo_state(
        self,
        state,
        strategy,
    ):
        if state.turn_player == 0:
            strategy.tempo_priority = max(
                strategy.tempo_priority,
                0.5,
            )

    # ---------------------------------------------------------
    def _late_game(
        self,
        state,
        strategy,
    ):
        if state.phase != "endgame":
            return
        strategy.gust_priority = 1.0
        strategy.aggression = max(
            strategy.aggression,
            1.0,
        )
        strategy.sacrifice_allowed = True
        strategy.prize_trade_target = 1