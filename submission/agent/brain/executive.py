"""
executive.py
Executive controller of the Brain architecture.
This module never performs search.
This module never evaluates board positions.
Its responsibility is ONLY to coordinate the various
Brain modules and merge their outputs into a single
high-level intent.
Current pipeline
BrainState
      │
      ▼
StrategicBrain
      │
      ▼
PredictiveBrain
      │
      ▼
TacticalBrain
      │
      ▼
ExecutiveBrain
      │
      ▼
DecisionIntent
Later phases will insert
AdaptiveBrain
OnlineLearningBrain
SupervisorBrain
without changing planner.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from .strategic import (
    StrategicBrain,
    StrategyState,
)
from .predictive import (
    PredictiveBrain,
    PredictionState,
)
from .tactical import (
    TacticalBrain,
)
# ---------------------------------------------------------
# Executive decision container
# ---------------------------------------------------------
@dataclass(slots=True)
class DecisionIntent:
    """
    High level recommendation exported to planner.
    These are NOT move scores.
    They are modifiers describing
    how planner should value actions.
    """
    aggression: float = 0.0
    setup: float = 0.0
    survival: float = 0.0
    tempo: float = 0.0
    resources: float = 0.0
    bench: float = 0.0
    gust: float = 0.0
    expected_pressure: float = 0.0
    board_risk: float = 0.0
    preferred_attacker: int | None = None
    preferred_target: int | None = None
    sacrifice_allowed: bool = False
    comeback_mode: bool = False
    prize_trade_target: int = 1

# ---------------------------------------------------------
# Executive Brain
# ---------------------------------------------------------
class ExecutiveBrain:
    """
    Central coordinator.
    Planner should only communicate
    with ExecutiveBrain.
    ExecutiveBrain communicates with
        StrategicBrain
        PredictiveBrain
        TacticalBrain
    and later
        AdaptiveBrain
        OnlineLearningBrain
        SupervisorBrain
    """
    def __init__(self):
        self.strategy = StrategicBrain()
        self.predictor = PredictiveBrain()
        self.tactical = TacticalBrain()

    # -----------------------------------------------------
    def analyse(
        self,
        brain_state,
    ) -> DecisionIntent:
        strategy = self.strategy.analyse(
            brain_state,
        )
        prediction = self.predictor.analyse(
            brain_state,
            strategy,
        )
        return self._combine(
            brain_state,
            strategy,
            prediction,
        )

    # -----------------------------------------------------
    def tactical_bonus(
        self,
        obs_dict,
        option,
        gamedata,
    ):
        return self.tactical.score(
            obs_dict,
            option,
            gamedata,
        )

    # -----------------------------------------------------
    def _combine(
        self,
        brain_state,
        strategy: StrategyState,
        prediction: PredictionState,
    ) -> DecisionIntent:
        intent = DecisionIntent()
        self._copy_strategy(
            strategy,
            intent,
        )
        self._copy_prediction(
            prediction,
            intent,
        )
        self._resolve_conflicts(
            brain_state,
            intent,
        )
        self._normalize(intent)
        return intent

    # -----------------------------------------------------
    def _copy_strategy(
        self,
        strategy,
        intent,
    ):
        intent.aggression = strategy.aggression
        intent.setup = strategy.setup_priority
        intent.survival = strategy.survival_priority
        intent.tempo = strategy.tempo_priority
        intent.resources = strategy.resource_priority
        intent.bench = strategy.bench_priority
        intent.gust = strategy.gust_priority
        intent.sacrifice_allowed = (
            strategy.sacrifice_allowed
        )
        intent.comeback_mode = (
            strategy.comeback_mode
        )
        intent.prize_trade_target = (
            strategy.prize_trade_target
        )
        intent.preferred_attacker = (
            strategy.preferred_attacker
        )
        intent.preferred_target = (
            strategy.preferred_target
        )

    # -----------------------------------------------------
    def _copy_prediction(
        self,
        prediction,
        intent,
    ):
        intent.expected_pressure = (
            prediction.expected_damage
        )
        intent.board_risk = (
            prediction.board_risk
        )
        if (
            intent.preferred_attacker
            is None
        ):
            intent.preferred_attacker = (
                prediction.predicted_attacker
            )
        if (
            intent.preferred_target
            is None
        ):
            intent.preferred_target = (
                prediction.predicted_target
            )

    # -----------------------------------------------------
    def _resolve_conflicts(
        self,
        state,
        intent,
    ):
        """
        Merge contradictory objectives.
        This is intentionally deterministic.
        No learned parameters here.
        """
        # ---------------------------------------------
        # Comeback mode overrides aggression.
        # ---------------------------------------------
        if intent.comeback_mode:
            if intent.board_risk > 0:
                intent.aggression *= (
                    1.0
                    - min(
                        1.0,
                        intent.board_risk,
                    )
                )
                intent.survival = max(
                    intent.survival,
                    intent.board_risk,
                )
        # ---------------------------------------------
        if (
            intent.expected_pressure > 0
            and intent.survival > 0
        ):
            intent.setup *= (
                1.0
                - min(
                    1.0,
                    intent.expected_pressure,
                )
            )
        # ---------------------------------------------
        if intent.resources > 0:
            intent.tempo *= (
                1.0
                - 0.5
                * intent.resources
            )
        # ---------------------------------------------
        if (
            intent.sacrifice_allowed
            and intent.aggression > 0
        ):
            intent.survival *= 0.5
        # ---------------------------------------------
        if intent.gust > 0:
            intent.aggression = max(
                intent.aggression,
                intent.gust,
            )
        # ---------------------------------------------
        if (
            state.phase == "endgame"
            and intent.prize_trade_target == 1
        ):
            intent.aggression = max(
                intent.aggression,
                intent.tempo,
            )
        # -----------------------------------------------------

    def _normalize(
        self,
        intent,
    ):
        """
        Keep every priority inside a stable range.
        No hardcoded gameplay values.
        Pure numerical normalization.
        """
        intent.aggression = self._clamp(
            intent.aggression
        )
        intent.setup = self._clamp(
            intent.setup
        )
        intent.survival = self._clamp(
            intent.survival
        )
        intent.tempo = self._clamp(
            intent.tempo
        )
        intent.resources = self._clamp(
            intent.resources
        )
        intent.bench = self._clamp(
            intent.bench
        )
        intent.gust = self._clamp(
            intent.gust
        )
        intent.expected_pressure = max(
            0.0,
            intent.expected_pressure,
        )
        intent.board_risk = max(
            0.0,
            intent.board_risk,
        )

    # -----------------------------------------------------
    @staticmethod
    def _clamp(
        value,
    ):
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    # -----------------------------------------------------
    def planner_bias(
        self,
        intent: DecisionIntent,
    ):
        """
        Convert the Executive decision into planner-facing
        additive biases.
        Planner remains the owner of search.
        Executive merely supplies guidance.
        """
        return PlannerHints(
            aggression=intent.aggression,
            setup=intent.setup,
            survival=intent.survival,
            tempo=intent.tempo,
            resources=intent.resources,
            bench=intent.bench,
            gust=intent.gust,
            expected_pressure=intent.expected_pressure,
            board_risk=intent.board_risk,
            preferred_attacker=intent.preferred_attacker,
            preferred_target=intent.preferred_target,
            sacrifice_allowed=intent.sacrifice_allowed,
            comeback_mode=intent.comeback_mode,
            prize_trade_target=intent.prize_trade_target,
        )

# ---------------------------------------------------------
# Planner-facing immutable hints
# ---------------------------------------------------------
@dataclass(slots=True)
class PlannerHints:
    aggression: float
    setup: float
    survival: float
    tempo: float
    resources: float
    bench: float
    gust: float
    expected_pressure: float
    board_risk: float
    preferred_attacker: int | None
    preferred_target: int | None
    sacrifice_allowed: bool
    comeback_mode: bool
    prize_trade_target: int

# ---------------------------------------------------------
# Future expansion hooks
# ---------------------------------------------------------
class ExecutiveExtensions:
    """
    Placeholder interface.
    Later phases will plug modules here without requiring
    planner.py to change.
        AdaptiveBrain
        OnlineLearningBrain
        SupervisorBrain
    ExecutiveBrain will simply call them if present.
    """
    def before_planner(
        self,
        brain_state,
        intent,
    ):
        return intent
    
    def after_planner(
        self,
        brain_state,
        planner_result,
    ):
        return planner_result

# ---------------------------------------------------------
# Optional helper
# ---------------------------------------------------------
def build_intent(
    executive: ExecutiveBrain,
    brain_state,
):
    return executive.analyse(
        brain_state,
    )
    
# ---------------------------------------------------------
# Public exports
# ---------------------------------------------------------
__all__ = (
    "DecisionIntent",
    "PlannerHints",
    "ExecutiveBrain",
    "ExecutiveExtensions",
    "build_intent",
)