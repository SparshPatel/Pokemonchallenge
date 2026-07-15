"""
predictive.py
Predicts the opponent's most likely responses.
Responsibilities
----------------
• Estimate immediate threats
• Predict likely attacker
• Predict likely target
• Estimate board risk
• Estimate next-turn pressure
Contains NO search.
Contains NO engine calls.
Contains NO evaluation.
Produces PredictionState.
"""
from __future__ import annotations
from dataclasses import dataclass
# ---------------------------------------------------------
# Prediction state
# ---------------------------------------------------------
@dataclass(slots=True)
class PredictionState:
    expected_damage: float = 0.0
    knockout_probability: float = 0.0
    gust_probability: float = 0.0
    supporter_probability: float = 0.0
    energy_attachment_probability: float = 0.0
    setup_probability: float = 0.0
    board_risk: float = 0.0
    active_survival_probability: float = 1.0
    expected_prize_trade: int = 0
    predicted_attacker: int | None = None
    predicted_target: int | None = None

# ---------------------------------------------------------
# Predictive Brain
# ---------------------------------------------------------
class PredictiveBrain:
    """
    Stateless prediction layer.
    Given the current BrainState and StrategyState,
    estimates what the opponent is most likely to do.
    This is NOT search.
    This is NOT simulation.
    It simply predicts.
    """
    def analyse(
        self,
        brain_state,
        strategy_state,
    ) -> PredictionState:
        prediction = PredictionState()
        self._predict_attacker(
            brain_state,
            prediction,
        )
        self._predict_target(
            brain_state,
            prediction,
        )
        self._predict_pressure(
            brain_state,
            strategy_state,
            prediction,
        )
        self._predict_resources(
            brain_state,
            prediction,
        )
        self._predict_board_risk(
            brain_state,
            prediction,
        )
        return prediction

    # -----------------------------------------------------
    def _predict_attacker(
        self,
        state,
        prediction,
    ):
        best = None
        best_energy = -1
        candidates = []
        if state.opponent.active is not None:
            candidates.append(
                state.opponent.active
            )
        candidates.extend(
            state.opponent.bench
        )
        for pokemon in candidates:
            if pokemon is None:
                continue
            if pokemon.energies > best_energy:
                best_energy = pokemon.energies
                best = pokemon.id
        prediction.predicted_attacker = best

    # -----------------------------------------------------
    def _predict_target(
        self,
        state,
        prediction,
    ):
        if state.me.active is not None:
            prediction.predicted_target = (
                state.me.active.id
            )

    # -----------------------------------------------------
    def _predict_pressure(
        self,
        state,
        strategy,
        prediction,
    ):
        attackers = 0
        if (
            state.opponent.active
            and state.opponent.active.can_attack
        ):
            attackers += 1
        attackers += sum(
            pokemon.can_attack
            for pokemon in state.opponent.bench
        )
        prediction.expected_damage = float(
            attackers
        )
        if attackers == 0:
            prediction.active_survival_probability = 1.0
        else:
            prediction.active_survival_probability = (
                1.0 / (attackers + 1)
            )
        prediction.knockout_probability = min(
            1.0,
            attackers / 3.0,
        )

    # -----------------------------------------------------
    def _predict_resources(
        self,
        state,
        prediction,
    ):
        if state.opponent.hand_count > 0:
            prediction.supporter_probability = 1.0
        if state.opponent.deck_count > 0:
            prediction.energy_attachment_probability = 1.0
        if len(state.opponent.bench) < 5:
            prediction.setup_probability = 1.0

    # -----------------------------------------------------
    def _predict_board_risk(
        self,
        state,
        prediction,
    ):
        risk = 0.0
        if state.me.active is not None:
            if not state.me.active.can_attack:
                risk += 1.0
        if len(state.me.bench) == 0:
            risk += 1.0
        if state.losing:
            risk += 1.0
        prediction.board_risk = risk
        prediction.expected_prize_trade = (
            state.me.prizes_remaining
            - state.opponent.prizes_remaining
        )