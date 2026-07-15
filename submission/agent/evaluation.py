"""
evaluation.py
Dynamic evaluation framework for the Pokemon Challenge agent.
This module provides the deterministic fallback evaluator used when
ValueNet is unavailable. Unlike ValueNet, which predicts win
probability from learned weights, this evaluator generates heuristic
weights dynamically from the current game context.
Pipeline
Board
    ↓
value_net.extract_features(...)
    ↓
feature vector
    ↓
EvaluationContext
    ↓
DynamicWeightGenerator
    ↓
Evaluator
    ↓
scalar score
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass(slots=True)
class EvaluationContext:
    """
    Information about the current search context.
    None of these values are board features.
    They describe HOW the board should be evaluated.
    """
    game_phase: float
    prize_diff: float
    search_depth: int
    search_confidence: float
    opponent_embedding: Optional[Tuple[float, float, float]] = None
    
@dataclass(slots=True)
class EvaluationResult:
    """
    Output of the heuristic evaluator.
    """
    score: float
    weights: Dict[str, float]
    feature_scores: Dict[str, float]
    uncertainty: float = 0.0
    
class DynamicWeightGenerator:
    """
    Generates the effective heuristic weights used by the fallback evaluator.
    The effective weights are composed of two parts:
        effective = offline + alpha * online
    offline:
        Learned offline from large-scale self-play.
        Loaded once before the tournament.
    online:
        Adapted only during the current match.
        Starts at zero every game.
    The online component is never written back into the offline model.
    """
    def __init__(
        self,
        offline_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.15,
    ):
        self.offline_weights = dict(offline_weights or {})
        self.online_weights: Dict[str, float] = {}
        self.runtime_weights: Optional[Dict[str, float]] = None

    def reset_match(self):
        """
        Forget everything learned during the current match.
        """
        self.online_weights.clear()

    def get_weight(
        self,
        feature: str,
    ) -> float:
        offline = self.offline_weights.get(feature, 0.0)
        online = self.online_weights.get(feature, 0.0)
        return offline + self.alpha * online

    def generate(
        self,
        features: Dict[str, float],
        context: EvaluationContext,
    ) -> Dict[str, float]:
        if self.runtime_weights is not None:
            weights = dict(
                self.runtime_weights
            )
        else:
            weights = {}
            for feature in features:
                weights[feature] = self.get_weight(
                    feature
                )
        # Context modulation.
        phase = max(0.0, min(1.0, context.game_phase))
        if "prize_diff" in weights:
            weights["prize_diff"] *= (1.0 + phase)
        if "setup_ko" in weights:
            weights["setup_ko"] *= (1.0 + phase)
        if "bench_ready_frac" in weights:
            weights["bench_ready_frac"] *= (2.0 - phase)
        if "bench_frac" in weights:
            weights["bench_frac"] *= (2.0 - phase)
        if context.opponent_embedding is not None:
            attack_rate, ability_rate, item_rate = context.opponent_embedding
            if attack_rate > 0.55:
                if "my_active_hpfrac" in weights:
                    weights["my_active_hpfrac"] *= 1.15
                if "opp_threat" in weights:
                    weights["opp_threat"] *= 1.20
            if ability_rate > 0.35:
                if "setup_ko" in weights:
                    weights["setup_ko"] *= 1.10
            if item_rate > 0.40:
                if "bench_ready_frac" in weights:
                    weights["bench_ready_frac"] *= 1.10
        return weights
    
    def set_runtime_weights(
        self,
        weights,
    ):
        """
        Runtime weights predicted by WeightModel.
        Passing None disables runtime weighting.
        """
        if weights is None:
            self.runtime_weights = None
            return
        feature_order = (
            "prize_diff",
            "my_active_hpfrac",
            "opp_active_hpfrac",
            "board_control",
            "energy_advantage",
            "attack_ready",
            "tempo",
            "setup_ko",
            "hand_size",
            "bench_frac",
            "bench_ready_frac",
            "supporter_value",
        )
        self.runtime_weights = {
            name: float(value)
            for name, value in zip(
                feature_order,
                weights,
            )
        }
    
from collections import deque
class MatchLearner:
    """
    Learns temporary match-specific corrections using temporal-difference style
    updates.
    Offline weights are never modified.
    During the match we store every evaluated state together with the predicted
    value. At the end of the game we compute the prediction error and update only
    the temporary online weights.
    The online weights are discarded when the match ends.
    """
    def __init__(
        self,
        generator: DynamicWeightGenerator,
        learning_rate: float = 0.02,
        replay_size: int = 512,
    ):
        self.generator = generator
        self.learning_rate = learning_rate
        self.replay = deque(maxlen=replay_size)

    def reset(self):
        """
        Called before every new match.
        """
        self.replay.clear()
        self.generator.reset_match()

    def record(
        self,
        features: Dict[str, float],
        predicted_value: float,
        context: EvaluationContext,
    ):
        """
        Store one evaluated state.
        """
        self.replay.append(
            {
                "features": dict(features),
                "prediction": float(predicted_value),
                "context": context,
            }
        )

    def finish_match(
        self,
        final_reward: float,
    ):
        """
        Update the temporary online weights.
        final_reward should normally be
            +1.0   win
             0.0   draw
            -1.0   loss
        """
        for sample in self.replay:
            prediction = sample["prediction"]
            td_error = final_reward - prediction
            for feature, value in sample["features"].items():
                delta = (
                    self.learning_rate
                    * td_error
                    * value
                )
                self.generator.online_weights[feature] = (
                    self.generator.online_weights.get(feature, 0.0)
                    + delta
                )

class Evaluator:
    """
    Stateless evaluator.
    Responsibilities:
        1. Obtain effective weights.
        2. Compute linear evaluation.
        3. Record the evaluation for online adaptation.
    It does NOT:
        - extract features
        - learn
        - maintain search state
        - know anything about Pokémon rules
    """
    def __init__(
        self,
        generator: DynamicWeightGenerator,
        learner: MatchLearner | None = None,
    ):
        self.generator = generator
        self.learner = learner

    def evaluate(
        self,
        features: Dict[str, float],
        context: EvaluationContext,
    ) -> EvaluationResult:
        weights = self.generator.generate(
            features,
            context,
        )

        score = 0.0
        contributions = {}

        for feature, value in features.items():
            w = weights.get(feature, 0.0)
            contribution = value * w
            contributions[feature] = contribution
            score += contribution

        # -----------------------------
        # Synergy bonuses
        # -----------------------------

        # Ready attacker with energy lead.
        score += (
            features.get("attack_ready", 0.0)
            * features.get("energy_advantage", 0.0)
            * 0.40
        )

        # Strong board with setup.
        score += (
            features.get("board_control", 0.0)
            * features.get("bench_ready_frac", 0.0)
            * 0.25
        )

        # Prize race acceleration.
        score += (
            features.get("prize_diff", 0.0)
            * features.get("tempo", 0.0)
            * 0.30
        )

        # Convert setup into aggression.
        score += (
            features.get("setup_ko", 0.0)
            * features.get("attack_ready", 0.0)
            * 0.45
        )

        # -----------------------------
        # Penalties
        # -----------------------------

        # Lots of energy but no attacker.
        if (
            features.get("energy_advantage", 0.0) > 0.5
            and features.get("attack_ready", 0.0) < 0.2
        ):
            score -= 0.35

        # Weak active while behind.
        if (
            features.get("my_active_hpfrac", 0.0) < 0.30
            and features.get("prize_diff", 0.0) < 0.0
        ):
            score -= 0.45

        # Empty bench.
        if (
            features.get("bench_frac", 0.0) < 0.20
        ):
            score -= 0.20

        # -----------------------------
        # Endgame scaling
        # -----------------------------

        phase = max(0.0, min(1.0, context.game_phase))

        score += (
            features.get("setup_ko", 0.0)
            * phase
            * 0.30
        )

        score += (
            features.get("prize_diff", 0.0)
            * phase
            * 0.20
        )

        # -----------------------------
        # Record for online learner
        # -----------------------------

        if self.learner is not None:
            self.learner.record(
                features=features,
                predicted_value=score,
                context=context,
            )

        return EvaluationResult(
            score=score,
            feature_scores=contributions,
            weights=weights,
            uncertainty=0.0,
        )