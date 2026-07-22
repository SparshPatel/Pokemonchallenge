"""
online_learning.py
Online adaptation during a single match.
Responsibilities
----------------
• Observe action outcomes
• Track success/failure
• Update tactical preferences
• Update strategic preferences
• Produce lightweight learned adjustments
Contains NO search.
Contains NO engine calls.
Contains NO permanent storage.
Learning is reset every match.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .tactical import TacticalWeights
# ----------------------------------------------------------------------
# Learning state
# ----------------------------------------------------------------------
@dataclass(slots=True)
class OnlineLearningState:
    attack_bias: float = 0.0
    attach_bias: float = 0.0
    evolve_bias: float = 0.0
    retreat_bias: float = 0.0
    supporter_bias: float = 0.0
    item_bias: float = 0.0
    gust_bias: float = 0.0
    aggression_bias: float = 0.0
    survival_bias: float = 0.0
    setup_bias: float = 0.0
    tempo_bias: float = 0.0
    resource_bias: float = 0.0
    confidence: float = 0.0
    updates: int = 0

# ----------------------------------------------------------------------
# Experience
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Experience:
    action_type: str
    reward: float
    turn: int
    metadata: dict = field(default_factory=dict)

# ----------------------------------------------------------------------
# Online Learning Brain
# ----------------------------------------------------------------------
class OnlineLearningBrain:
    """
    Lightweight online learner.
    Learns only during the current game.
    Does NOT permanently save anything.
    """
    def __init__(self):
        self.state = OnlineLearningState()
        self.history: list[Experience] = []
        self.learning_rate = 0.05
        self.max_bias = 2.0

    # ---------------------------------------------------------
    def reset(self):
        self.state = OnlineLearningState()
        self.history.clear()

    # ---------------------------------------------------------
    def observe(
        self,
        action_type,
        reward,
        turn,
        **metadata,
    ):
        exp = Experience(
            action_type=action_type,
            reward=float(reward),
            turn=int(turn),
            metadata=dict(metadata),
        )
        self.history.append(exp)
        self._update(exp)

    # ---------------------------------------------------------
    def tactical_weights(
        self,
        base: TacticalWeights,
    ) -> TacticalWeights:
        return TacticalWeights(
            attack_bonus=base.attack_bonus
            + self.state.attack_bias,
            ko_bonus=base.ko_bonus,
            retreat_bonus=base.retreat_bonus
            + self.state.retreat_bias,
            retreat_penalty=base.retreat_penalty,
            supporter_bonus=base.supporter_bonus
            + self.state.supporter_bias,
            item_bonus=base.item_bonus
            + self.state.item_bias,
            gust_bonus=base.gust_bonus
            + self.state.gust_bias,
            bench_bonus=base.bench_bonus,
            attach_bonus=base.attach_bonus
            + self.state.attach_bias,
            attach_penalty=base.attach_penalty,
            evolve_bonus=base.evolve_bonus
            + self.state.evolve_bias,
            discard_penalty=base.discard_penalty,
        )

    # ---------------------------------------------------------
    def strategic_adjustments(self):
        return {
            "aggression": self.state.aggression_bias,
            "survival": self.state.survival_bias,
            "setup": self.state.setup_bias,
            "tempo": self.state.tempo_bias,
            "resource": self.state.resource_bias,
            "confidence": self.state.confidence,
        }

    # ---------------------------------------------------------
    def _update(
        self,
        exp: Experience,
    ):
        lr = self.learning_rate
        r = exp.reward
        if exp.action_type == "ATTACK":
            self.state.attack_bias += lr * r
            self.state.aggression_bias += lr * r
        elif exp.action_type == "ATTACH":
            self.state.attach_bias += lr * r
            self.state.setup_bias += lr * r
        elif exp.action_type == "EVOLVE":
            self.state.evolve_bias += lr * r
            self.state.setup_bias += lr * r
        elif exp.action_type == "RETREAT":
            self.state.retreat_bias += lr * r
            self.state.survival_bias += lr * r
        elif exp.action_type == "PLAY_SUPPORTER":
            self.state.supporter_bias += lr * r
            self.state.resource_bias += lr * r
        elif exp.action_type == "PLAY_ITEM":
            self.state.item_bias += lr * r
        elif exp.action_type == "GUST":
            self.state.gust_bias += lr * r
            self.state.tempo_bias += lr * r
        self.state.confidence += (
            lr * r * 0.25
        )
        self._clip()
        self.state.updates += 1

    # ---------------------------------------------------------
    def _clip(self):
        m = self.max_bias
        attrs = (
            "attack_bias",
            "attach_bias",
            "evolve_bias",
            "retreat_bias",
            "supporter_bias",
            "item_bias",
            "gust_bias",
            "aggression_bias",
            "survival_bias",
            "setup_bias",
            "tempo_bias",
            "resource_bias",
            "confidence",
        )
        for name in attrs:
            value = getattr(
                self.state,
                name,
            )
            if value > m:
                value = m
            elif value < -m:
                value = -m
            setattr(
                self.state,
                name,
                value,
            )

    # ---------------------------------------------------------
    def summary(self):
        return {
            "updates": self.state.updates,
            "history": len(self.history),
            "state": self.state,
        }