"""
tactical.py
Pure tactical feature extractor.
This module NEVER:
    • calls the engine
    • searches
    • evaluates board state directly
It simply measures tactical properties of one candidate action.
The returned features are later weighted by
AdaptiveBrain / WeightGenerator / ValueNet.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from agent.adapter import OptionType
# ----------------------------------------------------------------------
# Tactical feature vector
# ----------------------------------------------------------------------
@dataclass(slots=True)
class TacticalFeatures:
    attack: float = 0.0
    ko_now: float = 0.0
    setup_ko: float = 0.0
    retreat: float = 0.0
    safe_retreat: float = 0.0
    energy_loss: float = 0.0
    evolve: float = 0.0
    bench_setup: float = 0.0
    attach: float = 0.0
    attach_active: float = 0.0
    attach_future: float = 0.0
    supporter: float = 0.0
    item: float = 0.0
    gust: float = 0.0
    discard_cost: float = 0.0
    tempo_gain: float = 0.0
    threat_removed: float = 0.0
    board_development: float = 0.0

# ----------------------------------------------------------------------
# Tactical brain
# ----------------------------------------------------------------------
class TacticalBrain:
    """
    Stateless tactical analyser.
    Produces feature values.
    Never produces a final score.
    """
    def extract(
        self,
        obs_dict,
        state,
        option,
        planner,
    ) -> TacticalFeatures:
        features = TacticalFeatures()
        t = option.type
        if t == OptionType.ATTACK:
            self._attack(
                features,
                obs_dict,
                state,
                option,
                planner,
            )
        elif t == OptionType.ATTACH:
            self._attach(
                features,
                obs_dict,
                state,
                option,
                planner,
            )
        elif t == OptionType.EVOLVE:
            self._evolve(
                features,
                obs_dict,
                state,
                option,
                planner,
            )
        elif t == OptionType.RETREAT:
            self._retreat(
                features,
                obs_dict,
                state,
                option,
                planner,
            )
        elif t == OptionType.PLAY:
            self._play(
                features,
                obs_dict,
                state,
                option,
                planner,
            )
        return features

    # ------------------------------------------------------------------
    def _attack(
        self,
        f,
        obs,
        state,
        option,
        planner,
    ):
        f.attack = 1.0
        # future:
        # f.ko_now
        # f.setup_ko
        # f.threat_removed
        # f.tempo_gain

    # ------------------------------------------------------------------
    def _attach(
        self,
        f,
        obs,
        state,
        option,
        planner,
    ):
        f.attach = 1.0
        # determine whether attachment is
        # active
        # future attacker
        # wasted

    # ------------------------------------------------------------------
    def _evolve(
        self,
        f,
        obs,
        state,
        option,
        planner,
    ):
        f.evolve = 1.0
        # future
        # board development
        # hp increase
        # ability unlock

    # ------------------------------------------------------------------
    def _retreat(
        self,
        f,
        obs,
        state,
        option,
        planner,
    ):
        f.retreat = 1.0
        # future
        # safe retreat
        # energy loss

    # ------------------------------------------------------------------
    def _play(
        self,
        f,
        obs,
        state,
        option,
        planner,
    ):
        f.item = 1.0
        # future
        # supporter
        # gust
        # discard
        # board setup