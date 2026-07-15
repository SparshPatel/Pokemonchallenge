"""
adaptive.py
Dynamic in-game adaptation layer.
Responsibilities
----------------
• Observe how the match evolves
• Detect changing game pace
• Detect opponent tendencies
• Detect repeated behaviour
• Produce adaptation signals
Contains NO search.
Contains NO engine calls.
Contains NO move evaluation.
Output:
    AdaptiveState
Consumed by:
    ExecutiveBrain
    OnlineLearningBrain
"""
from __future__ import annotations
from dataclasses import dataclass, field
# ----------------------------------------------------------------------
# Persistent match statistics
# ----------------------------------------------------------------------
@dataclass(slots=True)
class MatchStatistics:
    turns: int = 0
    prize_delta: int = 0
    attacks_seen: int = 0
    knockouts_seen: int = 0
    retreats_seen: int = 0
    gust_seen: int = 0
    supporters_seen: int = 0
    items_seen: int = 0
    energy_attachments_seen: int = 0
    bench_targets_seen: int = 0
    active_targets_seen: int = 0
    setup_turns: int = 0
    stalled_turns: int = 0

# ----------------------------------------------------------------------
# Adaptive output
# ----------------------------------------------------------------------
@dataclass(slots=True)
class AdaptiveState:
    aggression_shift: float = 0.0
    survival_shift: float = 0.0
    setup_shift: float = 0.0
    tempo_shift: float = 0.0
    resource_shift: float = 0.0
    gust_awareness: float = 0.0
    bench_protection: float = 0.0
    pressure: float = 0.0
    game_speed: float = 0.0
    confidence: float = 0.0
    opponent_is_fast: bool = False
    opponent_is_slow: bool = False
    opponent_is_aggressive: bool = False
    opponent_is_setup_focused: bool = False
    opponent_targets_bench: bool = False
    opponent_targets_active: bool = False
    opponent_prefers_gust: bool = False
    opponent_prefers_supporters: bool = False

# ----------------------------------------------------------------------
# Adaptive Brain
# ----------------------------------------------------------------------
class AdaptiveBrain:
    """
    Learns only from THIS match.
    It does not permanently remember anything.
    It simply watches the game unfold and adjusts
    strategic priorities accordingly.
    """
    def __init__(self):
        self.stats = MatchStatistics()

    # -------------------------------------------------------------
    def analyse(
        self,
        brain_state,
        strategy_state,
        prediction_state,
    ) -> AdaptiveState:
        self._update_statistics(
            brain_state,
        )
        adaptive = AdaptiveState()
        self._estimate_game_speed(
            adaptive,
        )
        self._estimate_aggression(
            adaptive,
        )
        self._estimate_targeting(
            adaptive,
        )
        self._estimate_resources(
            adaptive,
        )
        self._estimate_confidence(
            adaptive,
            strategy_state,
            prediction_state,
        )
        return adaptive

    # -------------------------------------------------------------
    def reset(self):
        self.stats = MatchStatistics()

    # -------------------------------------------------------------
    def _update_statistics(
        self,
        state,
    ):
        self.stats.turns += 1
        self.stats.prize_delta = (
            state.me.prizes_remaining
            - state.opponent.prizes_remaining
        )
        if state.last_action is None:
            return
        action = state.last_action
        action_type = getattr(
            action,
            "type",
            None,
        )
        if action_type == "ATTACK":
            self.stats.attacks_seen += 1
        elif action_type == "RETREAT":
            self.stats.retreats_seen += 1
        elif action_type == "PLAY_SUPPORTER":
            self.stats.supporters_seen += 1
        elif action_type == "PLAY_ITEM":
            self.stats.items_seen += 1
        elif action_type == "ATTACH":
            self.stats.energy_attachments_seen += 1
        elif action_type == "GUST":
            self.stats.gust_seen += 1
        if getattr(
            action,
            "knockout",
            False,
        ):
            self.stats.knockouts_seen += 1
        target = getattr(
            action,
            "target",
            None,
        )
        if target == "ACTIVE":
            self.stats.active_targets_seen += 1
        elif target == "BENCH":
            self.stats.bench_targets_seen += 1
            
    # -------------------------------------------------------------
    def _estimate_game_speed(
        self,
        adaptive,
    ):
        turns = max(
            1,
            self.stats.turns,
        )
        attack_rate = (
            self.stats.attacks_seen
            / turns
        )
        ko_rate = (
            self.stats.knockouts_seen
            / turns
        )
        adaptive.game_speed = (
            attack_rate
            + ko_rate
        )
        if attack_rate > ko_rate:
            adaptive.opponent_is_fast = True
        else:
            adaptive.opponent_is_slow = True

    # -------------------------------------------------------------
    def _estimate_aggression(
        self,
        adaptive,
    ):
        attacks = self.stats.attacks_seen
        retreats = self.stats.retreats_seen
        if attacks > retreats:
            adaptive.opponent_is_aggressive = True
            adaptive.aggression_shift += (
                attacks - retreats
            )
        elif retreats > attacks:
            adaptive.setup_shift += (
                retreats - attacks
            )
            adaptive.opponent_is_setup_focused = True
    
    # -------------------------------------------------------------
    def _estimate_targeting(
        self,
        adaptive,
    ):
        total_targets = (
            self.stats.active_targets_seen
            + self.stats.bench_targets_seen
        )
        if total_targets <= 0:
            return
        bench_ratio = (
            self.stats.bench_targets_seen
            / total_targets
        )
        active_ratio = (
            self.stats.active_targets_seen
            / total_targets
        )
        adaptive.bench_protection = bench_ratio
        adaptive.gust_awareness = (
            self.stats.gust_seen
            / max(
                1,
                self.stats.turns,
            )
        )
        if bench_ratio > active_ratio:
            adaptive.opponent_targets_bench = True
            adaptive.bench_protection += (
                bench_ratio
            )
        else:
            adaptive.opponent_targets_active = True

    # -------------------------------------------------------------
    def _estimate_resources(
        self,
        adaptive,
    ):
        turns = max(
            1,
            self.stats.turns,
        )
        supporter_rate = (
            self.stats.supporters_seen
            / turns
        )
        attachment_rate = (
            self.stats.energy_attachments_seen
            / turns
        )
        if supporter_rate > 0.5:
            adaptive.opponent_prefers_supporters = True
            adaptive.resource_shift += (
                supporter_rate
            )
        if self.stats.gust_seen > 0:
            adaptive.opponent_prefers_gust = True
        adaptive.pressure = (
            attachment_rate
            + supporter_rate
        )

    # -------------------------------------------------------------
    def _estimate_confidence(
        self,
        adaptive,
        strategy,
        prediction,
    ):
        confidence = 0.0
        if not strategy.comeback_mode:
            confidence += 1.0
        confidence -= (
            prediction.board_risk
        )
        confidence += (
            prediction.active_survival_probability
        )
        confidence -= (
            prediction.knockout_probability
        )
        confidence -= (
            abs(
                self.stats.prize_delta
            )
            * 0.10
        )
        adaptive.confidence = confidence
        if confidence < 0.0:
            adaptive.survival_shift += (
                abs(confidence)
            )
            adaptive.resource_shift += (
                abs(confidence)
            )
        else:
            adaptive.aggression_shift += (
                confidence
            )
            adaptive.tempo_shift += (
                confidence
            )

    # -------------------------------------------------------------
    def snapshot(
        self,
    ):
        """
        Returns a copy of the current match statistics.
        Useful for:
            replay logging
            online learning
            debugging
            post-game analysis
        """
        return MatchStatistics(
            turns=self.stats.turns,
            prize_delta=self.stats.prize_delta,
            attacks_seen=self.stats.attacks_seen,
            knockouts_seen=self.stats.knockouts_seen,
            retreats_seen=self.stats.retreats_seen,
            gust_seen=self.stats.gust_seen,
            supporters_seen=self.stats.supporters_seen,
            items_seen=self.stats.items_seen,
            energy_attachments_seen=self.stats.energy_attachments_seen,
            bench_targets_seen=self.stats.bench_targets_seen,
            active_targets_seen=self.stats.active_targets_seen,
            setup_turns=self.stats.setup_turns,
            stalled_turns=self.stats.stalled_turns,
        )