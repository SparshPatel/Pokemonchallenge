"""Value-net leaf evaluation for the TurnPlanner (CPU, pure-numpy inference).
The planner scores each candidate end-of-turn board with a *value function*.
``agent.planner._eval`` is a hand-tuned linear combination of board features.
This module provides a drop-in learned alternative: the SAME feature vector is
fed to a small model (logistic regression / 1-hidden-layer MLP) whose weights
are trained offline on self-play rollouts (label = eventual game win).
Design constraints (submission must run on Kaggle, CPU, no heavy deps):
* Inference is pure numpy — no sklearn/torch at runtime.
* Weights load from a small ``.npz`` bundled next to this file; if the file is
  absent or malformed, ``ValueNet.available`` is False and the planner keeps
  using its hand-tuned ``_eval`` (fail-safe, never raises).
The feature extractor deliberately reuses the exact primitives that ``_eval``
uses (``_prizes_left``, ``_active``, ``_best_affordable_dmg``, ``_can_attack``,
``_energy_in_play``, ``gd.best_damage``, ``gd.prize_value``) so the learned
value is grounded in the same signal the hand model already trusted.
"""
from __future__ import annotations
import os
import math
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is available in the sim image
    np = None
# Ordered feature names — the training pipeline MUST emit vectors in this order.
FEATURE_NAMES = (
    # ----------------------------
    # Prize state
    # ----------------------------
    "prize_diff",
    "my_prize_left",
    "opp_prize_left",
    # ----------------------------
    # Active Pokémon
    # ----------------------------
    "my_active_hpfrac",
    "opp_active_hpfrac",
    "opp_active_dmgfrac",
    "my_ready",
    "setup_ko",
    "opp_threat",
    "active_quality",
    "active_loaded",
    # ----------------------------
    # Bench
    # ----------------------------
    "bench_frac",
    "bench_ready_frac",
    "bench_setup_ko",
    "opp_bench_dmg",
    # ----------------------------
    # Resources
    # ----------------------------
    "energy_frac",
    "hand_frac",
    # ----------------------------
    # Tempo
    # ----------------------------
    "my_can_attack",
    "opp_can_attack",
    "energy_advantage",
    "board_control",
    "supporter_available",
    "gust_available",
    "switch_available",
    "stadium_in_play",
    # ----------------------------
    # Risk
    # ----------------------------
    "multi_prize_risk",
    "bench_liability",
    "no_active",
    "bias",
)
N_FEATURES = len(FEATURE_NAMES)

def extract_features(state, me, gd, helpers):
    """
    Extract normalized board features.
    Values are intentionally smooth and normalized so the network
    generalizes well between different board states.
    """
    players = state.get("players") or []
    if len(players) < 2:
        return [0.0] * (N_FEATURES - 1) + [1.0]
    mp = players[me] if isinstance(players[me], dict) else {}
    op = players[1 - me] if isinstance(players[1 - me], dict) else {}
    my_left = helpers._prizes_left(mp)
    opp_left = helpers._prizes_left(op)
    my_act = helpers._active(mp)
    opp_act = helpers._active(op)
    f = {k: 0.0 for k in FEATURE_NAMES}
    # Convenience references
    my_bench = list(mp.get("bench") or [])
    opp_bench = list(op.get("bench") or [])
    my_energy = helpers._energy_in_play(mp)
    opp_energy = helpers._energy_in_play(op)
    # ----------------------------------------------------------
    # Prize race
    # ----------------------------------------------------------
    f["prize_diff"] = (opp_left - my_left) / 6.0
    f["my_prize_left"] = my_left / 6.0
    f["opp_prize_left"] = opp_left / 6.0
    # ----------------------------------------------------------
    # Active Pokémon
    # ----------------------------------------------------------
    if my_act:
        mhp = my_act.get("hp") or 0
        mmax = my_act.get("maxHp") or 0
        if mmax > 0:
            f["my_active_hpfrac"] = mhp / mmax
        if helpers._can_attack(my_act, gd):
            f["my_ready"] = 1.0
    else:
        f["no_active"] = 1.0
    if opp_act:
        ohp = opp_act.get("hp") or 0
        omax = opp_act.get("maxHp") or 0
        if omax > 0:
            f["opp_active_hpfrac"] = ohp / omax
            f["opp_active_dmgfrac"] = (omax - ohp) / omax
    # ----------------------------------------------------------
    # KO / threat
    # ----------------------------------------------------------
    if my_act and opp_act:
        my_dmg = helpers._best_affordable_dmg(
            my_act,
            opp_act,
            gd,
        )
        if my_dmg >= (opp_act.get("hp") or 0):
            f["setup_ko"] = 1.0
        opp_dmg = helpers._best_affordable_dmg(
            opp_act,
            my_act,
            gd,
        )
        if opp_dmg >= (my_act.get("hp") or 0):
            f["opp_threat"] = 1.0
    # ----------------------------------------------------------
    # Active quality
    # ----------------------------------------------------------
    if my_act:
        team_best = 0
        for p in [my_act] + list(mp.get("bench") or []):
            if isinstance(p, dict):
                team_best = max(
                    team_best,
                    gd.best_damage(p.get("id")),
                )
        active_best = gd.best_damage(my_act.get("id"))
        if team_best > 0:
            f["active_quality"] = active_best / team_best
        if active_best > 0 and opp_act:
            affordable = helpers._best_affordable_dmg(
                my_act,
                opp_act,
                gd,
            )
            f["active_loaded"] = min(
                affordable / active_best,
                1.0,
            )
    # ----------------------------------------------------------
    # Bench
    # ----------------------------------------------------------
    bench = list(mp.get("bench") or [])
    f["bench_frac"] = min(
        len(bench),
        5,
    ) / 5.0
    ready = 0
    for bp in bench:
        if isinstance(bp, dict) and helpers._can_attack(bp, gd):
            ready += 1
    f["bench_ready_frac"] = min(
        ready,
        5,
    ) / 5.0
    # ----------------------------------------------------------
    # Energy
    # ----------------------------------------------------------
    f["energy_frac"] = min(
        helpers._energy_in_play(mp),
        12,
    ) / 12.0
    # Tempo features
    f["my_can_attack"] = 1.0 if (
        my_act and helpers._can_attack(my_act, gd)
    ) else 0.0
    f["opp_can_attack"] = 1.0 if (
        opp_act and helpers._can_attack(opp_act, gd)
    ) else 0.0
    f["energy_advantage"] = (
        my_energy - opp_energy
    ) / 12.0
    my_board = len(my_bench)
    opp_board = len(opp_bench)
    f["board_control"] = (
        (my_board - opp_board)
        / 5.0
    )
    # ----------------------------------------------------------
    # Hand
    # ----------------------------------------------------------
    hand_count = int(mp.get("handCount") or 0)
    f["hand_frac"] = min(
        hand_count,
        10,
    ) / 10.0
    # Trainer availability
    hand_cards = mp.get("hand") or []
    supporter = 0
    gust = 0
    switch = 0
    for card in hand_cards:
        cid = card.get("id")
        if gd.is_supporter(cid):
            supporter = 1
        name = (
            gd.card_name.get(cid, "")
            .lower()
        )
        if (
            "boss" in name
            or "catcher" in name
        ):
            gust = 1

        if "switch" in name:
            switch = 1
    f["supporter_available"] = supporter
    f["gust_available"] = gust
    f["switch_available"] = switch
    # ----------------------------------------------------------
    # Opponent bench pressure
    # ----------------------------------------------------------
    bench_damage = 0.0
    bench_kos = 0
    if my_act:
        for bp in (op.get("bench") or []):
            if not isinstance(bp, dict):
                continue
            hp = bp.get("hp") or 0
            max_hp = bp.get("maxHp") or 0
            if max_hp > 0:
                bench_damage += (
                    (max_hp - hp)
                    / max_hp
                ) * gd.prize_value(bp.get("id"))
            if hp > 0:
                dmg = helpers._best_affordable_dmg(
                    my_act,
                    bp,
                    gd,
                )
                if dmg >= hp:
                    bench_kos += 1
    f["opp_bench_dmg"] = min(
        bench_damage,
        5.0,
    ) / 5.0
    f["bench_setup_ko"] = min(
        bench_kos,
        5,
    ) / 5.0
    # ----------------------------------------------------------
    # Bias
    # ----------------------------------------------------------
    f["bias"] = 1.0
    # Stadium
    f["stadium_in_play"] = (
        1.0
        if state.get("stadium")
        else 0.0
    )
    # Multi-prize risk
    risk = 0
    for p in [my_act] + my_bench:
        if not isinstance(p, dict):
            continue
        risk += gd.prize_value(
            p.get("id")
        )
    f["multi_prize_risk"] = risk / 10.0
    # Bench liability
    liability = 0
    for p in my_bench:
        hp = p.get("hp", 0)
        if hp < 90:
            liability += 1
    f["bench_liability"] = liability / 5.0
    return [f[name] for name in FEATURE_NAMES]
class ValueNet:
    """
    CPU-only value function.
    Supports:
        • logistic regression
        • one hidden layer MLP
    Additionally maintains an online bias term which the Supervisor can
    continuously adapt without retraining the full network.
    Final prediction:
        sigmoid(network + adaptive_bias)
    This lets long tournament runs calibrate evaluation while preserving the
    offline trained weights.
    """
    def __init__(
        self,
        path=None,
    ):
        self.available = False
        self.kind = None
        # Logistic model
        self._w = None
        # MLP model
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = None
        # Online calibration bias
        self.bias = 0.0
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "value_net.npz",
            )
        if np is None or not os.path.exists(path):
            return
        try:
            d = np.load(
                path,
                allow_pickle=False,
            )
            kind = (
                str(d["kind"])
                if "kind" in d
                else "logistic"
            )
            if kind == "logistic":
                w = np.asarray(
                    d["w"],
                    dtype=np.float64,
                ).ravel()
                if w.shape[0] != N_FEATURES:
                    return
                self._w = w
                self.kind = "logistic"
            elif kind == "mlp":
                self._W1 = np.asarray(
                    d["W1"],
                    dtype=np.float64,
                )
                self._b1 = np.asarray(
                    d["b1"],
                    dtype=np.float64,
                ).ravel()
                self._W2 = np.asarray(
                    d["W2"],
                    dtype=np.float64,
                ).ravel()
                self._b2 = float(
                    np.asarray(
                        d["b2"],
                    ).ravel()[0]
                )
                if self._W1.shape[0] != N_FEATURES:
                    return
                self.kind = "mlp"
            else:
                return
            self.available = True
        except Exception:
            self.available = False

    # -------------------------------------------------------------
    def raw_value(self, feats):
        """
        Raw network output before sigmoid.
        """
        x = np.asarray(
            feats,
            dtype=np.float64,
        ).ravel()
        if self.kind == "logistic":
            return float(x @ self._w)
        h = np.tanh(
            x @ self._W1 + self._b1
        )
        return float(
            h @ self._W2 + self._b2
        )

    # -------------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (
                1.0 + np.exp(-z)
            )
        e = np.exp(z)
        return e / (1.0 + e)

    # -------------------------------------------------------------
    def predict(
        self,
        feats,
    ):
        """
        Returns calibrated win probability.
        """
        if not self.available:
            return 0.5
        z = self.raw_value(feats)
        z += self.bias
        temperature = 1.5
        z /= temperature
        z = max(min(z, 20.0), -20.0)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        e = np.exp(z)
        return e / (1.0 + e)

    def evaluate(
        self,
        feats,
    ):
        """
        Planner-friendly evaluation.
        Returns a symmetric value in [-1, 1] instead of a probability.
        """
        p = self.predict(feats)
        return (2.0 * p) - 1.0
    
    # -------------------------------------------------------------
    def calibrate(
        self,
        predicted,
        outcome,
        lr=0.05,
    ):
        """
        Lightweight online calibration.
        predicted : network probability
        outcome :
            1 = eventually won
            0 = eventually lost
        Only adjusts a single bias term.
        """
        error = outcome - predicted
        self.bias += lr * error
        
    # -------------------------------------------------------------
    def reset_calibration(self):
        self.bias = 0.0