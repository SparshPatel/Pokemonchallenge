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

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is available in the sim image
    np = None


# Ordered feature names — the training pipeline MUST emit vectors in this order.
FEATURE_NAMES = (
    "prize_diff",        # (opp_left - my_left) / 6, in [-1, 1]
    "my_prize_left",     # my_left / 6
    "opp_prize_left",    # opp_left / 6
    "my_active_hpfrac",  # my active hp / maxHp, 0 if none
    "opp_active_hpfrac", # opp active hp / maxHp, 0 if none
    "opp_active_dmgfrac",# (maxHp-hp)/maxHp on opp active
    "my_ready",          # 1 if our active can attack now
    "setup_ko",          # 1 if our affordable dmg >= opp active hp
    "opp_threat",        # 1 if opp affordable dmg >= our active hp
    "active_quality",    # active best_dmg / our best available best_dmg
    "active_loaded",     # active affordable_dmg / active best_dmg
    "bench_frac",        # our bench count / 5
    "bench_ready_frac",  # our bench-ready count / 5
    "energy_frac",       # our energy in play / 12 (soft cap)
    "hand_frac",         # our hand count / 10 (soft cap)
    "opp_bench_dmg",     # sum over opp bench of (maxHp-hp)/maxHp * prize_value, /5
    "bench_setup_ko",    # count opp bench we can KO now (after gust), /5
    "no_active",         # 1 if we have no active (very bad)
    "bias",              # constant 1.0
)
N_FEATURES = len(FEATURE_NAMES)


def extract_features(state, me, gd, helpers):
    """Return a length-N_FEATURES python list of floats for ``state`` from
    player ``me``'s perspective.

    ``helpers`` is a namespace/module exposing the planner primitives so this
    module has no import cycle with planner.py: ``_active``, ``_prizes_left``,
    ``_best_affordable_dmg``, ``_can_attack``, ``_energy_in_play``.
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
    f["prize_diff"] = (opp_left - my_left) / 6.0
    f["my_prize_left"] = my_left / 6.0
    f["opp_prize_left"] = opp_left / 6.0

    if opp_act:
        omax = opp_act.get("maxHp") or 0
        ohp = opp_act.get("hp") or 0
        if omax > 0:
            f["opp_active_hpfrac"] = ohp / omax
            f["opp_active_dmgfrac"] = (omax - ohp) / omax
        if my_act and ohp > 0:
            dmg = helpers._best_affordable_dmg(my_act, opp_act, gd)
            if dmg >= ohp:
                f["setup_ko"] = 1.0

    if my_act:
        mmax = my_act.get("maxHp") or 0
        mhp = my_act.get("hp") or 0
        if mmax > 0:
            f["my_active_hpfrac"] = mhp / mmax
        if helpers._can_attack(my_act, gd):
            f["my_ready"] = 1.0
        if opp_act and mhp > 0:
            othreat = helpers._best_affordable_dmg(opp_act, my_act, gd)
            if othreat >= mhp:
                f["opp_threat"] = 1.0
        best_pot = 0
        for p in [my_act] + list(mp.get("bench") or []):
            if isinstance(p, dict):
                best_pot = max(best_pot, gd.best_damage(p.get("id")))
        act_pot = gd.best_damage(my_act.get("id"))
        if best_pot > 0:
            f["active_quality"] = act_pot / best_pot
        if act_pot > 0:
            aff = helpers._best_affordable_dmg(my_act, opp_act or {}, gd)
            f["active_loaded"] = min(aff / act_pot, 1.0)
    else:
        f["no_active"] = 1.0

    bench = list(mp.get("bench") or [])
    f["bench_frac"] = min(len(bench), 5) / 5.0
    ready = sum(1 for bp in bench if isinstance(bp, dict) and helpers._can_attack(bp, gd))
    f["bench_ready_frac"] = min(ready, 5) / 5.0
    f["energy_frac"] = min(helpers._energy_in_play(mp), 12) / 12.0
    f["hand_frac"] = min(int(mp.get("handCount") or 0), 10) / 10.0

    ob_dmg = 0.0
    ko_ct = 0
    for bp in (op.get("bench") or []):
        if not isinstance(bp, dict):
            continue
        omax = bp.get("maxHp") or 0
        ohp = bp.get("hp") or 0
        pv = gd.prize_value(bp.get("id"))
        if omax > 0 and ohp < omax:
            ob_dmg += ((omax - ohp) / omax) * pv
        if my_act and 0 < ohp:
            if helpers._best_affordable_dmg(my_act, bp, gd) >= ohp:
                ko_ct += 1
    f["opp_bench_dmg"] = min(ob_dmg, 5.0) / 5.0
    f["bench_setup_ko"] = min(ko_ct, 5) / 5.0
    f["bias"] = 1.0

    return [f[k] for k in FEATURE_NAMES]


class ValueNet:
    """Pure-numpy value model. Supports logistic (1 layer) or 1-hidden MLP.

    Weight bundle (.npz) keys:
      kind = "logistic":  w[N_FEATURES]           -> sigmoid(x·w)
      kind = "mlp":       W1[N_FEATURES,H], b1[H], W2[H], b2  -> sigmoid(...)
    Output is P(win) in (0,1); the planner scales it into its value range.
    """

    def __init__(self, path=None):
        self.available = False
        self.kind = None
        self._w = None
        self._W1 = self._b1 = self._W2 = self._b2 = None
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "value_net.npz")
        if np is None or not os.path.exists(path):
            return
        try:
            d = np.load(path, allow_pickle=False)
            kind = str(d["kind"]) if "kind" in d else "logistic"
            if kind == "logistic":
                w = np.asarray(d["w"], dtype=np.float64).ravel()
                if w.shape[0] != N_FEATURES:
                    return
                self._w = w
                self.kind = "logistic"
            elif kind == "mlp":
                self._W1 = np.asarray(d["W1"], dtype=np.float64)
                self._b1 = np.asarray(d["b1"], dtype=np.float64).ravel()
                self._W2 = np.asarray(d["W2"], dtype=np.float64).ravel()
                self._b2 = float(np.asarray(d["b2"]).ravel()[0])
                if self._W1.shape[0] != N_FEATURES:
                    return
                self.kind = "mlp"
            else:
                return
            self.available = True
        except Exception:
            self.available = False

    def predict(self, feats):
        """feats: length-N_FEATURES sequence -> P(win) float in (0,1)."""
        x = np.asarray(feats, dtype=np.float64).ravel()
        if self.kind == "logistic":
            z = float(x @ self._w)
        else:
            h = np.tanh(x @ self._W1 + self._b1)
            z = float(h @ self._W2 + self._b2)
        # numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        e = np.exp(z)
        return e / (1.0 + e)
