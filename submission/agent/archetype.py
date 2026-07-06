"""Opponent archetype detection + specialized play profiles ("sub-agents").

The autonomous rules/planner stack plays one generic game plan. But the field
contains structurally different opponents, and the *same* board calls for
different priorities depending on who is across the table:

* **WALL / PIVOT** — high-HP single-prize tanks that retreat the moment they take
  damage, denying clean KOs and grinding the prize race. The counter is to drag
  wounded Pokémon back up with Boss's Orders / gust and value spread damage on
  the bench, rather than politely trading with a fresh full-HP wall each turn.
* **AGGRO / PLUNDERER** — fast, high-damage ex attackers that race prizes. The
  counter is to prioritise survivability and clean prize trades: don't over-
  commit Energy onto one attacker that gets KO'd for free, keep a ready answer
  on the bench, and value our own HP / threat-avoidance more.
* **BALANCED / UNKNOWN** — no strong signal yet; play the default game plan.

This module watches the opponent's board across our decisions (fully observable
signals only, so it never depends on fragile log accounting) and, after a few
turns, classifies the opponent and emits *weight deltas* for both the rule
policy (:mod:`agent.rules`) and the planner leaf-eval (:mod:`agent.planner`).
The deltas are small, bounded nudges layered on top of the tuned base weights —
they change emphasis, never legality, and default to zero (base behaviour) until
a confident read is available. Everything is crash-safe: any error leaves the
detector in the UNKNOWN state, i.e. the unchanged base agent.
"""
from __future__ import annotations

from .gamedata import GameData

# Archetype labels.
UNKNOWN = "unknown"
WALL = "wall"
AGGRO = "aggro"

# Per-archetype additive deltas applied to ``rules.WEIGHTS`` keys.
_RULES_DELTA: dict[str, dict[str, float]] = {
    WALL: {
        # Drag wounded walls back into the Active spot and finish them.
        "gust_ko": 200.0,
        "gust_ex": 60.0,
        "gust_mega_ex": 90.0,
        # A wall race is won by 2HKO chains and bench pressure, so value
        # damage-on-a-high-prize-target and two-shot setups more.
        "attack_prize_bonus": 25.0,
        "attack_twoshot_bonus": 30.0,
        # Commit Energy to one real attacker that can punch through big HP
        # instead of dribbling it across the board.
        "attach_concentrate": 20.0,
        "attach_dmg_cap": 60.0,
    },
    AGGRO: {
        # Survive the race: retreat a doomed Active to a fresh attacker and keep
        # trading cleanly rather than getting KO'd for free.
        "retreat_danger": 150.0,
        # Fully load ONE attacker so every swing trades a prize, don't spread.
        "attach_concentrate": 25.0,
        # Late-game urgency matters more when we're being raced.
        "attack_urgency": 100.0,
    },
}

# Per-archetype additive deltas applied to ``planner.EVAL`` (leaf value) keys.
_EVAL_DELTA: dict[str, dict[str, float]] = {
    WALL: {
        "opp_bench_dmg": 25.0,    # credit chip damage sitting on the bench
        "bench_setup_ko": 18.0,   # value gust+KO on a wounded bench target
        "prize": 20.0,            # the prize race is the whole game vs a wall
    },
    AGGRO: {
        "opp_threat": 40.0,       # avoid leaving our Active in KO range
        "my_hp": 14.0,            # survivability
        "no_active": 80.0,        # never get caught with no Active in a race
        "bench_ready": 10.0,      # keep a powered answer on the bench
    },
}

# High-HP threshold (in HP) above which a non-ex Pokémon reads as a "wall".
_WALL_HP = 130
# maxHp seen anywhere that reads as a dedicated tank.
_TANK_HP = 200


class ArchetypeDetector:
    """Stateful, per-game opponent classifier. Reset automatically on new games."""

    def __init__(self, gd: GameData):
        self.gd = gd
        self.reset()

    def reset(self) -> None:
        self.decisions = 0
        self.turn = 0
        self.opp_ex_ids: set[int] = set()
        self.opp_wall_ids: set[int] = set()
        self.max_opp_hp = 0
        self.prizes_lost = 0          # prizes the opponent has taken from us
        self.opp_retreats = 0
        self._last_opp_active: tuple | None = None  # (serial, id, hp)
        self._locked: str = UNKNOWN

    # --- observation ------------------------------------------------------
    def update(self, obs_dict) -> None:
        """Fold one of our decision observations into the running signals."""
        try:
            self._update(obs_dict)
        except Exception:
            pass  # never let detection break the agent

    def _update(self, obs_dict) -> None:
        if not isinstance(obs_dict, dict):
            return
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return
        turn = state.get("turn")
        yi = state.get("yourIndex", 0)
        if not isinstance(yi, int):
            yi = 0

        # New game? Turn counter went backwards -> reset accumulated signals.
        if isinstance(turn, int):
            if self.decisions > 0 and turn < self.turn:
                self.reset()
            self.turn = max(self.turn, turn)
        self.decisions += 1

        players = state.get("players") or []
        if len(players) < 2:
            return
        me = players[yi] if 0 <= yi < len(players) and isinstance(players[yi], dict) else {}
        op = players[1 - yi] if isinstance(players[1 - yi], dict) else {}

        # Prizes the opponent has taken from us so far (we start with 6).
        my_prize = me.get("prize")
        if isinstance(my_prize, list):
            self.prizes_lost = max(self.prizes_lost, 6 - len(my_prize))

        # Opponent board: ex/mega presence, tank HP, wall bodies.
        opp_active = None
        oa = op.get("active") or []
        if oa and isinstance(oa[0], dict):
            opp_active = oa[0]
        for mon in ([opp_active] + list(op.get("bench") or [])):
            if not isinstance(mon, dict):
                continue
            cid = mon.get("id")
            maxhp = int(mon.get("maxHp") or 0)
            self.max_opp_hp = max(self.max_opp_hp, maxhp)
            if cid is not None and self.gd.is_ex(cid):
                self.opp_ex_ids.add(cid)
            elif cid is not None and maxhp >= _WALL_HP:
                self.opp_wall_ids.add(cid)

        # Pivot detection: the opponent's Active identity changed while the
        # previous Active is still alive (on the bench) -> they retreated it.
        if isinstance(opp_active, dict):
            cur = (opp_active.get("serial"), opp_active.get("id"), opp_active.get("hp"))
            prev = self._last_opp_active
            if prev is not None and prev[0] is not None and cur[0] != prev[0]:
                bench_serials = {
                    b.get("serial") for b in (op.get("bench") or [])
                    if isinstance(b, dict)
                }
                if prev[0] in bench_serials:
                    self.opp_retreats += 1
            self._last_opp_active = cur

    # --- classification ---------------------------------------------------
    def archetype(self) -> str:
        """Current best read of the opponent archetype.

        Conservative: returns UNKNOWN until enough of the game has been seen
        (a few turns) so early-setup noise cannot flip us into a bad profile.
        Once WALL or AGGRO is asserted with a strong margin it *locks* for the
        game to avoid oscillating profiles.
        """
        if self._locked != UNKNOWN:
            return self._locked
        if self.turn < 3:
            return UNKNOWN

        wall = 0
        wall += 1 if self.max_opp_hp >= _TANK_HP else 0
        wall += 1 if len(self.opp_wall_ids) >= 2 else 0
        wall += 1 if self.opp_retreats >= 2 else 0
        # Slow prize race late = grindy wall.
        wall += 1 if (self.turn >= 6 and self.prizes_lost <= 1) else 0
        wall -= 1 if len(self.opp_ex_ids) >= 2 else 0

        aggro = 0
        aggro += 1 if len(self.opp_ex_ids) >= 2 else 0
        aggro += 1 if self.prizes_lost >= 3 else 0
        # Fast prize pressure (>= ~0.5 prize/turn taken off us).
        aggro += 1 if (self.turn >= 4 and self.prizes_lost * 2 >= self.turn) else 0
        aggro -= 1 if len(self.opp_wall_ids) >= 2 else 0

        if wall >= 2 and wall > aggro:
            if wall >= 3:
                self._locked = WALL
            return WALL
        if aggro >= 2 and aggro > wall:
            if aggro >= 3:
                self._locked = AGGRO
            return AGGRO
        return UNKNOWN

    # --- profile emission -------------------------------------------------
    def rules_delta(self) -> dict[str, float]:
        return dict(_RULES_DELTA.get(self.archetype(), {}))

    def eval_delta(self) -> dict[str, float]:
        return dict(_EVAL_DELTA.get(self.archetype(), {}))
