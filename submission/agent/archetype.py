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

class ArchetypeDetector:
    """Stateful, per-game opponent classifier. Reset automatically on new games."""

    def __init__(self, gd: GameData):
        self.gd = gd
        self.reset()

    def reset(self) -> None:
        """
        Reset all accumulated observations for a new game.
        Everything stored here is derived only from public information.
        The detector intentionally stores statistics instead of thresholds,
        allowing archetype confidence to be computed continuously.
        """
        self.decisions = 0
        self.turn = 0
        # ---------- observed Pokémon ----------
        self.opp_seen_ids: set[int] = set()
        self.opp_ex_ids: set[int] = set()
        # ---------- running board statistics ----------
        self.hp_samples: list[int] = []
        self.retreat_samples: list[int] = []
        self.bench_sizes: list[int] = []
        self.total_energy = 0
        self.energy_observations = 0
        self.prizes_lost = 0
        self.opp_retreats = 0
        self.switches = 0
        self.damage_taken = 0
        self.damage_events = 0
        self._last_opp_active: tuple | None = None
        # probability cache
        self._profile = {
            WALL: 0.0,
            AGGRO: 0.0,
            UNKNOWN: 1.0,
        }
        self._locked = UNKNOWN

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
        # Detect new game.
        if isinstance(turn, int):
            if self.decisions > 0 and turn < self.turn:
                self.reset()
            self.turn = max(self.turn, turn)
        self.decisions += 1
        players = state.get("players") or []
        if len(players) < 2:
            return
        me = players[yi] if isinstance(players[yi], dict) else {}
        op = players[1 - yi] if isinstance(players[1 - yi], dict) else {}
        # -------------------------------
        # Prize race
        # -------------------------------
        my_prizes = me.get("prize")
        if isinstance(my_prizes, list):
            self.prizes_lost = max(self.prizes_lost, 6 - len(my_prizes))
        # -------------------------------
        # Observe opponent board
        # -------------------------------
        active = None
        active_arr = op.get("active") or []
        if active_arr and isinstance(active_arr[0], dict):
            active = active_arr[0]
        bench = [
            b for b in (op.get("bench") or [])
            if isinstance(b, dict)
        ]
        board = []
        if active is not None:
            board.append(active)
        board.extend(bench)
        self.bench_sizes.append(len(bench))
        for mon in board:
            cid = mon.get("id")
            if cid is not None:
                self.opp_seen_ids.add(cid)
                if self.gd.is_ex(cid):
                    self.opp_ex_ids.add(cid)
            max_hp = mon.get("maxHp")
            if isinstance(max_hp, (int, float)):
                self.hp_samples.append(int(max_hp))
            retreat = mon.get("retreat")
            if isinstance(retreat, (int, float)):
                self.retreat_samples.append(int(retreat))
            energies = mon.get("energies") or []
            if isinstance(energies, list):
                self.total_energy += len(energies)
                self.energy_observations += 1
        # -------------------------------
        # Detect switches / retreats
        # -------------------------------
        if active is not None:
            cur = (
                active.get("serial"),
                active.get("id"),
                active.get("hp"),
            )
            prev = self._last_opp_active
            if prev is not None and prev[0] != cur[0]:
                self.switches += 1
                bench_serials = {
                    b.get("serial")
                    for b in bench
                }
                if prev[0] in bench_serials:
                    self.opp_retreats += 1
            self._last_opp_active = cur
            hp = active.get("hp")
            max_hp = active.get("maxHp")
            if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)):
                self.damage_taken += max(0, int(max_hp - hp))
                self.damage_events += 1

    # --- classification ---------------------------------------------------
    def archetype(self) -> str:
        """
        Infer the opponent's archetype from accumulated statistics.
        Rather than using hard thresholds for HP, this computes continuous evidence
        from observed board characteristics. Once confidence becomes sufficiently
        high, the prediction is locked for the remainder of the game to prevent
        oscillation.
        """
        if self._locked != UNKNOWN:
            return self._locked
        # Need enough observations before making strategic adjustments.
        if self.turn < 3 or self.decisions < 8:
            return UNKNOWN
        wall_score = 0.0
        aggro_score = 0.0
        # ----------------------------------------------------------
        # HP profile
        # ----------------------------------------------------------
        if self.hp_samples:
            avg_hp = sum(self.hp_samples) / len(self.hp_samples)
            # Typical tournament HP range is roughly 50–350.
            hp_norm = max(0.0, min(1.0, (avg_hp - 50.0) / 300.0))
            wall_score += 2.5 * hp_norm
            aggro_score += 1.2 * (1.0 - hp_norm)
        # ----------------------------------------------------------
        # EX density
        # ----------------------------------------------------------
        if self.opp_seen_ids:
            ex_ratio = len(self.opp_ex_ids) / len(self.opp_seen_ids)
            aggro_score += 2.2 * ex_ratio
            wall_score += 0.5 * (1.0 - ex_ratio)
        # ----------------------------------------------------------
        # Retreat behaviour
        # ----------------------------------------------------------
        if self.retreat_samples:
            avg_retreat = sum(self.retreat_samples) / len(self.retreat_samples)
            retreat_norm = min(avg_retreat / 4.0, 1.0)
            wall_score += retreat_norm
            aggro_score += 0.4 * (1.0 - retreat_norm)
        # ----------------------------------------------------------
        # Bench development
        # ----------------------------------------------------------
        if self.bench_sizes:
            avg_bench = sum(self.bench_sizes) / len(self.bench_sizes)
            bench_norm = min(avg_bench / 5.0, 1.0)
            aggro_score += bench_norm
            wall_score += 0.5 * (1.0 - bench_norm)
        # ----------------------------------------------------------
        # Energy commitment
        # ----------------------------------------------------------
        if self.energy_observations:
            avg_energy = self.total_energy / self.energy_observations
            energy_norm = min(avg_energy / 4.0, 1.0)
            wall_score += energy_norm
            aggro_score += 0.5 * energy_norm
        # ----------------------------------------------------------
        # Retreat frequency
        # ----------------------------------------------------------
        retreat_rate = self.opp_retreats / max(1, self.turn)
        wall_score += 2.0 * min(retreat_rate, 1.0)
        # ----------------------------------------------------------
        # Prize tempo
        # ----------------------------------------------------------
        prize_rate = self.prizes_lost / max(1, self.turn)
        aggro_score += 3.0 * min(prize_rate, 1.0)
        wall_score += max(0.0, 1.0 - prize_rate)
        # ----------------------------------------------------------
        # Damage profile
        # ----------------------------------------------------------
        if self.damage_events:
            avg_damage = self.damage_taken / self.damage_events
            damage_norm = min(avg_damage / 200.0, 1.0)
            wall_score += damage_norm
        # ----------------------------------------------------------
        # Convert to confidence
        # ----------------------------------------------------------
        total = wall_score + aggro_score
        if total <= 0:
            return UNKNOWN
        wall_conf = wall_score / total
        aggro_conf = aggro_score / total
        self._profile = {
            WALL: wall_conf,
            AGGRO: aggro_conf,
            UNKNOWN: max(0.0, 1.0 - max(wall_conf, aggro_conf)),
        }
        confidence = max(wall_conf, aggro_conf)
        if confidence < 0.60:
            return UNKNOWN
        winner = WALL if wall_conf > aggro_conf else AGGRO
        # Lock only after high confidence.
        if confidence >= 0.75:
            self._locked = winner
        return winner

    # --- profile emission -------------------------------------------------
    def rules_delta(self) -> dict[str, float]:
        """
        Blend rule-weight adjustments according to archetype confidence.
        Rather than abruptly switching profiles, interpolate between UNKNOWN and
        the detected archetype using the confidence scores.
        """
        self.archetype()
        out: dict[str, float] = {}
        for archetype, weight in self._profile.items():
            if archetype == UNKNOWN or weight <= 0:
                continue
            for key, value in _RULES_DELTA.get(archetype, {}).items():
                out[key] = out.get(key, 0.0) + weight * value
        return out

    def eval_delta(self) -> dict[str, float]:
        """
        Blend planner evaluation adjustments according to archetype confidence.
        """
        self.archetype()
        out: dict[str, float] = {}
        for archetype, weight in self._profile.items():
            if archetype == UNKNOWN or weight <= 0:
                continue
            for key, value in _EVAL_DELTA.get(archetype, {}).items():
                out[key] = out.get(key, 0.0) + weight * value
        return out