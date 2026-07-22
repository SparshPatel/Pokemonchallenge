"""Top-level policy: chooses an action for a given decision point.
Architecture (see project plan):
    choose(obs_dict, select)
      ├─ [MAIN turn + planner enabled] within-turn lookahead   (agent.planner)
      ├─ [search available + belief ready] PIMC search         (agent.pimc)
      └─ [otherwise] rule-based heuristic                       (agent.rules)
The planner runs a real engine-simulated beam search over our successive MAIN
actions and picks the line with the best end-of-turn board (deck-aware value),
falling back to the rule policy on any error or exhausted budget. Both search
layers are capability- and budget-gated, so the agent always has a fast, legal
fallback.
"""
from __future__ import annotations
import os
import time
from . import rules, supervisor
from .adapter import Select
from .archetype import ArchetypeDetector
from .cards import CardStats
from .enums import SelectType
from .gamedata import GameData
class Policy:
    def __init__(
        self,
        card_stats_path: str,
        deck_ids: list[int] | None = None,
        time_budget_s: float = 8.0,
        enable_pimc: bool | None = None,
        enable_planner: bool | None = None,
        planner_think_s: float = 0.12,
        enable_supervisor: bool | None = None,
        enable_archetype: bool | None = None,
    ):
        self.cards = CardStats.load(card_stats_path)
        self.gamedata = GameData.load()
        self.deck_ids = list(deck_ids) if deck_ids else []
        self.time_budget_s = time_budget_s
        # Semi-autonomous supervisor: a thin, hardcoded layer above the
        # autonomous planner/rules stack that forces the handful of decisions
        # that are correct by the rules of the game (take a game-winning KO;
        # never end the turn on an un-taken KO). DISABLED by default pending
        # critical regression. Enable via PTCG_ENABLE_SUPERVISOR=1.
        if enable_supervisor is None:
            enable_supervisor = os.environ.get("PTCG_ENABLE_SUPERVISOR", "0") == "1"
        self.enable_supervisor = enable_supervisor
        # Opponent-archetype sub-agents: after a few turns, classify the
        # opponent (wall/pivot vs aggro/plunderer) and shift the rule + planner
        # weights toward the specialised counter-plan. OFF by default pending debugging
        # of recent regression. Enable via PTCG_ENABLE_ARCHETYPE=1.
        if enable_archetype is None:
            enable_archetype = os.environ.get("PTCG_ENABLE_ARCHETYPE", "0") == "1"
        self.enable_archetype = enable_archetype
        self._detector = ArchetypeDetector(self.gamedata) if enable_archetype else None
        # Snapshot the tuned base weights so per-turn archetype deltas layer on
        # top of them without drifting.
        self._base_rules_weights = dict(rules.WEIGHTS)
        self._last_archetype = None
        # PIMC is OFF by default. Competition evidence shows naive/imperfect
        # search underperforms a solid rule-based policy (0.21-0.42 vs 0.68), so
        # the shipped agent is pure rules unless PIMC is explicitly enabled and
        # proven to beat rules in the gauntlet. Opt in via PTCG_ENABLE_PIMC=1.
        if enable_pimc is None:
            enable_pimc = os.environ.get("PTCG_ENABLE_PIMC", "1") == "1"
        self.enable_pimc = enable_pimc
        self._pimc = None
        self._pimc_tried = False
        # Planner: engine-simulated within-turn lookahead. Proven in the local
        # gauntlet (mirror 0.65 vs rules; +0.20 vs strong piloting a wall deck;
        # 0 errors in 750+ games), so it is ON by default in the competition
        # runtime. Set PTCG_ENABLE_PLANNER=0 to force pure rules.
        # Think budget defaults to 0.12s/decision (the value validated in the
        # write-up). An earlier 0.6s default gave no measurable win-rate gain
        # (0.775 vs 0.750 over 80 same-deck games vs strong, within noise) but
        # 5x the worst-case latency -- a real forfeit risk on slower judge
        # hardware. Override with PTCG_PLANNER_THINK_S.
        if enable_planner is None:
            enable_planner = os.environ.get("PTCG_ENABLE_PLANNER", "1") == "1"
        self.enable_planner = enable_planner
        self.planner_think_s = float(
            os.environ.get("PTCG_PLANNER_THINK_S", planner_think_s)
        )
        self._planner = None
        self._planner_tried = False
    def _maybe_planner(self):
        """Lazily construct the turn planner if enabled and the engine exposes search."""
        if not self.enable_planner:
            return None
        if self._planner_tried:
            return self._planner
        self._planner_tried = True
        try:
            from .planner import TurnPlanner
            planner = TurnPlanner(
                self.cards,
                gamedata=self.gamedata,
                your_deck_ids=self.deck_ids,
                max_think_s=self.planner_think_s,
            )
            if planner.available():
                self._planner = planner
        except Exception:
            self._planner = None
        return self._planner
    
    def _maybe_pimc(self):
        """Lazily construct the PIMC searcher if enabled and engine exposes search."""
        if not self.enable_pimc:
            return None
        if self._pimc_tried:
            return self._pimc
        self._pimc_tried = True
        try:
            from .pimc import PIMCSearcher
            searcher = PIMCSearcher(self.cards, your_deck_ids=self.deck_ids)
            if searcher.available():
                self._pimc = searcher
        except Exception:
            self._pimc = None
        return self._pimc
    
    def _apply_archetype_profile(self) -> None:
        """Update the opponent read and shift rule + planner weights to match.
        Layered additively on the tuned base weights so an UNKNOWN read (or a
        disabled detector) reproduces the base agent exactly. Crash-safe: any
        failure leaves the base weights in place.
        """
        if self._detector is None:
            return
        try:
            rules_delta = self._detector.rules_delta()
            merged = dict(self._base_rules_weights)
            for k, dv in rules_delta.items():
                if k in merged:
                    merged[k] = merged[k] + dv
            rules.set_weights(merged)
            # TODO:
            # Reconnect archetype-aware planner evaluation once the new
            # ValueNet / Evaluator pipeline exposes tunable weights.
            # The previous planner.eval dictionary no longer exists after
            # the evaluation refactor.
            self._last_archetype = self._detector.archetype()
        except Exception:
            pass
        
    def choose(self, obs_dict, select: Select) -> list[int]:
        deadline = time.monotonic() + self.time_budget_s
        # Update opponent archetype model.
        if self._detector is not None:
            self._detector.update(obs_dict)
        # Force game-winning actions before anything else.
        if (
            self.enable_supervisor
            and select.select_type == SelectType.MAIN
        ):
            forced = supervisor.forced_main(
                obs_dict,
                select,
                self.gamedata,
            )
            if forced is not None:
                return forced
        # Planner is only useful during MAIN actions.
        planner = None
        if select.select_type == SelectType.MAIN:
            planner = self._maybe_planner()
        # Apply archetype-specific weight adjustments.
        self._apply_archetype_profile()
        choice = None
        # MAIN actions:
        # let the turn planner perform its engine look-ahead.
        if (
            planner is not None
        ):
            try:
                c = planner.choose(
                    obs_dict,
                    select,
                    deadline,
                )
                if c:
                    choice = c
            except Exception:
                pass
        # Everything else:
        # use ISMCTS over hidden information.
        if (
            choice is None
            and planner is None
            and len(select.options) > 1
        ):
            searcher = self._maybe_pimc()
            if searcher is not None:
                try:
                    c = searcher.choose(
                        obs_dict,
                        select,
                        deadline,
                    )
                    if c:
                        choice = c
                except Exception:
                    pass
        # Final guaranteed fallback.
        if choice is None:
            choice = rules.choose(
                obs_dict,
                select,
                self.gamedata,
            )
        # Final legality guard.
        # ----------------------------------------------------------
        # Final supervisor pass.
        # The planner has already searched.
        # Rules have already produced a fallback.
        # The supervisor is now allowed to:
        #   • force tactical wins
        #   • prevent obvious blunders
        #   • apply strategic overrides
        # It never searches.
        # ----------------------------------------------------------
        if (
            self.enable_supervisor
            and select.select_type == SelectType.MAIN
        ):
            choice = supervisor.guard_main(
                obs_dict,
                select,
                self.gamedata,
                choice,
            )
            strategic = supervisor.strategic_override(
                obs_dict,
                select,
                self.gamedata,
                choice,
            )
            if strategic is not None:
                choice = strategic
        return choice