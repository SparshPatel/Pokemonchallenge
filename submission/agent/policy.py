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

from . import rules
from .adapter import Select
from .cards import CardStats
from .gamedata import GameData


class Policy:
    def __init__(
        self,
        card_stats_path: str,
        deck_ids: list[int] | None = None,
        time_budget_s: float = 8.0,
        enable_pimc: bool | None = None,
        enable_planner: bool | None = None,
        planner_think_s: float = 0.6,
    ):
        self.cards = CardStats.load(card_stats_path)
        self.gamedata = GameData.load()
        self.deck_ids = list(deck_ids) if deck_ids else []
        self.time_budget_s = time_budget_s
        # PIMC is OFF by default. Competition evidence shows naive/imperfect
        # search underperforms a solid rule-based policy (0.21-0.42 vs 0.68), so
        # the shipped agent is pure rules unless PIMC is explicitly enabled and
        # proven to beat rules in the gauntlet. Opt in via PTCG_ENABLE_PIMC=1.
        if enable_pimc is None:
            enable_pimc = os.environ.get("PTCG_ENABLE_PIMC", "0") == "1"
        self.enable_pimc = enable_pimc
        self._pimc = None
        self._pimc_tried = False

        # Planner: engine-simulated within-turn lookahead. Proven in the local
        # gauntlet (mirror 0.65 vs rules; +0.20 vs strong piloting a wall deck;
        # 0 errors in 750+ games), so it is ON by default in the competition
        # runtime. Set PTCG_ENABLE_PLANNER=0 to force pure rules.
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

    def choose(self, obs_dict, select: Select) -> list[int]:
        deadline = time.monotonic() + self.time_budget_s

        planner = self._maybe_planner()
        if planner is not None:
            try:
                choice = planner.choose(obs_dict, select, deadline)
                if choice:
                    return choice
            except Exception:
                pass  # fall through to PIMC / heuristic

        searcher = self._maybe_pimc()
        if searcher is not None:
            try:
                choice = searcher.choose(obs_dict, select, deadline)
                if choice:
                    return choice
            except Exception:
                pass  # fall through to heuristic

        return rules.choose(obs_dict, select, self.gamedata)
