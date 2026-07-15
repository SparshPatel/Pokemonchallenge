"""
opponent_model.py
Opponent simulation during search.
Purpose
-------
Provide a consistent interface for simulating the opponent's turn.
Today:
    Uses rule-based policy.
Future:
    • Archetype-specific opponent models
    • Learned opponent policy
    • Belief-aware opponent actions
    • Monte-Carlo opponent sampling
Planner never directly plays opponent actions.
"""
from __future__ import annotations
import time
from . import rules

class OpponentModel:
    def __init__(
        self,
        gamedata,
        planner,
        max_steps=40,
    ):
        self.gamedata = gamedata
        self.planner = planner
        self.max_steps = max_steps

    # ---------------------------------------------------------
    def evaluate_reply(
        self,
        engine,
        search_state,
        state,
        me,
        enabled,
    ):
        """
        Simulate opponent response.
        If disabled:
            evaluate immediately.
        If enabled:
            play opponent turn before evaluation.
        """
        acting = state.get("yourIndex", me)
        if acting == me:
            return None
        if not enabled:
            return self.planner._eval(
                state,
                me,
            )
        return self.drive_to_player(
            engine,
            search_state,
            me,
        )

    # ---------------------------------------------------------
    def drive_to_player(
        self,
        engine,
        search_state,
        me,
    ):
        """
        Play until our next MAIN action.
        """
        for _ in range(self.max_steps):
            if (
                time.monotonic()
                >= self.planner.tree.ctx.deadline
            ):
                break
            if (
                self.planner.tree.ctx.nodes
                >= self.planner.max_nodes
            ):
                break
            node = self.planner._as_obs_dict(search_state)
            if node is None:
                break
            state = node.get("current")
            if not isinstance(state, dict):
                break
            result = state.get(
                "result",
                -1,
            )
            if (
                isinstance(result, int)
                and result >= 0
            ):
                return self.planner._terminal_value(
                    result,
                    me,
                )
            select = self.planner.extract_select(node)
            if (
                select is None
                or not select.options
            ):
                return self.planner._eval(
                    state,
                    me,
                )
            acting = state.get(
                "yourIndex",
                me,
            )
            if (
                acting == me
                and select.select_type.name == "MAIN"
            ):
                return self.planner._eval(
                    state,
                    me,
                )
            try:
                choice = rules.choose(
                    node,
                    select,
                    self.gamedata,
                )
            except Exception:
                choice = list(
                    range(
                        max(
                            1,
                            select.min_count,
                        )
                    )
                )
            try:
                search_state = engine.search_step(
                    search_state.searchId,
                    choice,
                )
            except Exception:
                break
            self.planner.tree.ctx.nodes += 1
        node = self.planner._as_obs_dict(
            search_state,
        )
        state = (
            node.get("current")
            if node
            else None
        )
        if isinstance(state, dict):
            return self.planner._eval(
                state,
                me,
            )
        return 0.0