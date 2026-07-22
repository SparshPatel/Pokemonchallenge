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
from .adapter import extract_select

class OpponentModel:
    def __init__(
        self,
        planner,
        tree,
        max_steps=40,
    ):
        self.planner = planner
        self.tree = tree
        self.gamedata = planner.gamedata
        self.max_steps = max_steps
        self.attack_turns = 0
        self.ability_turns = 0
        self.item_turns = 0
        self.total_turns = 0

    # ---------------------------------------------------------
    def observe(
        self,
        obs_dict,
    ):
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return
        select = extract_select(obs_dict)
        if select is None:
            return
        if not select.options:
            return
        self.total_turns += 1
        try:
            idx = rules._choose_main(
                obs_dict,
                select,
                self.gamedata,
            )
        except Exception:
            return
        chosen = None
        for option in select.options:
            if option.index == idx:
                chosen = option
                break
        if chosen is None:
            return
        t = chosen.type.name
        if t == "ATTACK":
            self.attack_turns += 1
        elif t == "ABILITY":
            self.ability_turns += 1
        elif t == "PLAY":
            self.item_turns += 1

    # ---------------------------------------------------------
    def embedding(
        self,
    ):
        total = max(
            1,
            self.total_turns,
        )
        return (
            self.attack_turns / total,
            self.ability_turns / total,
            self.item_turns / total,
        )

    # ---------------------------------------------------------
    def evaluate_reply(
        self,
        engine,
        search_state,
        state,
        me,
        enabled,
    ):
        acting = state.get(
            "yourIndex",
            me,
        )
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
        for _ in range(self.max_steps):
            if (
                time.monotonic()
                >= self.tree.ctx.deadline
            ):
                break
            if (
                self.tree.ctx.nodes
                >= self.planner.max_nodes
            ):
                break
            node = self.planner._as_obs_dict(
                search_state,
            )
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
            select = extract_select(
                node,
            )
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
            self.tree.ctx.nodes += 1
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