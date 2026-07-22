"""
search_rollout.py
Recursive search engine.
This module performs one complete search rollout.
Responsibilities
----------------
• Recursive search
• Cutoff handling
• Terminal handling
• Forced selections
• Expansion
• Backpropagation
This module deliberately contains NO Pokémon heuristics.
Planner owns orchestration.
SearchTree owns statistics.
SearchRollout owns recursion.
"""
from __future__ import annotations
class SearchRollout:
    def __init__(
        self,
        planner,
        tree,
        opponent_model,
    ):
        self.planner = planner
        self.tree = tree
        self.opponent_model = opponent_model

    # ---------------------------------------------------------
    def search(
        self,
        engine,
        search_state,
        me,
        depth,
        path,
    ):
        self.tree.apply_virtual_loss(path)
        try:
            node = self.planner._as_obs_dict(search_state)
            if node is None:
                return 0.0
            state = node.get("current")
            if not isinstance(state, dict):
                return 0.0
            key = self.planner._state_key(
                state,
                me,
            )
            if key is None:
                key = ("leaf", id(state))
            cached = self.tree.ctx.cache.search.get(
                (key, depth)
            )
            if cached is not None:
                self.tree.backup(
                    path,
                    cached,
                )
                return cached
            select = self.planner.extract_select(node)
            # ---------------------------------------------
            # Terminal
            # ---------------------------------------------
            terminal = self.check_terminal(
                state,
                me,
            )
            if terminal is not None:
                self.tree.ctx.cache.search[(key, depth)] = terminal
                self.tree.backup(
                    path,
                    terminal,
                )
                return terminal
            # ---------------------------------------------
            # Cutoff
            # ---------------------------------------------
            cutoff = self.cutoff(
                state,
                select,
                me,
                depth,
            )
            if cutoff is not None:
                self.tree.ctx.cache.search[(key, depth)] = cutoff
                self.tree.backup(
                    path,
                    cutoff,
                )
                return cutoff
            # ---------------------------------------------
            # Opponent
            # ---------------------------------------------
            value = self.opponent_model.evaluate_reply(
                engine,
                search_state,
                state,
                me,
                self.planner.opp_response,
            )
            if value is not None:
                self.tree.ctx.cache.search[(key, depth)] = value
                self.tree.backup(
                    path,
                    value,
                )
                return value
            # ---------------------------------------------
            # Forced selections
            # ---------------------------------------------
            if select.select_type != self.planner.SelectType.MAIN:
                return self.handle_forced(
                    engine,
                    search_state,
                    node,
                    state,
                    select,
                    me,
                    depth,
                    path,
                )
            # ---------------------------------------------
            # Expansion
            # ---------------------------------------------
            value = self.planner.expand_search(
                engine,
                search_state,
                node,
                state,
                select,
                me,
                depth,
                path,
            )
            self.tree.ctx.cache.search[(key, depth)] = value
            self.tree.backup(
                path,
                value,
            )
            return value
        finally:
            self.tree.revert_virtual_loss(path)

    # ---------------------------------------------------------
    def cutoff(
        self,
        state,
        select,
        me,
        depth,
    ):
        if (
            depth <= 0
            or self.tree.ctx.nodes >= self.planner.max_nodes
        ):
            score = self.planner._eval(
                state,
                me,
            )
            if (
                self.planner.adaptive is not None
                and self.planner.opponent_embedding is not None
            ):
                try:
                    strategy = type(
                        "Strategy",
                        (),
                        {
                            "comeback_mode": False,
                        },
                    )()
                    prediction = type(
                        "Prediction",
                        (),
                        {
                            "board_risk": 0.0,
                            "active_survival_probability": 1.0,
                            "knockout_probability": 0.0,
                        },
                    )()
                    adaptive = self.planner.adaptive.analyse(
                        state,
                        strategy,
                        prediction,
                    )
                    emb = self.planner.opponent_embedding
                    score += (
                        adaptive.aggression_shift
                        * emb[0]
                        * 25.0
                    )
                    score += (
                        adaptive.setup_shift
                        * emb[1]
                        * 20.0
                    )
                    score += (
                        adaptive.resource_shift
                        * emb[2]
                        * 15.0
                    )
                except Exception:
                    pass
            return score
        return None

    # ---------------------------------------------------------
    def check_terminal(
        self,
        state,
        me,
    ):
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
        return None

    # ---------------------------------------------------------
    def handle_forced(
        self,
        engine,
        search_state,
        node,
        state,
        select,
        me,
        depth,
        path,
    ):
        try:
            choice = self.planner.rules.choose(
                node,
                select,
                self.planner.gamedata,
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
            nxt = engine.search_step(
                search_state.searchId,
                choice,
            )
        except Exception:
            return self.planner._eval(
                state,
                me,
            )
        self.tree.ctx.nodes += 1
        return self.search(
            engine,
            nxt,
            me,
            depth,
            path,
        )