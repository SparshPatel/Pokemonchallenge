"""
search_expansion.py
Tree expansion utilities.
Responsibilities
----------------
• Node expansion
• Progressive widening
• PUCT child selection
• Engine stepping
• Root expansion
Contains NO evaluation logic.
Contains NO recursion.
Contains NO Pokémon strategy.
Planner/SearchRollout orchestrate.
"""
from __future__ import annotations
import math
class SearchExpansion:
    def __init__(
        self,
        planner,
        tree,
        policy_prior,
    ):
        self.planner = planner
        self.tree = tree
        self.policy = policy_prior

    # ---------------------------------------------------------
    def expand_node(
        self,
        search_id,
        node,
        select,
    ):
        stats = self.tree.node(search_id)
        if stats.expanded:
            return
        if select.select_type == self.planner.SelectType.MAIN:
            candidate_options = self.policy.candidate_actions(
                node,
                select,
                search_id,
            )
            candidate_select = self.planner.Select(
                options=candidate_options,
                min_count=select.min_count,
                max_count=select.max_count,
                select_type=select.select_type,
                context=select.context,
                remain_energy_cost=select.remain_energy_cost,
                remain_damage_counter=select.remain_damage_counter,
                deck=select.deck,
                raw=select.raw,
            )
        else:
            candidate_select = select
        priors = self.policy.priors(
            node,
            candidate_select,
            search_id,
        )
        if (
            self.planner.opponent_embedding is not None
            and priors
        ):
            emb = self.planner.opponent_embedding
            attack_bias = 1.0 + emb[0] * 0.15
            setup_bias = 1.0 + emb[1] * 0.10
            for option in candidate_select.options:
                if option.index not in priors:
                    continue
                try:
                    t = option.type.name
                except Exception:
                    t = str(option.type)
                if t == "ATTACK":
                    priors[option.index] *= attack_bias
                elif t in (
                    "PLAY",
                    "ATTACH",
                    "EVOLVE",
                    "ABILITY",
                ):
                    priors[option.index] *= setup_bias
            total = sum(priors.values())
            if total > 0:
                inv = 1.0 / total
                for action in priors:
                    priors[action] *= inv
        if (
            search_id
            == getattr(
                self.planner,
                "_root_search_id",
                None,
            )
            and len(priors) > 1
        ):
            self.add_dirichlet_noise(priors)
        stats.priors = priors
        stats.expanded = True

    # ---------------------------------------------------------
    def add_dirichlet_noise(
        self,
        priors,
    ):
        actions = list(priors.keys())
        noise = [
            self.planner.rng.gammavariate(
                self.planner.dirichlet_alpha,
                1.0,
            )
            for _ in actions
        ]
        total = sum(noise)
        if total <= 0:
            return
        noise = [
            n / total
            for n in noise
        ]
        eps = self.planner.dirichlet_epsilon
        for action, eta in zip(actions, noise):
            priors[action] = (
                (1.0 - eps)
                * priors[action]
                + eps * eta
            )

    # ---------------------------------------------------------
    def progressive_width(
        self,
        search_id,
        max_children,
    ):
        node = self.tree.node(search_id)
        width = int(
            self.planner.min_pw
            + node.visits ** self.planner.pw_alpha
        )
        return min(
            width,
            max_children,
        )

    # ---------------------------------------------------------
    def ucb(
        self,
        parent_id,
        child_id,
        prior,
    ):
        parent = self.tree.node(parent_id)
        child = self.tree.node(child_id)
        parent_visits = (
            parent.visits
            + parent.virtual_visits
        )
        child_visits = (
            child.visits
            + child.virtual_visits
        )
        if child.visits == 0:
            q = (
                parent.value
                - 0.35
                * self.planner.TERMINAL_WIN
            )
        else:
            q = child.value
        u = (
            self.planner.cpuct
            * prior
            * math.sqrt(parent_visits + 1)
            / (1 + child_visits)
        )
        return q + u

    # ---------------------------------------------------------
    def select_child(
        self,
        search_id,
    ):
        node = self.tree.node(search_id)
        best_action = None
        best_child = None
        best_score = float("-inf")
        for action, child_id in node.children.items():
            score = self.ucb(
                search_id,
                child_id,
                node.priors.get(action, 0.0),
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_id
        return best_action, best_child

    # ---------------------------------------------------------
    def step(
        self,
        engine,
        search_state,
        action,
    ):
        try:
            nxt = engine.search_step(
                search_state.searchId,
                [action],
            )
        except Exception:
            return None
        self.tree.ctx.nodes += 1
        self.tree.node(nxt.searchId)
        return nxt

    # ---------------------------------------------------------
    def expand_search(
        self,
        engine,
        search_state,
        node,
        state,
        select,
        me,
        depth,
        path,
        rollout,
    ):
        self.expand_node(
            search_state.searchId,
            node,
            select,
        )
        stats = self.tree.node(
            search_state.searchId,
        )
        priors = sorted(
            stats.priors.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        allowed = self.progressive_width(
            search_state.searchId,
            len(priors),
        )
        # Expand new child
        if len(stats.children) < allowed:
            for action, _ in priors:
                if action in stats.children:
                    continue
                nxt = self.step(
                    engine,
                    search_state,
                    action,
                )
                if nxt is None:
                    continue
                child_id = nxt.searchId
                stats.children[action] = child_id
                child_obs = self.planner._as_obs_dict(
                    nxt,
                )
                if child_obs is not None:
                    child_select = self.planner.extract_select(
                        child_obs,
                    )
                    if child_select is not None:
                        self.expand_node(
                            child_id,
                            child_obs,
                            child_select,
                        )
                return rollout.search(
                    engine,
                    nxt,
                    me,
                    depth - 1,
                    path + [child_id],
                )
        # Otherwise descend
        action, _ = self.select_child(
            search_state.searchId,
        )
        if action is None:
            return self.planner._eval(
                state,
                me,
            )
        nxt = self.step(
            engine,
            search_state,
            action,
        )
        if nxt is None:
            return self.planner._eval(
                state,
                me,
            )
        return rollout.search(
            engine,
            nxt,
            me,
            depth - 1,
            path + [nxt.searchId],
        )