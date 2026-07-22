"""
policy_prior.py
Move ordering and policy priors.
This module estimates the probability that each legal action is
worth exploring.
It deliberately does NOT choose actions.
Responsibilities
----------------
• Candidate pruning
• Prior probabilities
• Action scoring
• Progressive move ordering
Future:
    - PolicyNet
    - Opponent-specific priors
    - RL-trained priors
"""
from __future__ import annotations
import math
from . import rules
from .enums import OptionType

TYPE_PRIORITY = {
    OptionType.ABILITY: 7,
    OptionType.ATTACK: 6,
    OptionType.EVOLVE: 5,
    OptionType.ATTACH: 4,
    OptionType.PLAY: 3,
    OptionType.RETREAT: 2,
    OptionType.DISCARD: 1,
    OptionType.END: 0,
}

class PolicyPrior:
    def __init__(self, planner):
        self.planner = planner
        self.gamedata = planner.gamedata

    # ---------------------------------------------------------
    def candidate_actions(
        self,
        node,
        select,
        search_id=None,
    ):
        """
        Beam-search candidate pruning.
        We never explore every legal move.
        We keep only the strongest actions.
        """
        try:
            rules_pick = rules._choose_main(
                node,
                select,
                self.gamedata,
            )
        except Exception:
            rules_pick = None
        scored = []
        for option in select.options:
            score = self.score(
                option,
                rules_pick,
                search_id,
            )
            scored.append(
                (
                    score,
                    option,
                )
            )
        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )
        keep = []
        attacks = []
        end_action = None
        for _, option in scored:
            if option.type == OptionType.ATTACK:
                attacks.append(option)
                continue
            if option.type == OptionType.END:
                end_action = option
                continue
            if len(keep) < 8:
                keep.append(option)
        keep.extend(attacks)
        if not keep and end_action is not None:
            keep.append(end_action)
        seen = set()
        ordered = []
        for option in keep:
            if option.index in seen:
                continue
            seen.add(option.index)
            ordered.append(option)
        return ordered

    # ---------------------------------------------------------
    def score(
        self,
        option,
        rules_pick,
        search_id=None,
    ):
        score = 1.0
        score += TYPE_PRIORITY.get(
            option.type,
            0,
        )
        if option.index == rules_pick:
            score += 5.0
        if (
            search_id is not None
            and search_id in self.planner.tree.ctx.tree
        ):
            parent = self.planner.tree.node(search_id)
            child_id = parent.children.get(
                option.index
            )
            if child_id is not None:
                child = self.planner.tree.node(child_id)
                if child.visits:
                    score += min(
                        math.log1p(child.visits),
                        2.5,
                    )
                    score += 2.5 * math.tanh(
                        child.value / 75.0
                    )
        emb = getattr(
            self.planner,
            "opponent_embedding",
            None,
        )
        if emb is not None:
            attack_rate, ability_rate, item_rate = emb
            if (
                option.type == OptionType.ATTACK
                and attack_rate > 0.55
            ):
                score += 1.25
            elif (
                option.type == OptionType.ABILITY
                and ability_rate > 0.30
            ):
                score += 1.0
            elif (
                option.type == OptionType.PLAY
                and item_rate > 0.40
            ):
                score += 0.8
        return max(score, 0.01)

    # ---------------------------------------------------------
    def priors(
        self,
        node,
        select,
        search_id=None,
    ):
        """
        Build normalized policy priors.
        """
        try:
            rules_pick = rules._choose_main(
                node,
                select,
                self.gamedata,
            )
        except Exception:
            rules_pick = None
        scores = {}
        for option in select.options:
            score = self.score(
                option,
                rules_pick,
                search_id,
            )
            if option.type == OptionType.EVOLVE:
                score += 2.5
            elif option.type == OptionType.ATTACH:
                score += 2.0
            elif option.type == OptionType.RETREAT:
                score += 2.0
            elif option.type == OptionType.ATTACK:
                score += 3.0
            scores[option.index] = max(
                score,
                0.001,
            )
        return self.normalize(scores)

    # ---------------------------------------------------------
    @staticmethod
    def normalize(scores):
        total = sum(scores.values())
        if total <= 0:
            n = max(
                len(scores),
                1,
            )
            return {
                k: 1.0 / n
                for k in scores
            }
        return {
            k: v / total
            for k, v in scores.items()
        }