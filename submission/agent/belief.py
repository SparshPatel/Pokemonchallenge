"""Opponent belief state for partially-observable search.

The engine hides the opponent's hand (``None``), deck (count only) and face-down
cards. PIMC search needs *complete* opponent states, so we maintain a posterior
over which known deck the opponent is playing and sample consistent
determinizations from it.

Key empirical finding from the competition (validated by other participants):
search grounded in a realistic opponent model beats naive search by ~5x;
filling hidden info with placeholders makes search *harmful*. So belief quality
is the lever that makes lookahead worthwhile.

This module is engine-agnostic: it consumes observed opponent Card IDs and the
known opponent card *count*, and produces sampled hidden states. Wiring those
samples into ``search_begin`` happens in :mod:`agent.pimc`.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class CandidateDeck:
    """A known archetype deck: 60 Card IDs and a human-readable label."""

    name: str
    card_ids: list[int]
    prior: float = 1.0

    @property
    def counts(self) -> Counter:
        return Counter(self.card_ids)


@dataclass
class Determinization:
    """A fully-specified guess of the opponent's hidden cards."""

    hand: list[int]
    deck: list[int]
    prize: list[int]
    active: int | None = None


@dataclass
class BeliefState:
    """Posterior over candidate decks, updated from observed opponent cards."""

    candidates: list[CandidateDeck]
    observed: Counter = field(default_factory=Counter)
    _log_weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._log_weights:
            total = sum(c.prior for c in self.candidates) or 1.0
            self._log_weights = [
                _safe_log(c.prior / total) for c in self.candidates
            ]

    # --- updating ---------------------------------------------------------
    def observe(self, card_id: int) -> None:
        """Register that the opponent revealed ``card_id`` (played/discarded)."""
        self.observed[card_id] += 1
        for i, cand in enumerate(self.candidates):
            # A deck is impossible if it cannot contain this many copies.
            if cand.counts.get(card_id, 0) < self.observed[card_id]:
                self._log_weights[i] = float("-inf")

    def posterior(self) -> list[float]:
        """Normalized probability over candidate decks (consistent ones only)."""
        m = max(self._log_weights) if self._log_weights else 0.0
        if m == float("-inf"):
            # All candidates ruled out — fall back to uniform.
            n = len(self.candidates)
            return [1.0 / n] * n if n else []
        exps = [
            (0.0 if w == float("-inf") else pow(2.718281828, w - m))
            for w in self._log_weights
        ]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    def most_likely(self) -> CandidateDeck | None:
        post = self.posterior()
        if not post:
            return None
        return self.candidates[max(range(len(post)), key=post.__getitem__)]

    # --- sampling ---------------------------------------------------------
    def sample_determinizations(
        self,
        k: int,
        opp_hand_count: int,
        opp_deck_count: int,
        opp_prize_count: int,
        rng: random.Random | None = None,
    ) -> list[Determinization]:
        """Draw ``k`` hidden-state guesses consistent with observed cards.

        Card *counts* must match the engine's reported counts exactly, which the
        caller supplies via ``opp_*_count``.
        """
        rng = rng or random.Random()
        post = self.posterior()
        out: list[Determinization] = []
        for _ in range(k):
            cand = self._weighted_pick(post, rng)
            if cand is None:
                continue
            out.append(
                self._determinize(
                    cand, opp_hand_count, opp_deck_count, opp_prize_count, rng
                )
            )
        return out

    def _weighted_pick(self, post, rng) -> CandidateDeck | None:
        if not self.candidates:
            return None
        r = rng.random()
        acc = 0.0
        for cand, p in zip(self.candidates, post):
            acc += p
            if r <= acc:
                return cand
        return self.candidates[-1]

    def _determinize(
        self, cand, hand_count, deck_count, prize_count, rng
    ) -> Determinization:
        remaining = list(cand.card_ids)
        # Remove cards we have already seen leave the deck.
        seen = self.observed.copy()
        kept: list[int] = []
        for cid in remaining:
            if seen.get(cid, 0) > 0:
                seen[cid] -= 1
            else:
                kept.append(cid)
        rng.shuffle(kept)
        hand = kept[:hand_count]
        rest = kept[hand_count:]
        prize = rest[:prize_count]
        deck = rest[prize_count : prize_count + deck_count]
        return Determinization(hand=hand, deck=deck, prize=prize)


def _safe_log(x: float) -> float:
    if x <= 0:
        return float("-inf")
    import math

    return math.log(x)
