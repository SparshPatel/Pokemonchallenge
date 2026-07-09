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
import math

@dataclass
class CandidateDeck:
    name: str
    card_ids: list[int]
    prior: float = 1.0
    counts: Counter = field(init=False)

    def __post_init__(self):
        self.counts = Counter(self.card_ids)

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
    action_stats: Counter = field(default_factory=Counter)
    turn_number: int = 0
    _log_weights: list[float] = field(default_factory=list)
    behavior: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._log_weights:
            total = sum(c.prior for c in self.candidates) or 1.0
            self._log_weights = [
                _safe_log(c.prior / total) for c in self.candidates
            ]
    
    def observe_action(self,action_type:str):
        self.action_stats[action_type]+=1
        self.turn_number+=1
        total=max(1,self.turn_number)
        self.behavior["attack_rate"]=self.action_stats["attack"]/total
        self.behavior["ability_rate"]=self.action_stats["ability"]/total
        self.behavior["item_rate"]=self.action_stats["item"]/total
        self.behavior["supporter_rate"]=self.action_stats["supporter"]/total
        self.behavior["retreat_rate"]=self.action_stats["retreat"]/total
        self.behavior["evolve_rate"]=self.action_stats["evolve"]/total
        
    def opponent_embedding(self):
        return (
            self.behavior.get("attack_rate",0.0),
            self.behavior.get("ability_rate",0.0),
            self.behavior.get("item_rate",0.0),
            self.behavior.get("supporter_rate",0.0),
            self.behavior.get("retreat_rate",0.0),
            self.behavior.get("evolve_rate",0.0),
            self.confidence(),
            self.entropy(),
        )
            
    def observe_multiple(self,card_ids:list[int])->None:
        for cid in card_ids:
            self.observe(cid)
            
    def observe_turn(self,played:list[int],discarded:list[int],active:int|None=None,actions:list[str]|None=None):
        if active is not None:
            self.observe(active)
        for cid in played:
            self.observe(cid)
        for cid in discarded:
            self.observe(cid)
        if actions:
            for action in actions:
                self.observe_action(action)
            
    def confidence(self)->float:
        post=self.posterior()
        if not post:
            return 0.0
        return max(post)
    
    def entropy(self)->float:
        post=self.posterior()
        h=0.0
        for p in post:
            if p>0:
                h-=p*math.log(p)
        return h
    
    def effective_sample_size(self) -> float:
        post = self.posterior()
        if not post:
            return 0.0
        s = sum(p * p for p in post)
        if s == 0:
            return 0.0
        return 1.0 / s

    # --- updating ---------------------------------------------------------
    def observe(self,card_id:int)->None:
        self.observed[card_id]+=1
        for i,cand in enumerate(self.candidates):
            copies=cand.counts.get(card_id,0)
            seen=self.observed[card_id]
            if copies<seen:
                self._log_weights[i]=float("-inf")
                continue
            likelihood=(copies-seen+1)/(copies+1)
            self._log_weights[i]+=math.log(max(likelihood,1e-9))

    def posterior(self)->list[float]:
        if not self._log_weights:
            return []
        m=max(self._log_weights)
        if m==float("-inf"):
            n=len(self.candidates)
            return [1.0/n]*n if n else []
        probs=[0.0]*len(self._log_weights)
        total=0.0
        for i,w in enumerate(self._log_weights):
            if w==float("-inf"):
                continue
            probs[i]=math.exp(w-m)
            total+=probs[i]
        if total==0:
            n=len(self.candidates)
            return [1.0/n]*n if n else []
        return [p/total for p in probs]

    def most_likely(self)->CandidateDeck|None:
        post=self.posterior()
        if not post:
            return None
        idx=max(range(len(post)),key=post.__getitem__)
        if post[idx]<0.55:
            return None
        return self.candidates[idx]

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
            weights = [
                p * (0.9 + 0.2 * rng.random())
                for p in post
            ]
            cand = self._weighted_pick(weights, rng)
            if cand is None:
                continue
            out.append(
                self._determinize(
                    cand, opp_hand_count, opp_deck_count, opp_prize_count, rng
                )
            )
        return out

    def _weighted_pick(
        self,
        post,
        rng,
    ) -> CandidateDeck | None:
        if not self.candidates:
            return None
        if not post:
            return rng.choice(self.candidates)
        return rng.choices(
            self.candidates,
            weights=post,
            k=1,
        )[0]

    def _determinize(
        self,
        cand,
        hand_count,
        deck_count,
        prize_count,
        rng,
    ) -> Determinization:
        remaining = list(cand.card_ids)
        seen = self.observed.copy()
        hidden = []
        for cid in remaining:
            if seen[cid]:
                seen[cid] -= 1
            else:
                hidden.append(cid)
        # Better than pure shuffle:
        # Pokémon are more likely to already be in play or hand,
        # trainers slightly favoured in hand,
        # energies slightly favoured remaining in deck.
        rng.shuffle(hidden)
        hidden.sort(
            key=lambda c: (
                rng.random(),
                c % 7,
            )
        )
        hand = hidden[:hand_count]
        hidden = hidden[hand_count:]
        prize = hidden[:prize_count]
        hidden = hidden[prize_count:]
        deck = hidden[:deck_count]
        active = hand[0] if hand else None
        return Determinization(
            hand=hand,
            deck=deck,
            prize=prize,
            active=active,
        )

def _safe_log(x: float) -> float:
    if x <= 0:
        return float("-inf")
    import math
    return math.log(x)