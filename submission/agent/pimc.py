"""Partially-Observable Monte Carlo (PIMC) search via the cabt lookahead API.

The engine exposes a forward-search interface (cabt ``api`` / ``sim`` module)::

    to_observation_class(obs_dict) -> Observation
    search_begin(agent_observation, your_deck, your_prize,
                 opponent_deck, opponent_prize, opponent_hand,
                 opponent_active, manual_coin=False) -> SearchState
    search_step(search_id, select) -> result
    search_end()
    search_release(search_id)

Because the opponent's hidden cards (and our own face-down deck/prizes) must be
supplied as concrete Card IDs, we draw several *determinizations* from the
belief state, evaluate each candidate action under each determinization, and
average. This turns an imperfect-information decision into a set of
perfect-information rollouts.

The searcher is *capability-gated*: if the bundled ``cg`` package or its search
functions are unavailable, :meth:`available` returns ``False`` and the policy
falls back to the rule-based heuristic. This keeps the agent crash-safe before
the engine is wired in locally.

Key empirical finding (validated by other competitors): search grounded in a
realistic opponent model beats naive search ~5x; filling hidden info with
placeholders makes search *harmful*. Belief quality is therefore the lever.
"""
from __future__ import annotations

import dataclasses
import importlib
import random
import time
from collections import Counter

from . import rules
from .adapter import Select, current_state, extract_select, your_index
from .belief import BeliefState, CandidateDeck
from .cards import CardStats
from .gamedata import GameData

# Modules that may expose the search surface, in priority order.
_ENGINE_MODULES = ("cg.api", "cg.sim", "cg", "api", "sim")
_SEARCH_FUNCS = ("search_begin", "search_step", "search_release")


class PIMCSearcher:
    def __init__(
        self,
        cards: CardStats,
        your_deck_ids: list[int] | None = None,
        candidate_decks: list[CandidateDeck] | None = None,
        n_determinizations: int = 4,
        seed: int = 0,
        max_think_s: float = 0.4,
        max_candidates: int = 10,
    ):
        self.cards = cards
        self.gamedata = GameData.load()
        self.your_deck_ids = list(your_deck_ids or [])
        self.k = n_determinizations
        self.max_think_s = max_think_s
        self.max_candidates = max_candidates
        self.rng = random.Random(seed)
        self._engine = self._locate_engine()
        # Opponent belief: default to assuming a mirror of our own deck until a
        # richer archetype set is supplied. Posterior narrows as cards reveal.
        decks = candidate_decks or self._default_candidates()
        self.belief = BeliefState(candidates=decks) if decks else None

    # --- capability check -------------------------------------------------
    def _locate_engine(self):
        for name in _ENGINE_MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if all(hasattr(mod, fn) for fn in _SEARCH_FUNCS):
                return mod
        return None

    def available(self) -> bool:
        return self._engine is not None

    def _default_candidates(self) -> list[CandidateDeck]:
        if not self.your_deck_ids:
            return []
        return [CandidateDeck(name="mirror", card_ids=list(self.your_deck_ids))]

    # --- main entry -------------------------------------------------------
    def choose(self, obs_dict, select: Select, deadline: float) -> list[int] | None:
        """Return the best option index list, or ``None`` to defer to rules."""
        eng = self._engine
        if eng is None or not select.options:
            return None
        # Bound PIMC's own thinking time so it stays viable per-decision.
        deadline = min(deadline, time.monotonic() + self.max_think_s)

        candidates = self._candidate_actions(select)
        if len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]
        if not candidates:
            return None

        # Update belief from revealed opponent cards before sampling.
        self._update_belief(obs_dict)

        scores = {tuple(c): 0.0 for c in candidates}
        counts = {tuple(c): 0 for c in candidates}

        determinizations = self._build_determinizations(obs_dict)
        if not determinizations:
            return None

        for det in determinizations:
            if time.monotonic() >= deadline:
                break
            for action in candidates:
                if time.monotonic() >= deadline:
                    break
                value = self._evaluate(obs_dict, action, det, deadline)
                if value is None:
                    continue
                key = tuple(action)
                scores[key] += value
                counts[key] += 1

        scored = [
            (scores[k] / counts[k], list(k)) for k in scores if counts[k] > 0
        ]
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    # --- candidate enumeration -------------------------------------------
    def _candidate_actions(self, select: Select) -> list[list[int]]:
        idxs = [o.index for o in select.options]
        if select.min_count <= 1 and select.max_count <= 1:
            return [[i] for i in idxs]
        actions = [[i] for i in idxs if select.min_count <= 1]
        if select.min_count >= 1:
            actions.append(sorted(idxs[: max(1, select.min_count)]))
        return actions or [sorted(idxs[: max(1, select.min_count)])]

    # --- belief / determinization ----------------------------------------
    def _update_belief(self, obs_dict) -> None:
        """Feed any newly revealed opponent Card IDs into the belief posterior."""
        if self.belief is None:
            return
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return
        yi = your_index(obs_dict)
        opp = _opponent_player(state, yi)
        if not isinstance(opp, dict):
            return
        # Opponent's discard pile and in-play cards are public information.
        for cid in _visible_opponent_card_ids(opp):
            try:
                self.belief.observe(cid)
            except Exception:
                pass

    def _build_determinizations(self, obs_dict):
        """Sample hidden opponent states consistent with observations."""
        if self.belief is None:
            return [None]
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return [None]
        yi = your_index(obs_dict)
        opp = _opponent_player(state, yi)
        if not isinstance(opp, dict):
            return [None]
        hand_n = int(opp.get("handCount") or 0)
        deck_n = int(opp.get("deckCount") or 0)
        prize_n = len([p for p in (opp.get("prize") or []) if p is None]) or 6
        try:
            return self.belief.sample_determinizations(
                self.k, hand_n, deck_n, prize_n, self.rng
            ) or [None]
        except Exception:
            return [None]

    # --- engine rollout ---------------------------------------------------
    def _evaluate(self, obs_dict, action, determinization, deadline) -> float | None:
        """Roll the action forward via the real ``search_*`` API and score it.

        ``search_begin`` takes an Observation *object* (not a dict) plus
        count-matched predictions for every hidden zone; it returns a
        ``SearchState`` (``.observation``, ``.searchId``). We apply our candidate
        action, then drive both players with the rule policy until the game ends
        or a depth/time cap, and score the terminal state. Returns ``None`` on
        any engine error so the averaging loop simply skips the sample.
        """
        eng = self._engine
        try:
            obs = eng.to_observation_class(obs_dict)
        except Exception:
            return None
        if getattr(obs, "search_begin_input", None) is None:
            return None

        args = self._search_args(obs_dict, determinization)
        if args is None:
            return None
        your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active = args

        started = False
        try:
            ss = eng.search_begin(
                obs, your_deck, your_prize, opp_deck,
                opp_prize, opp_hand, opp_active, False,
            )
            started = True
            ss = eng.search_step(ss.searchId, list(action))
            return self._rollout(eng, ss, obs_dict, deadline)
        except Exception:
            return None
        finally:
            if started:
                try:
                    eng.search_end()
                except Exception:
                    pass

    def _rollout(self, eng, ss, root_obs_dict, deadline, max_depth: int = 48) -> float:
        """Play the determinized game forward with the rule policy for both
        sides until terminal or a cap, then score it."""
        yi = your_index(root_obs_dict)
        last = ss
        for _ in range(max_depth):
            if time.monotonic() >= deadline:
                break
            obs = ss.observation
            cur = getattr(obs, "current", None)
            if cur is None:
                break
            result = getattr(cur, "result", -1)
            if isinstance(result, int) and result >= 0:
                return 1.0 if result == yi else (0.0 if result == 2 else -1.0)
            sel = getattr(obs, "select", None)
            if sel is None:
                break
            try:
                odict = dataclasses.asdict(obs)
            except Exception:
                break
            select = extract_select(odict)
            if select is None or not select.options:
                choice = list(range(int(getattr(sel, "minCount", 0) or 0)))
            else:
                try:
                    choice = rules.choose(odict, select, self.gamedata)
                except Exception:
                    choice = list(range(max(1, int(getattr(sel, "minCount", 0) or 0))))
            try:
                ss = eng.search_step(ss.searchId, choice)
            except Exception:
                break
            last = ss
        return self._terminal_value(last, yi)

    def _terminal_value(self, ss, yi) -> float:
        """Heuristic value of a non-terminal rollout end: our prize progress."""
        obs = getattr(ss, "observation", None)
        cur = getattr(obs, "current", None)
        players = getattr(cur, "players", None) if cur is not None else None
        if not isinstance(players, list) or len(players) < 2:
            return 0.0
        me, opp = players[yi], players[1 - yi]
        my_left = sum(1 for p in (getattr(me, "prize", []) or []) if p is None)
        opp_left = sum(1 for p in (getattr(opp, "prize", []) or []) if p is None)
        # Fewer of OUR prizes left = closer to winning. Keep below terminal ±1.
        return (opp_left - my_left) / 10.0

    # --- hidden-state construction ---------------------------------------
    def _search_args(self, obs_dict, det):
        """Build the six count-matched hidden-zone predictions for search_begin,
        or ``None`` if a consistent assignment cannot be formed."""
        if det is None:
            return None
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return None
        yi = your_index(obs_dict)
        me = _player(state, yi)
        opp = _opponent_player(state, yi)

        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        deck_given = isinstance(sel, dict) and sel.get("deck") is not None

        my_deck_n = int(me.get("deckCount") or 0)
        my_prize_n = len(me.get("prize") or [])
        your_deck, your_prize = self._self_hidden(obs_dict, my_deck_n, my_prize_n)
        if deck_given:
            your_deck = []
        elif len(your_deck) < my_deck_n:
            return None
        if len(your_prize) < my_prize_n:
            return None

        opp_deck = _pad(list(det.deck), int(opp.get("deckCount") or 0), self.your_deck_ids)
        opp_prize = _pad(list(det.prize), len(opp.get("prize") or []), self.your_deck_ids)
        opp_hand = _pad(list(det.hand), int(opp.get("handCount") or 0), self.your_deck_ids)
        if opp_deck is None or opp_prize is None or opp_hand is None:
            return None

        opp_active = []
        active = opp.get("active") or []
        if active and active[0] is None:
            cid = det.active if det.active is not None else self._first_basic(det)
            if cid is None:
                return None
            opp_active = [cid]
        return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active

    def _self_hidden(self, obs_dict, deck_n: int, prize_n: int):
        """Determinize our own face-down deck and prizes from the registered
        deck: subtract everything we can see, then split the remainder."""
        if not self.your_deck_ids:
            return [], []
        unseen = Counter(self.your_deck_ids)
        state = current_state(obs_dict)
        if isinstance(state, dict):
            me = _player(state, your_index(obs_dict))
            for cid in _visible_self_card_ids(me):
                if unseen.get(cid, 0) > 0:
                    unseen[cid] -= 1
        pool = list(unseen.elements())
        self.rng.shuffle(pool)
        your_prize = pool[:prize_n]
        your_deck = pool[prize_n:]
        fb = self.your_deck_ids[0]
        while len(your_prize) < prize_n:
            your_prize.append(fb)
        while len(your_deck) < deck_n:
            your_deck.append(fb)
        return your_deck, your_prize

    def _first_basic(self, det):
        for cid in list(det.hand) + list(det.deck) + list(det.prize):
            if self.gamedata.is_basic_pokemon(cid):
                return cid
        return None


# --- state helpers --------------------------------------------------------
def _pad(lst: list[int], n: int, fallback: list[int]) -> list[int] | None:
    """Return ``lst`` trimmed/padded to exactly ``n`` ids, or ``None`` if it is
    too short and no fallback id is available to pad with."""
    if n <= 0:
        return []
    if len(lst) >= n:
        return lst[:n]
    if not fallback:
        return None
    out = list(lst)
    fb = fallback[0]
    while len(out) < n:
        out.append(fb)
    return out


def _players(state) -> list:
    pl = state.get("players") if isinstance(state, dict) else None
    return pl if isinstance(pl, list) else []


def _player(state, idx) -> dict:
    pl = _players(state)
    if 0 <= idx < len(pl) and isinstance(pl[idx], dict):
        return pl[idx]
    return {}


def _opponent_player(state, your_idx) -> dict:
    return _player(state, 1 - your_idx)


def _card_id(card) -> int | None:
    if isinstance(card, dict):
        cid = card.get("cardId")
        if cid is None:
            cid = card.get("id")
        return cid if isinstance(cid, int) else None
    return None


def _pokemon_card_ids(mon) -> list[int]:
    out: list[int] = []
    if not isinstance(mon, dict):
        return out
    cid = mon.get("id")
    if isinstance(cid, int):
        out.append(cid)
    for key in ("energyCards", "tools"):
        for c in mon.get(key) or []:
            v = _card_id(c)
            if v is not None:
                out.append(v)
    for c in mon.get("preEvolution") or []:
        v = _card_id(c)
        if v is not None:
            out.append(v)
    return out


def _visible_opponent_card_ids(opp) -> list[int]:
    """Public opponent cards: discard pile and everything in play."""
    out: list[int] = []
    for c in opp.get("discard") or []:
        v = _card_id(c)
        if v is not None:
            out.append(v)
    for area in ("active", "bench"):
        for mon in opp.get(area) or []:
            out.extend(_pokemon_card_ids(mon))
    return out


def _visible_self_card_ids(me) -> list[int]:
    """Cards we can see on our own side: hand + in play + discard."""
    out: list[int] = []
    for c in me.get("hand") or []:
        v = _card_id(c)
        if v is not None:
            out.append(v)
    for c in me.get("discard") or []:
        v = _card_id(c)
        if v is not None:
            out.append(v)
    for area in ("active", "bench"):
        for mon in me.get(area) or []:
            out.extend(_pokemon_card_ids(mon))
    return out
