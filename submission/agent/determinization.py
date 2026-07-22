"""
determinization.py
Hidden-information reconstruction for planner search.
Responsibilities
----------------
• Build determinized game states
• Sample hidden decks
• Sample prize cards
• Sample opponent hand
• Build multiple determinizations
Contains NO search logic.
Contains NO evaluation logic.
Contains NO Pokémon decision making.
Planner simply asks:
    determinizations = determinizer.build(...)
"""
from __future__ import annotations
import random
from collections import Counter

class Determinizer:
    def __init__(
        self,
        gamedata,
        your_deck_ids,
        opponent_deck_ids=None,
        seed=0,
    ):
        self.gamedata = gamedata
        self.your_deck_ids = list(your_deck_ids or [])
        self.opponent_deck_ids = list(
            opponent_deck_ids
            or self.your_deck_ids
        )
        self.rng = random.Random(seed)

    # ---------------------------------------------------------
    def build(
        self,
        obs_dict,
        state,
        me,
        k=2,
    ):
        players = state.get("players") or []
        if len(players) < 2:
            return []
        my_player = players[me]
        opp_player = players[1 - me]
        select = obs_dict.get("select")
        deck_given = (
            isinstance(select, dict)
            and select.get("deck") is not None
        )
        counts = (
            int(my_player.get("deckCount") or 0),
            len(my_player.get("prize") or []),
            int(opp_player.get("deckCount") or 0),
            len(opp_player.get("prize") or []),
            int(opp_player.get("handCount") or 0),
        )
        fallback = (
            self.your_deck_ids[0]
            if self.your_deck_ids
            else 0
        )
        my_pool = self.build_my_pool(my_player)
        opp_active_needed = (
            bool(opp_player.get("active"))
            and opp_player["active"][0] is None
        )
        determinizations = []
        seen = set()
        attempts = max(
            k * 5,
            10,
        )
        while (
            len(determinizations) < k
            and attempts > 0
        ):
            attempts -= 1
            opp_pool = self.build_opponent_pool(
                opp_player,
            )
            det = self._sample_once(
                my_pool,
                opp_pool,
                counts,
                deck_given,
                fallback,
                opp_active_needed,
            )
            if det is None:
                continue
            signature = (
                tuple(det[0]),
                tuple(det[1]),
                tuple(det[2][:10]),
                tuple(det[3]),
            )
            if signature in seen:
                continue
            seen.add(signature)
            determinizations.append(det)
        return determinizations

    # ---------------------------------------------------------
    def build_opponent_pool(
        self,
        player,
    ):
        """
        Build the unseen opponent card pool.
        Removes every opponent card that is already publicly visible
        (active, bench, discard, attached cards, revealed hand, etc.)
        before sampling hidden cards.
        """
        unseen = Counter(self.opponent_deck_ids)
        for cid in visible_ids(player):
            if unseen.get(cid, 0):
                unseen[cid] -= 1
        pool = list(unseen.elements())
        self.rng.shuffle(pool)
        return pool

    # ---------------------------------------------------------
    def build_opponent_pool(self):
        pool = list(self.opponent_deck_ids)
        self.rng.shuffle(pool)
        return pool

    # ---------------------------------------------------------
    def _sample_once(
        self,
        my_pool_base,
        opponent_pool,
        counts,
        deck_given,
        fallback,
        opponent_active_needed,
    ):
        """
        Build exactly one determinization.
        Returns
        -------
        (
            your_deck,
            your_prizes,
            opponent_deck,
            opponent_prizes,
            opponent_hand,
            opponent_active,
        )
        or None if no legal opponent Active can be generated.
        """
        (
            my_deck_count,
            my_prize_count,
            opp_deck_count,
            opp_prize_count,
            opp_hand_count,
        ) = counts
        # ----------------------------
        # Our hidden cards
        # ----------------------------
        my_pool = list(my_pool_base)
        self.rng.shuffle(my_pool)
        if deck_given:
            your_deck = []
        else:
            your_deck = my_pool[:my_deck_count]
        your_prizes = my_pool[
            my_deck_count:
            my_deck_count + my_prize_count
        ]
        if not deck_given:
            your_deck = pad(
                your_deck,
                my_deck_count,
                fallback,
            )
        your_prizes = pad(
            your_prizes,
            my_prize_count,
            fallback,
        )
        # ----------------------------
        # Opponent hidden cards
        # ----------------------------
        opponent_deck = pad(
            opponent_pool[:opp_deck_count],
            opp_deck_count,
            fallback,
        )
        opponent_prizes = pad(
            opponent_pool[
                opp_deck_count:
                opp_deck_count + opp_prize_count
            ],
            opp_prize_count,
            fallback,
        )
        opponent_hand = pad(
            opponent_pool[
                opp_deck_count + opp_prize_count:
                opp_deck_count + opp_prize_count + opp_hand_count
            ],
            opp_hand_count,
            fallback,
        )
        # ----------------------------
        # Unknown opponent active
        # ----------------------------
        opponent_active = []
        if opponent_active_needed:
            basic = first_basic(
                opponent_hand + opponent_deck,
                self.gamedata,
            )
            if basic is None:
                return None
            opponent_active = [basic]
        return (
            your_deck,
            your_prizes,
            opponent_deck,
            opponent_prizes,
            opponent_hand,
            opponent_active,
        )

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def first_basic(ids, gamedata):
    for cid in ids:
        if gamedata.is_basic_pokemon(cid):
            return cid
    return None

def pad(lst, n, fallback):
    if n <= 0:
        return []
    out = list(lst)
    while len(out) < n:
        out.append(fallback)
    return out[:n]

def visible_ids(player):
    out = []
    for zone in (
        player.get("hand") or [],
        player.get("discard") or [],
    ):
        for card in zone:
            cid = card_id(card)
            if cid is not None:
                out.append(cid)
    for mon in (
        [active(player)]
        + list(player.get("bench") or [])
    ):
        if not isinstance(mon, dict):
            continue
        if isinstance(mon.get("id"), int):
            out.append(mon["id"])
        for key in (
            "energyCards",
            "tools",
            "preEvolution",
        ):
            for card in mon.get(key) or []:
                cid = card_id(card)
                if cid is not None:
                    out.append(cid)
    return out

def active(player):
    arr = player.get("active") or []
    if arr and isinstance(arr[0], dict):
        return arr[0]
    return None

def card_id(card):
    if not isinstance(card, dict):
        return None
    cid = card.get("id")
    if cid is None:
        cid = card.get("cardId")
    return cid if isinstance(cid, int) else None