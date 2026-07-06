"""Strong rule-based opponent ("lucario_v2"-style) for benchmarking.

This is a *genuinely strong* independent heuristic — the bar our submission must
clear, not another weak punching bag. It is self-contained (no shared ``agent``
package) but reads the real engine card/attack tables from ``cg.api`` so it can
reason about energy costs, attack damage and lethal range.

Design (independent of our submission's rules.py so the comparison is fair):

* **Lethal first** — if any affordable attack KOs the opponent's Active, take it.
* **Energy-need awareness** — attach Energy only to a Pokémon that still needs it
  to power its best attack, preferring the Active attacker, then a strong Bench
  sitter. Never over-attach.
* **Attack when ready** — once the Active can land a meaningful attack and no
  higher-value setup remains, attack; otherwise keep developing.
* **Board development** — use free abilities, evolve, bench Basics, dig with
  draw/search trainers in a sensible order.
* **Retreat sparingly** — only when the Active is in lethal range and a healthier
  attacker is available.

Stdlib + ``cg`` only. Crash-safe: any failure degrades to a legal choice.
"""
from __future__ import annotations

import os

# --- OptionType / SelectType / AreaType / CardType / EnergyType ints -------
O_NUMBER, O_YES, O_NO, O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY = 0, 1, 2, 3, 4, 5, 6
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK = 7, 8, 9, 10, 11, 12, 13
O_END, O_SKILL, O_SPECIAL = 14, 15, 16

S_MAIN, S_CARD, S_ENERGY, S_YES_NO = 0, 1, 4, 9

A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE = 2, 3, 4, 5, 6

CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM = 0, 1, 2, 3, 4
CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = 5, 6

E_COLORLESS = 0

# Acquire (take the best) vs discard (shed the worst) SelectContext ints.
ACQUIRE_CTX = {1, 2, 3, 4, 5, 6, 7, 18, 19, 21, 22}   # setup/switch/to_*/evolves/attach
DISCARD_CTX = {8, 9, 10, 11, 26, 27, 29, 30}           # discard/to_deck/to_prize/...


# ---------------------------------------------------------------------------
# Engine card/attack data (lazy, cached, crash-safe)
# ---------------------------------------------------------------------------
class _Data:
    _inst = None

    def __init__(self):
        self.ctype = {}          # cardId -> cardType
        self.basic = set()       # basic Pokémon ids
        self.ex = set()
        self.attacks = {}        # attackId -> (damage, energies list)
        self.card_attacks = {}   # cardId -> [attackId]
        self.ok = False

    @classmethod
    def get(cls):
        if cls._inst is None:
            d = cls()
            d._load()
            cls._inst = d
        return cls._inst

    def _load(self):
        try:
            from cg import api
        except Exception:
            return
        try:
            for a in api.all_attack():
                self.attacks[a.attackId] = (int(a.damage or 0),
                                            [int(e) for e in (a.energies or [])])
            for c in api.all_card_data():
                self.ctype[c.cardId] = int(c.cardType)
                if getattr(c, "basic", False) and int(c.cardType) == CT_POKEMON:
                    self.basic.add(c.cardId)
                if getattr(c, "ex", False) or getattr(c, "megaEx", False):
                    self.ex.add(c.cardId)
                self.card_attacks[c.cardId] = list(c.attacks or [])
            self.ok = True
        except Exception:
            pass

    def is_pokemon(self, cid): return self.ctype.get(cid) == CT_POKEMON
    def is_basic(self, cid): return cid in self.basic
    def is_supporter(self, cid): return self.ctype.get(cid) == CT_SUPPORTER
    def is_item(self, cid): return self.ctype.get(cid) == CT_ITEM
    def is_energy(self, cid): return self.ctype.get(cid) in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY)
    def attack(self, aid): return self.attacks.get(aid, (0, []))


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------
_DECK = None


def _deck():
    global _DECK
    if _DECK is not None:
        return _DECK
    ids = []
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.lstrip("-").isdigit():
                    ids.append(int(s))
    except Exception:
        pass
    _DECK = ids
    return ids


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------
def agent(obs):
    if not isinstance(obs, dict):
        return [0]
    sel = obs.get("select")
    if sel is None:
        return _deck()
    try:
        return _decide(obs, sel)
    except Exception:
        return _fallback(sel)


def _decide(obs, sel):
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return list(range(max(0, int(sel.get("minCount") or 0))))

    stype = sel.get("type")
    d = _Data.get()
    state = obs.get("current") if isinstance(obs, dict) else None
    yi = state.get("yourIndex", 0) if isinstance(state, dict) else 0

    if stype == S_MAIN:
        return _main(options, state, yi, d)
    if stype == S_YES_NO:
        for i, o in enumerate(options):
            if o.get("type") == O_YES:
                return [i]
        return [0]
    if stype == S_ENERGY:
        return _lowest(sel, n)
    return _cards(options, sel, state, yi, d)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def _main(options, state, yi, d):
    hand = _hand(state, yi)
    me = _player(state, yi)
    opp = _player(state, 1 - yi)
    opp_hp = _active_hp(opp)
    my_active = _active(me)
    bench_room = _bench_room(me)

    best_i, best_s = 0, float("-inf")
    for i, o in enumerate(options):
        s = _score(o, hand, me, my_active, opp_hp, bench_room, d)
        if s > best_s:
            best_s, best_i = s, i
    return [best_i]


def _score(o, hand, me, my_active, opp_hp, bench_room, d):
    t = o.get("type")

    if t == O_ATTACK:
        dmg, _ = d.attack(o.get("attackId"))
        if opp_hp is not None and dmg >= opp_hp > 0:
            return 100000 + dmg                    # lethal — always take
        # Attack ends the turn: rank below productive setup, above pass.
        return 400 + dmg * 0.5

    if t == O_ABILITY:
        return 1000                                # free value, do first

    if t == O_EVOLVE:
        return 800

    if t == O_ATTACH:
        # Only valuable if the target still needs energy for its best attack.
        target = _attach_target(o, me)
        if target is not None and not _needs_energy(target, d):
            return 60                              # already powered — low value
        return 700 if o.get("inPlayArea") == A_ACTIVE else 500

    if t == O_PLAY:
        cid = _card_at(hand, o.get("index"))
        if d.is_basic(cid):
            return 600 if bench_room else 80       # develop the bench
        if d.is_supporter(cid):
            return 520                             # draw / search engine
        if d.is_item(cid):
            return 480
        return 540                                 # tool / stadium / other

    if t == O_RETREAT:
        # Retreat only when the Active is about to die and we have a backup.
        if _active_in_danger(my_active, opp_hp) and _has_bench_attacker(me, d):
            return 300
        return 30

    if t == O_DISCARD:
        return 20
    if t == O_END:
        return -1000
    return 5


# ---------------------------------------------------------------------------
# CARD selects (setup / search / discard)
# ---------------------------------------------------------------------------
def _cards(options, sel, state, yi, d):
    ctx = sel.get("context")
    acquire = ctx in ACQUIRE_CTX
    deck = sel.get("deck")
    scored = []
    for i, o in enumerate(options):
        cid = _resolve_id(o, deck, state, yi)
        scored.append((_value(cid, d), i))
    scored.sort(reverse=acquire)

    n = len(options)
    mn = int(sel.get("minCount") or 0)
    mx = sel.get("maxCount")
    mx = n if mx is None else int(mx)
    k = (mx or mn) if acquire else mn
    k = max(mn, min(k, n))
    return sorted(i for _, i in scored[:k])


def _value(cid, d):
    if cid is None:
        return 1.0
    if d.is_basic(cid):
        dmg = max((d.attack(a)[0] for a in d.card_attacks.get(cid, [])), default=0)
        return 100.0 + dmg * 0.1 + (30.0 if cid in d.ex else 0.0)
    if d.is_pokemon(cid):
        return 70.0
    if d.is_supporter(cid):
        return 40.0
    if d.is_item(cid):
        return 30.0
    if d.is_energy(cid):
        return 15.0
    return 10.0


# ---------------------------------------------------------------------------
# Energy / readiness reasoning
# ---------------------------------------------------------------------------
def _needs_energy(mon, d):
    """True if ``mon`` cannot yet pay for its most expensive attack."""
    if not isinstance(mon, dict):
        return True
    attached = [int(e) for e in (mon.get("energies") or [])]
    costs = [d.attack(a)[1] for a in d.card_attacks.get(mon.get("id"), [])]
    if not costs:
        return False
    # If it can afford its single hardest attack, it does not need more.
    hardest = max(costs, key=len) if costs else []
    return not _can_pay(hardest, attached)


def _can_pay(cost, attached):
    """Can ``attached`` energies pay ``cost`` (colorless = any)?"""
    if len(attached) < len(cost):
        return False
    pool = list(attached)
    # Specific (non-colorless) requirements first.
    for need in cost:
        if need == E_COLORLESS:
            continue
        if need in pool:
            pool.remove(need)
        else:
            return False
    # Colorless can be paid by anything remaining.
    colorless = sum(1 for c in cost if c == E_COLORLESS)
    return len(pool) >= colorless


def _active_in_danger(active, opp_hp):
    return isinstance(active, dict) and isinstance(active.get("hp"), int) and active["hp"] <= 80


def _has_bench_attacker(me, d):
    for mon in (me.get("bench") or []):
        if isinstance(mon, dict) and d.card_attacks.get(mon.get("id")):
            return True
    return False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _player(state, idx):
    pl = state.get("players") if isinstance(state, dict) else None
    if isinstance(pl, list) and 0 <= idx < len(pl) and isinstance(pl[idx], dict):
        return pl[idx]
    return {}


def _hand(state, yi):
    return _player(state, yi).get("hand") or []


def _active(me):
    a = me.get("active") or []
    return a[0] if a and isinstance(a[0], dict) else None


def _active_hp(p):
    a = p.get("active") or []
    if a and isinstance(a[0], dict):
        return a[0].get("hp")
    return None


def _bench_room(me):
    bench = me.get("bench") or []
    return len(bench) < int(me.get("benchMax") or 5)


def _card_at(hand, idx):
    if idx is None or not isinstance(hand, list) or not (0 <= idx < len(hand)):
        return None
    e = hand[idx]
    return e.get("id") if isinstance(e, dict) else None


def _attach_target(o, me):
    area, idx = o.get("inPlayArea"), o.get("inPlayIndex")
    if idx is None:
        return None
    arr = (me.get("active") if area == A_ACTIVE else me.get("bench")) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx]
    return None


_AREA_FIELD = {A_HAND: "hand", A_BENCH: "bench", A_DISCARD: "discard",
               A_ACTIVE: "active", A_PRIZE: "prize"}


def _resolve_id(o, deck, state, yi):
    if o.get("cardId") is not None:
        return o.get("cardId")
    idx = o.get("index")
    if deck and idx is not None and 0 <= idx < len(deck):
        e = deck[idx]
        if isinstance(e, dict):
            return e.get("id")
    field = _AREA_FIELD.get(o.get("area"))
    if field is None or idx is None or not isinstance(state, dict):
        return None
    pi = o.get("playerIndex")
    pi = yi if pi is None else pi
    arr = _player(state, pi).get(field) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx].get("id")
    return None


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------
def _lowest(sel, n):
    mn = int(sel.get("minCount") or 0)
    mx = sel.get("maxCount")
    mx = n if mx is None else int(mx)
    k = max(mn, 1) if mx >= 1 else mn
    return list(range(min(k, n)))


def _fallback(sel):
    options = sel.get("option") or []
    n = len(options)
    mn = max(0, min(int(sel.get("minCount") or 0), n))
    if mn == 0 and sel.get("type") == S_MAIN and n >= 1:
        mn = 1
    return list(range(mn))
