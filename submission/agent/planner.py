"""Within-turn lookahead planner via the cabt persistent search tree.

The shipped :mod:`agent.rules` policy scores each option *in isolation* and plays
greedily, one micro-decision at a time. It cannot answer questions of the form
"if I play this Ball, fetch Riolu, evolve, attach, and *then* attack, do I win
the race?" — it just takes the highest-scored single option and hopes the rest of
the turn works out.

``TurnPlanner`` closes that gap. At one of our MAIN decisions it:

1. **Determinizes** the hidden state — our own face-down deck/prizes are
   reconstructed accurately from the known 60-card list; the opponent is filled
   from a belief archetype (mirror by default). Our-turn simulation depends
   mostly on *our* deck, which we know, so the determinization is high quality
   for the thing we are optimizing.
2. Opens a real engine search with ``search_begin`` and **branches** over our
   successive MAIN actions using ``search_step`` (the engine exposes a persistent
   search tree: every ``search_step`` returns a fresh ``searchId`` from its
   parent, so sibling actions can be explored from the same node). Forced
   sub-selections in between (which card to fetch/discard) are resolved with the
   rule policy.
3. Evaluates the **end-of-turn board** with a deck-aware value function (prize
   race, damage on the defender, set-up KOs, our attacker's readiness and
   safety) rather than a hand-tuned per-option proxy.

It is **anytime and self-limiting** (wall-clock deadline + node budget) and
**crash-safe**: any engine error or exhausted budget falls back to the rule
policy, and the search context is always torn down in a ``finally``. It is opt-in
via ``PTCG_ENABLE_PLANNER=1`` until it is proven to beat rules in the gauntlet.
"""
from __future__ import annotations

import dataclasses
import importlib
import random
import time
from collections import Counter

from . import rules
from .adapter import Select, current_state, extract_select, your_index
from .enums import CardType, OptionType, SelectType
from .gamedata import GameData

try:  # optional learned leaf value; absent numpy/weights => stays disabled
    from . import value_net as _value_net
except Exception:  # pragma: no cover
    _value_net = None

# Modules that may expose the search surface, in priority order.
_ENGINE_MODULES = ("cg.api", "cg", "api")
_SEARCH_FUNCS = ("search_begin", "search_step", "search_end", "to_observation_class")


# Leaf evaluation weights — board value from OUR perspective. Tuned so the prize
# race dominates, then damage progress, then tempo/board. Offline-trainable.
EVAL = {
    "prize": 120.0,      # per net prize still owed (opp_left - my_left)
    "opp_dmg": 70.0,     # fraction of the defender's HP we have removed
    "setup_ko": 45.0,    # we can KO the defender next turn (x its prize value)
    "my_hp": 22.0,       # our active's HP fraction (survivability)
    "my_ready": 18.0,    # our active can already pay for an attack
    "no_active": 220.0,  # penalty: we have no Active (we just lost it / about to)
    # Attacker quality: get the strongest attacker into the Active seat, fully
    # loaded. Counters the observed loss pattern of swinging a 10-dmg baby
    # (Riolu) into a 300+ HP wall while Koraidon ex / Mega Lucario ex sit
    # benched, and of firing an under-energised attack (Koraidon 2e/50 instead
    # of 3e/200). Both are bounded [0, weight] and modest vs prize/threat.
    "active_quality": 30.0,  # (active best_dmg / our best available best_dmg) * w
    "active_loaded": 16.0,   # (active affordable_dmg / active best_dmg) * w
    "bench": 12.0,       # per benched Pokemon (board development)
    "bench_ready": 15.0, # per benched Pokemon that can already pay for an attack
    "energy": 6.0,       # per Energy in play on our side (tempo)
    "opp_threat": 60.0,  # penalty: opponent can KO our active next turn (x prize)
    "hand": 2.0,         # per card in hand (resources)
    "win": 100_000.0,    # terminal win (loss = -win, draw = 0)
    # Bench-awareness: pivot/wall decks retreat before KOs — damage stays on
    # bench Pokémon and the eval must credit that progress.
    "opp_bench_dmg": 35.0,    # (maxHp-hp)/maxHp * prize_value per damaged bench
    "bench_setup_ko": 22.0,   # we can KO a bench target via gust (x prize_value)
}


class TurnPlanner:
    def __init__(
        self,
        cards,
        gamedata: GameData | None = None,
        your_deck_ids: list[int] | None = None,
        opponent_deck_ids: list[int] | None = None,
        max_think_s: float = 0.6,
        beam_width: int = 4,
        max_depth: int = 6,
        n_determinizations: int = 2,
        max_nodes: int = 2000,
        opp_response: bool | None = None,
        max_opp_steps: int = 40,
        seed: int = 0,
        eval_weights: dict | None = None,
        use_value_net: bool | None = None,
        value_net_path: str | None = None,
    ):
        import os

        self.cards = cards
        self.gamedata = gamedata or GameData.load()
        self.your_deck_ids = list(your_deck_ids or [])
        # Opponent belief pool for determinization; default to a mirror.
        self.opponent_deck_ids = list(opponent_deck_ids or self.your_deck_ids)
        self.max_think_s = max_think_s
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.k = n_determinizations
        self.max_nodes = max_nodes
        # Simulate one opponent-response turn (drive it with the rule policy) and
        # evaluate the board WE face on our next turn — gives the planner
        # defensive awareness. A/B on our deck (H2H 60g) showed it is neutral
        # (0.467, z=-0.52) yet ~3x slower, so it is OFF by default; the static
        # end-of-turn eval is both faster and no worse. Toggle for research via
        # PTCG_PLANNER_OPP=1.
        if opp_response is None:
            opp_response = os.environ.get("PTCG_PLANNER_OPP", "0") == "1"
        self.opp_response = opp_response
        self.max_opp_steps = max_opp_steps
        self.rng = random.Random(seed)
        # Leaf-eval weights: explicit override > env JSON (for baked-in tuned
        # weights, kept stdlib-only) > module default. Instance-local so the
        # offline trainer can evaluate candidate weight vectors in-process.
        self.eval = dict(EVAL)
        if eval_weights is None:
            raw = os.environ.get("PTCG_EVAL_WEIGHTS")
            if raw:
                try:
                    import json
                    src = raw
                    if os.path.isfile(raw):
                        with open(raw, "r", encoding="utf-8") as fh:
                            src = fh.read()
                    eval_weights = json.loads(src)
                except Exception:
                    eval_weights = None
        if eval_weights:
            for k, v in eval_weights.items():
                if k in self.eval:
                    try:
                        self.eval[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        # Optional learned leaf value. Adds a bounded ADDITIVE bonus on top of
        # the hand-tuned _eval (a tie-breaker) when a trained weight bundle is
        # present AND enabled (param > PTCG_VALUE_NET env; default OFF until it
        # beats hand-eval in an A/B). A full replacement lost the hand-eval's
        # fine-grained within-turn ordering, so the net only nudges. Terminal
        # win/loss values are untouched so they still dominate.
        # P(win) in (0,1) maps to (_vn_scale * (p - 0.5)); _vn_scale is kept
        # small vs the prize weight (120) so the net breaks ties, not decisions,
        # and stays << the terminal win value (self.eval["win"] = 1e5).
        self.value_net = None
        self._vn_scale = 60.0
        if use_value_net is None:
            use_value_net = os.environ.get("PTCG_VALUE_NET", "0") == "1"
        if use_value_net and _value_net is not None:
            try:
                net = _value_net.ValueNet(value_net_path)
                if net.available:
                    self.value_net = net
            except Exception:
                self.value_net = None
        self._engine = self._locate_engine()
        self._deadline = 0.0
        self._nodes = 0

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

    # --- main entry -------------------------------------------------------
    def choose(self, obs_dict, select: Select, deadline: float) -> list[int] | None:
        """Best first MAIN action for this decision, or ``None`` to defer to rules."""
        eng = self._engine
        if eng is None or not select.options:
            return None
        # Plan only our own MAIN turn decisions; everything else (card picks,
        # forced sub-selections) is fast and handled by the rule policy.
        if select.select_type != SelectType.MAIN:
            return None
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return None
        me = your_index(obs_dict)
        if not isinstance(state.get("turn"), int) or state.get("turn", 0) < 1:
            return None

        try:
            obs_cls = eng.to_observation_class(obs_dict)
        except Exception:
            return None
        if getattr(obs_cls, "search_begin_input", None) is None:
            return None

        dets = self._build_determinizations(obs_dict, state, me)
        if not dets:
            return None

        self._deadline = min(deadline, time.monotonic() + self.max_think_s)

        agg: dict[int, list[float]] = {}
        for det in dets:
            if time.monotonic() >= self._deadline:
                break
            first_vals = self._search_one(eng, obs_cls, det, obs_dict, me)
            if first_vals:
                for a, v in first_vals.items():
                    agg.setdefault(a, []).append(v)

        if not agg:
            return None
        best_a, best_v = None, float("-inf")
        for a, vs in agg.items():
            m = sum(vs) / len(vs)
            if m > best_v:
                best_v, best_a = m, a
        return [best_a] if best_a is not None else None

    def _search_one(self, eng, obs_cls, det, obs_dict, me) -> dict[int, float] | None:
        """Run one determinized search; return {first_option_index: value}."""
        try:
            ss = eng.search_begin(obs_cls, *det, False)
        except Exception:
            return None
        self._nodes = 0
        try:
            return self._expand_root(eng, ss, me)
        except Exception:
            return None
        finally:
            try:
                eng.search_end()
            except Exception:
                pass

    # --- search tree ------------------------------------------------------
    def _expand_root(self, eng, root_ss, me) -> dict[int, float]:
        """Branch over each candidate first MAIN action; recurse for its value."""
        node = _as_obs_dict(root_ss)
        if node is None:
            return {}
        select = extract_select(node)
        if select is None or not select.options:
            return {}
        candidates = self._candidate_main_options(node, select)
        out: dict[int, float] = {}
        root_id = root_ss.searchId
        for idx in candidates:
            if time.monotonic() >= self._deadline or self._nodes >= self.max_nodes:
                break
            try:
                child = eng.search_step(root_id, [idx])
            except Exception:
                continue
            self._nodes += 1
            out[idx] = self._plan(eng, child, me, self.max_depth - 1)
        return out

    def _plan(self, eng, ss, me, depth) -> float:
        """Value of the position at ``ss`` for player ``me`` (our-turn search)."""
        node = _as_obs_dict(ss)
        if node is None:
            return 0.0
        state = node.get("current")
        if not isinstance(state, dict):
            return 0.0
        result = state.get("result", -1)
        if isinstance(result, int) and result >= 0:
            return self._terminal_value(result, me)

        select = extract_select(node)
        if select is None or not select.options:
            return self._eval(state, me)

        acting = state.get("yourIndex", me)
        if acting != me:
            # Our turn is over. Optionally simulate the opponent's reply (driven
            # by the rule policy) up to our next turn and evaluate the board we
            # then face — otherwise value the board we are leaving behind.
            if self.opp_response:
                return self._drive_to_my_turn(eng, ss, me)
            return self._eval(state, me)

        if select.select_type != SelectType.MAIN:
            # Forced sub-selection (fetch/discard/place): resolve with rules and
            # continue within the same MAIN ply (does not consume search depth).
            try:
                choice = rules.choose(node, select, self.gamedata)
            except Exception:
                choice = list(range(max(1, select.min_count)))
            if time.monotonic() >= self._deadline or self._nodes >= self.max_nodes:
                return self._eval(state, me)
            try:
                nxt = eng.search_step(ss.searchId, choice)
            except Exception:
                return self._eval(state, me)
            self._nodes += 1
            return self._plan(eng, nxt, me, depth)

        if depth <= 0 or time.monotonic() >= self._deadline or self._nodes >= self.max_nodes:
            return self._eval(state, me)

        candidates = self._candidate_main_options(node, select)
        best = float("-inf")
        for idx in candidates:
            if time.monotonic() >= self._deadline or self._nodes >= self.max_nodes:
                break
            try:
                child = eng.search_step(ss.searchId, [idx])
            except Exception:
                continue
            self._nodes += 1
            v = self._plan(eng, child, me, depth - 1)
            if v > best:
                best = v
        if best == float("-inf"):
            return self._eval(state, me)
        return best

    def _drive_to_my_turn(self, eng, ss, me) -> float:
        """Drive every decision with the rule policy until it is our next MAIN
        turn, the game ends, or the budget is exhausted; evaluate there.

        This plays out the opponent's whole reply (and any forced choice we must
        make in response, e.g. promoting a new Active after a KO) on the
        determinized board, so the returned value reflects the position we will
        actually face — capturing whether our end-of-turn board survives.
        """
        for _ in range(self.max_opp_steps):
            if time.monotonic() >= self._deadline or self._nodes >= self.max_nodes:
                break
            node = _as_obs_dict(ss)
            if node is None:
                break
            state = node.get("current")
            if not isinstance(state, dict):
                break
            result = state.get("result", -1)
            if isinstance(result, int) and result >= 0:
                return self._terminal_value(result, me)
            select = extract_select(node)
            if select is None or not select.options:
                return self._eval(state, me)
            acting = state.get("yourIndex", me)
            # Reached our next turn: evaluate the board the opponent handed back.
            if acting == me and select.select_type == SelectType.MAIN:
                return self._eval(state, me)
            try:
                choice = rules.choose(node, select, self.gamedata)
            except Exception:
                choice = list(range(max(1, select.min_count)))
            try:
                ss = eng.search_step(ss.searchId, choice)
            except Exception:
                break
            self._nodes += 1
        node = _as_obs_dict(ss)
        state = node.get("current") if node else None
        if isinstance(state, dict):
            return self._eval(state, me)
        return 0.0

    def _terminal_value(self, result: int, me: int) -> float:
        if result == me:
            return self.eval["win"]
        if result == 2:  # draw
            return 0.0
        return -self.eval["win"]

    # --- candidate ordering ----------------------------------------------
    _TYPE_PRIORITY = {
        OptionType.ABILITY: 7,
        OptionType.ATTACK: 6,
        OptionType.EVOLVE: 5,
        OptionType.ATTACH: 4,
        OptionType.PLAY: 3,
        OptionType.RETREAT: 2,
        OptionType.DISCARD: 1,
        OptionType.END: 0,
    }

    def _candidate_main_options(self, node, select: Select) -> list[int]:
        """Ordered, de-duplicated beam of MAIN option indices to explore.

        Always includes the rule policy's own pick first (so the planner can
        never do worse than evaluating the greedy line), then the highest-
        priority distinct options up to ``beam_width``.

        Boss's Orders / Prime Catcher are injected into the beam when the
        opponent has any bench Pokémon — regardless of TYPE_PRIORITY — because
        the gust+attack combo is the primary counter to retreat-heavy decks and
        PLAY=3 priority would otherwise keep it out of a typical mid-game beam.
        """
        try:
            rules_pick = rules._choose_main(node, select, self.gamedata)
        except Exception:
            rules_pick = select.options[0].index
        ordered = [rules_pick]

        # Detect whether opponent has a bench (gust is worth exploring).
        state = node.get("current") if isinstance(node, dict) else None
        me = state.get("yourIndex", 0) if isinstance(state, dict) else 0
        players = state.get("players") or [] if isinstance(state, dict) else []
        opp = players[1 - me] if len(players) > 1 and isinstance(players[1 - me], dict) else {}
        opp_has_bench = bool(opp.get("bench"))

        # GUST_IDS: Boss's Orders and Prime Catcher — high-value PLAY options
        # that open the gust+attack line; inject unconditionally when opp has bench.
        _GUST_IDS = (1182, 1088)
        if opp_has_bench:
            hand = (players[me].get("hand") or []) if len(players) > me and isinstance(players[me], dict) else []
            for opt in select.options:
                if opt.index in ordered:
                    continue
                cid = opt.card_id
                if cid is None and opt.hand_index is not None and 0 <= opt.hand_index < len(hand):
                    hc = hand[opt.hand_index]
                    cid = hc.get("id") if isinstance(hc, dict) else None
                if cid in _GUST_IDS:
                    ordered.append(opt.index)
                    break  # one gust candidate is enough

        rest = sorted(
            select.options,
            key=lambda o: self._TYPE_PRIORITY.get(o.type, 0),
            reverse=True,
        )
        for o in rest:
            if o.index not in ordered:
                ordered.append(o.index)
            if len(ordered) >= self.beam_width:
                break
        return ordered[: self.beam_width]

    # --- leaf evaluation --------------------------------------------------
    def _eval(self, state: dict, me: int) -> float:
        # Learned leaf value (if enabled + loaded). Used as a bounded ADDITIVE
        # bonus on top of the hand-eval (a tie-breaker), not a replacement —
        # a full replacement lost the hand-eval's fine-grained within-turn
        # ordering. Falls through cleanly on any error.
        vn_bonus = 0.0
        if self.value_net is not None and _value_net is not None:
            try:
                feats = _value_net.extract_features(
                    state, me, self.gamedata, _HELPERS
                )
                p = self.value_net.predict(feats)
                vn_bonus = self._vn_scale * (p - 0.5)
            except Exception:
                vn_bonus = 0.0
        gd = self.gamedata
        ev = self.eval
        players = state.get("players") or []
        if len(players) < 2:
            return 0.0
        mp = players[me] if isinstance(players[me], dict) else {}
        op = players[1 - me] if isinstance(players[1 - me], dict) else {}

        score = 0.0
        # Prize race: fewer of OUR prizes still owed = closer to winning.
        my_left = _prizes_left(mp)
        opp_left = _prizes_left(op)
        score += (opp_left - my_left) * ev["prize"]

        my_act = _active(mp)
        opp_act = _active(op)

        # Damage on the defender + set-up KO potential.
        if opp_act:
            maxhp = opp_act.get("maxHp") or 0
            hp = opp_act.get("hp") or 0
            if maxhp > 0:
                score += ((maxhp - hp) / maxhp) * ev["opp_dmg"]
            if my_act and hp > 0:
                dmg = _best_affordable_dmg(my_act, opp_act, gd)
                if dmg >= hp:
                    score += ev["setup_ko"] * gd.prize_value(opp_act.get("id"))

        # Our active: survivability, readiness, and the threat against it.
        if my_act:
            maxhp = my_act.get("maxHp") or 0
            hp = my_act.get("hp") or 0
            if maxhp > 0:
                score += (hp / maxhp) * ev["my_hp"]
            if _can_attack(my_act, gd):
                score += ev["my_ready"]
            if opp_act and hp > 0:
                othreat = _best_affordable_dmg(opp_act, my_act, gd)
                if othreat >= hp:
                    score -= ev["opp_threat"] * gd.prize_value(my_act.get("id"))
        else:
            score -= ev["no_active"]

        # Attacker quality: prefer our strongest available attacker in the
        # Active seat, fully loaded. Nudges the planner to promote/retreat to a
        # real attacker instead of swinging a weak baby, and to charge an
        # attacker to its biggest attack instead of firing an under-energised
        # one. Both terms are bounded and only compare within OUR own board.
        if my_act:
            best_pot = 0
            for p in [my_act] + list(mp.get("bench") or []):
                if isinstance(p, dict):
                    best_pot = max(best_pot, gd.best_damage(p.get("id")))
            act_pot = gd.best_damage(my_act.get("id"))
            if best_pot > 0:
                score += (act_pot / best_pot) * ev["active_quality"]
            if act_pot > 0:
                aff = _best_affordable_dmg(my_act, opp_act or {}, gd)
                score += min(aff / act_pot, 1.0) * ev["active_loaded"]

        # Board development and tempo.
        score += len(mp.get("bench") or []) * ev["bench"]
        score += _energy_in_play(mp) * ev["energy"]
        score += int(mp.get("handCount") or 0) * ev["hand"]

        # Powered bench attackers: each bench Pokémon that can already pay for
        # an attack is worth significantly more than an unpowered one — it can
        # immediately threaten after a KO, pivot, or gust.
        for bp in (mp.get("bench") or []):
            if isinstance(bp, dict) and _can_attack(bp, gd):
                score += ev["bench_ready"]

        # Bench-awareness: pivot/wall decks retreat wounded Pokémon before KOs,
        # so the opponent's active always looks "fresh" — but the damage stays on
        # bench Pokémon. Credit that progress so the planner values spreading
        # damage and playing Boss's Orders to drag up weakened targets.
        for bp in (op.get("bench") or []):
            if not isinstance(bp, dict):
                continue
            maxhp = bp.get("maxHp") or 0
            hp = bp.get("hp") or 0
            pv = gd.prize_value(bp.get("id"))
            if maxhp > 0 and hp < maxhp:
                score += ((maxhp - hp) / maxhp) * pv * ev["opp_bench_dmg"]
            # Can we KO this bench target now (after gusting it Active)?
            if my_act and 0 < hp:
                dmg = _best_affordable_dmg(my_act, bp, gd)
                if dmg >= hp:
                    score += ev["bench_setup_ko"] * pv

        return score + vn_bonus

    # --- determinization --------------------------------------------------
    def _build_determinizations(self, obs_dict, state, me) -> list[tuple]:
        """Build ``k`` count-matched hidden-state predictions for search_begin."""
        players = state.get("players") or []
        if len(players) < 2:
            return []
        mp = players[me] if isinstance(players[me], dict) else {}
        op = players[1 - me] if isinstance(players[1 - me], dict) else {}

        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        deck_given = isinstance(sel, dict) and sel.get("deck") is not None

        my_deck_n = int(mp.get("deckCount") or 0)
        my_prize_n = len(mp.get("prize") or [])
        opp_deck_n = int(op.get("deckCount") or 0)
        opp_prize_n = len(op.get("prize") or [])
        opp_hand_n = int(op.get("handCount") or 0)

        # Our own unseen pool = decklist minus everything we can see.
        unseen = Counter(self.your_deck_ids)
        for cid in _visible_ids(mp):
            if unseen.get(cid, 0) > 0:
                unseen[cid] -= 1
        my_pool_base = list(unseen.elements())

        # Opponent facedown active (if any) must be a Basic Pokemon id.
        opp_active_needed = False
        oa = op.get("active") or []
        if oa and oa[0] is None:
            opp_active_needed = True

        fb = self.your_deck_ids[0] if self.your_deck_ids else 0
        dets: list[tuple] = []
        for _ in range(self.k):
            my_pool = list(my_pool_base)
            self.rng.shuffle(my_pool)
            if deck_given:
                your_deck: list[int] = []
            else:
                your_deck = my_pool[:my_deck_n]
            your_prize = my_pool[my_deck_n:my_deck_n + my_prize_n]
            your_deck = _pad(your_deck, my_deck_n, fb) if not deck_given else []
            your_prize = _pad(your_prize, my_prize_n, fb)

            opp_pool = list(self.opponent_deck_ids)
            self.rng.shuffle(opp_pool)
            opp_deck = _pad(opp_pool[:opp_deck_n], opp_deck_n, fb)
            opp_prize = _pad(opp_pool[opp_deck_n:opp_deck_n + opp_prize_n], opp_prize_n, fb)
            opp_hand = _pad(
                opp_pool[opp_deck_n + opp_prize_n:opp_deck_n + opp_prize_n + opp_hand_n],
                opp_hand_n, fb,
            )
            opp_active: list[int] = []
            if opp_active_needed:
                cid = self._first_basic(opp_hand + opp_deck)
                if cid is None:
                    continue
                opp_active = [cid]

            if your_prize is None or opp_deck is None or opp_prize is None or opp_hand is None:
                continue
            dets.append((your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active))
        return dets

    def _first_basic(self, ids):
        for cid in ids:
            if self.gamedata.is_basic_pokemon(cid):
                return cid
        return None


# --- module helpers -------------------------------------------------------
def _as_obs_dict(ss):
    """Convert a SearchState's Observation dataclass to a plain dict."""
    obs = getattr(ss, "observation", None)
    if obs is None:
        return None
    try:
        return dataclasses.asdict(obs)
    except Exception:
        return None


def _active(player: dict) -> dict | None:
    arr = player.get("active") or []
    if arr and isinstance(arr[0], dict):
        return arr[0]
    return None


def _prizes_left(player: dict) -> int:
    """Number of prizes this player still has to take (i.e. is still owed).

    Counting remaining entries in the prize array — NOT the number of face-down
    (None) entries — because ``search_begin`` reveals the prizes we predict, so
    in a search state every prize is a known card (``none == 0``). The array
    itself shrinks as prizes are taken (verified: 6 -> 5 -> 3), so ``len`` is the
    correct, representation-independent measure in both real and search states.
    """
    prize = player.get("prize")
    return len(prize) if isinstance(prize, list) else 6


def _energy_in_play(player: dict) -> int:
    total = 0
    for mon in [_active(player)] + list(player.get("bench") or []):
        if isinstance(mon, dict):
            total += len(mon.get("energies") or [])
    return total


def _best_affordable_dmg(attacker: dict, defender: dict, gd: GameData) -> int:
    cid = attacker.get("id")
    if cid is None:
        return 0
    attached = attacker.get("energies") or []
    def_id = defender.get("id") if isinstance(defender, dict) else None
    best = 0
    for aid in gd.card_attacks.get(cid, []):
        if gd.can_pay(gd.attack_cost(aid), attached):
            dmg = gd.attack_damage(aid)
            if def_id is not None:
                dmg = gd.effective_damage(cid, dmg, def_id)
            best = max(best, dmg)
    return best


def _can_attack(pkmn: dict, gd: GameData) -> bool:
    cid = pkmn.get("id")
    if cid is None:
        return False
    attached = pkmn.get("energies") or []
    for aid in gd.card_attacks.get(cid, []):
        if gd.can_pay(gd.attack_cost(aid), attached):
            return True
    return False


def _pad(lst, n: int, fb: int):
    if n <= 0:
        return []
    out = list(lst)
    if len(out) >= n:
        return out[:n]
    while len(out) < n:
        out.append(fb)
    return out


def _visible_ids(player: dict) -> list[int]:
    out: list[int] = []
    for c in player.get("hand") or []:
        cid = _cid(c)
        if cid is not None:
            out.append(cid)
    for c in player.get("discard") or []:
        cid = _cid(c)
        if cid is not None:
            out.append(cid)
    for mon in [_active(player)] + list(player.get("bench") or []):
        if not isinstance(mon, dict):
            continue
        if isinstance(mon.get("id"), int):
            out.append(mon["id"])
        for key in ("energyCards", "tools", "preEvolution"):
            for c in mon.get(key) or []:
                cid = _cid(c)
                if cid is not None:
                    out.append(cid)
    return out


def _cid(card):
    if isinstance(card, dict):
        v = card.get("id")
        if v is None:
            v = card.get("cardId")
        return v if isinstance(v, int) else None
    return None


# Namespace of leaf-eval primitives passed to value_net.extract_features so the
# learned value uses exactly the same board signals as the hand-tuned _eval.
import types as _types

_HELPERS = _types.SimpleNamespace(
    _active=_active,
    _prizes_left=_prizes_left,
    _best_affordable_dmg=_best_affordable_dmg,
    _can_attack=_can_attack,
    _energy_in_play=_energy_in_play,
)
