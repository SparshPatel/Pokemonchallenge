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

from altair import value
from cachetools import cached
from .evaluation import (
    FeatureVector,
    WeightGenerator,
    Evaluator,
)
from .evaluation import (
    PlannerFeatureExtractor,
    WeightGenerator,
    Evaluator,
)
import dataclasses
import importlib
import math
import os
import random
import time
from collections import Counter
from functools import lru_cache

from . import rules
from .adapter import Select, current_state, extract_select, your_index
from .enums import CardType, OptionType, SelectType
from .gamedata import GameData

try:
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
        gamedata: GameData | None=None,
        your_deck_ids:list[int]|None=None,
        opponent_deck_ids:list[int]|None=None,
        max_think_s:float=0.6,
        beam_width:int=4,
        max_depth:int=6,
        n_determinizations:int=2,
        max_nodes:int=2000,
        opp_response:bool|None=None,
        max_opp_steps:int=40,
        seed:int=0,
        eval_weights:dict|None=None,
        use_value_net:bool|None=None,
        value_net_path:str|None=None,
    ):
        self.cards=cards
        self.gamedata=gamedata or GameData.load()
        self.your_deck_ids=list(your_deck_ids or [])
        self.opponent_deck_ids=list(opponent_deck_ids or self.your_deck_ids)
        self.max_think_s=max_think_s
        self.beam_width=beam_width
        self.max_depth=max_depth
        self.k=n_determinizations
        self.max_nodes=max_nodes
        self.rng=random.Random(seed)
        self.eval=dict(EVAL)
        if eval_weights:
            self.eval.update(eval_weights)
        self.value_net=None
        self._vn_scale=60.0
        self._engine=self._locate_engine()
        self._deadline=0.0
        self._nodes=0
        self.weight_generator=WeightGenerator()
        self.evaluator=Evaluator(self.weight_generator)
        self.feature_extractor=PlannerFeatureExtractor(self.gamedata)
        self.transposition={}
        self.node_visits={}
        # ----- UCT -----
        self.Q={}
        self.N={}
        self.P={}
        self.virtual_loss={}
        self.cpuct=1.4
        self.exploration_c=0.15
        self.previous_features=None

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
    def choose(
        self,
        obs_dict,
        select: Select,
        deadline: float,
    ) -> list[int] | None:
        """
        Search every determinization and aggregate the value of each root action.
        Rather than simply averaging values, penalize actions whose value varies
        wildly across determinizations. This produces much more stable decisions
        under hidden information.
        """
        eng = self._engine
        if eng is None or not select.options:
            return None

        if select.select_type != SelectType.MAIN:
            return None
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return None
        me = your_index(obs_dict)
        if not isinstance(state.get("turn"), int):
            return None

        try:
            obs_cls = eng.to_observation_class(obs_dict)
        except Exception:
            return None
        if getattr(obs_cls, "search_begin_input", None) is None:
            return None
        determinizations = self._build_determinizations(
            obs_dict,
            state,
            me,
        )
        if not determinizations:
            return None
        self._deadline = min(
            deadline,
            time.monotonic() + self.max_think_s,
        )
        aggregated: dict[int, list[float]] = {}
        for det in determinizations:
            if (
                self._nodes >= self.max_nodes
                or time.monotonic() >= self._deadline
            ):
                break
            values = self._search_one(
                eng,
                obs_cls,
                det,
                obs_dict,
                me,
            )
            if not values:
                continue
            for action, value in values.items():
                aggregated.setdefault(action, []).append(value)
        if not aggregated:
            return None
        best_action = None
        best_score = float("-inf")
        for action, vals in aggregated.items():
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            else:
                variance = 0.0
            confidence_penalty = 0.15 * math.sqrt(variance)
            score = mean - confidence_penalty
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            return None
        return [best_action]

    # ========================= CHANGE 2 =========================
    # Replace _search_one()
    # ============================================================

    def _search_one(
        self,
        eng,
        obs_cls,
        det,
        obs_dict,
        me,
    ) -> dict[int, float] | None:
        try:
            ss = eng.search_begin(obs_cls, *det, False)
        except Exception:
            return None
        self._nodes = 0
        self.previous_features = None
        # Fresh search statistics.
        self.transposition.clear()
        self.N.clear()
        self.Q.clear()
        # Fresh learner for this search.
        self.weight_generator = WeightGenerator()
        self.evaluator = Evaluator(self.weight_generator)
        try:
            values = self._expand_root(
                eng,
                ss,
                me,
            )
        except Exception:
            values = None
        finally:
            try:
                eng.search_end()
            except Exception:
                pass
        return values

    # --- search tree ------------------------------------------------------
    # ========================= CHANGE 3 =========================
    # Replace _expand_root()
    # ============================================================

    def _expand_root(
        self,
        eng,
        root_ss,
        me,
    ) -> dict[int, float]:
        """
        Root Monte Carlo search.
        Each iteration performs
            Selection
                ↓
            Expansion
                ↓
            Evaluation
                ↓
            Backup
        until the search budget expires.
        """
        node = _as_obs_dict(root_ss)
        if node is None:
            return {}
        select = extract_select(node)
        if select is None or not select.options:
            return {}
        root_key = ("root", root_ss.searchId)
        self.N.setdefault(root_key, 0)
        self.Q.setdefault(root_key, 0.0)
        priors = self._policy_prior(node, select, root_key)
        values = {}
        while (
            self._nodes < self.max_nodes
            and time.monotonic() < self._deadline
        ):
            # -------------------------
            # Selection
            # -------------------------
            best_action = None
            best_score = -float("inf")
            for action, prior in priors.items():
                child_key = (root_key, action)
                score = self._ucb_score(
                    root_key,
                    child_key,
                    prior,
                )
                if score > best_score:
                    best_score = score
                    best_action = action
            if best_action is None:
                break
            # -------------------------
            # Expansion
            # -------------------------
            try:
                child = eng.search_step(
                    root_ss.searchId,
                    [best_action],
                )
            except Exception:
                priors.pop(best_action, None)
                continue
            self._nodes += 1
            # -------------------------
            # Simulation / Evaluation
            # -------------------------
            value = self._plan(
                eng,
                child,
                me,
                self.max_depth - 1,
            )
            # -------------------------
            # Backup
            # -------------------------
            self.N[root_key] += 1
            child_key = (root_key, best_action)
            self.N[child_key] = (
                self.N.get(child_key, 0) + 1
            )
            self.Q[child_key] = (
                self.Q.get(child_key, 0.0) + value
            )
            values[best_action] = self.Q[child_key] / self.N[child_key]
        return values
    
    def _state_key(self, state: dict, me: int):
        """
        Produce a deterministic hashable representation of the public board state.
        Hidden information (deck contents, hand identities, prizes, etc.) is
        deliberately ignored so that equivalent public positions share the same
        transposition entry.
        """
        players = state.get("players") or []
        if len(players) < 2:
            return None
        mp = players[me] if isinstance(players[me], dict) else {}
        op = players[1 - me] if isinstance(players[1 - me], dict) else {}
        def encode_player(player):
            active = _active(player)
            if active is None:
                active_repr = None
            else:
                active_repr = (
                    active.get("id"),
                    active.get("hp"),
                    len(active.get("energies") or []),
                )
            bench = []
            for mon in player.get("bench") or []:
                if not isinstance(mon, dict):
                    continue
                bench.append(
                    (
                        mon.get("id"),
                        mon.get("hp"),
                        len(mon.get("energies") or []),
                    )
                )
            bench.sort()
            return (
                _prizes_left(player),
                player.get("handCount", 0),
                active_repr,
                tuple(bench),
                _energy_in_play(player),
            )
        return (
            encode_player(mp),
            encode_player(op),
            state.get("turn", 0),
            state.get("turnPlayer", me),
        )

    def _visit_count(self,key):
        return self.N.get(key,0)

    def _q_value(self,key):
        n=self.N.get(key,0)
        if n==0:
            return 0.0
        return self.Q.get(key,0.0)/n

    def _ucb_score(
        self,
        parent_key,
        child_key,
        prior,
    ):
        parent_visits = max(1, self.N.get(parent_key, 1),)
        child_visits = self.N.get(child_key, 0)
        q = self._q_value(child_key)
        exploration = (self.cpuct * prior * math.sqrt(parent_visits) / (1 + child_visits))
        # Progressive bias.
        # A heuristic prior should matter early in the search,
        # then gradually disappear as empirical values become reliable.
        # This gives good move ordering without permanently
        # forcing the search toward the rule policy.
        bias = (self.exploration_c * prior / (1 + child_visits))
        return q + exploration + bias
    
    def _policy_prior(self, node, select, key=None):
        """
        Produce adaptive PUCT priors.
        The prior combines:
            • rule policy recommendation
            • action type prior
            • historical search value
            • historical visit count
            • opponent behaviour embedding
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
            score = float(self._TYPE_PRIORITY.get(option.type, 0))
            # -------------------------------------------------
            # Rule policy prior
            # -------------------------------------------------
            if option.index == rules_pick:
                score += 8.0
            # -------------------------------------------------
            # Search statistics
            # -------------------------------------------------
            if key is not None:
                child = (key, option.index)
                visits = self.N.get(child, 0)
                if visits:
                    q = self.Q.get(child, 0.0) / visits
                    score += 2.0 * math.tanh(q / 50.0)
                    score += min(
                        math.log1p(visits),
                        2.5,
                    )
            # -------------------------------------------------
            # Opponent adaptation
            # -------------------------------------------------
            emb = self.evaluator.opponent_embedding
            if emb is not None:
                attack_rate = emb[0]
                ability_rate = emb[1]
                item_rate = emb[2]
                if (
                    option.type == OptionType.ATTACK
                    and attack_rate > 0.50
                ):
                    score += 1.5
                elif (
                    option.type == OptionType.EVOLVE
                    and attack_rate > 0.50
                ):
                    score += 1.0
                elif (
                    option.type == OptionType.ABILITY
                    and ability_rate > 0.30
                ):
                    score += 1.0
                elif (
                    option.type == OptionType.PLAY
                    and item_rate > 0.40
                ):
                    score += 0.75
            scored.append(
                (
                    max(score, 0.01),
                    option.index,
                )
            )
        scored.sort(reverse=True)
        if key is None:
            width = len(scored)
        else:
            width = self._progressive_width(
                key,
                len(scored),
            )
        scored = scored[:width]
        total = sum(score for score, _ in scored)
        if total <= 0:
            n = max(1, len(scored))
            return {
                idx: 1.0 / n
                for _, idx in scored
            }
        priors = {}
        for score, idx in scored:
            priors[idx] = score / total
        return priors
    
    def _progressive_width(self,key,n_actions):
        visits=self.N.get(key,0)
        width=int(1+math.sqrt(visits))
        if width<2:
            width=2
        return min(width,n_actions)
    
    def encode(player):
        act = _active(player)
        return (
            _prizes_left(player),
            player.get("handCount", 0),
            _energy_in_play(player),
            tuple(sorted((p.get("id"), p.get("hp")) for p in (player.get("bench") or []) if isinstance(p, dict))),
            None if act is None else (act.get("id"), act.get("hp"), len(act.get("energies") or [])),
        )
        return (encode(mp), encode(op))
     
    # ========================= CHANGE 4 =========================
    # Replace _plan() completely
    # ============================================================

    def _plan(self, eng, ss, me, depth):
        node = _as_obs_dict(ss)
        if node is None:
            return 0.0

        state = node.get("current")
        if not isinstance(state, dict):
            return 0.0

        key = self._state_key(state, me)
        if key is None:
            key = ("leaf", id(state))

        result = state.get("result", -1)
        if isinstance(result, int) and result >= 0:
            value = self._terminal_value(result, me)
            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value
            return value

        if (
            depth <= 0
            or self._nodes >= self.max_nodes
            or time.monotonic() >= self._deadline
        ):
            value = self._eval(state, me)
            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value
            return value

        select = extract_select(node)

        if select is None or not select.options:
            value = self._eval(state, me)
            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value
            return value

        acting = state.get("yourIndex", me)

        if acting != me:
            if self.opp_response:
                value = self._drive_to_my_turn(eng, ss, me)
            else:
                value = self._eval(state, me)

            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value
            return value

        if select.select_type != SelectType.MAIN:
            try:
                choice = rules.choose(node, select, self.gamedata)
            except Exception:
                choice = list(range(max(1, select.min_count)))

            try:
                nxt = eng.search_step(ss.searchId, choice)
            except Exception:
                value = self._eval(state, me)
                self.N[key] = self.N.get(key, 0) + 1
                self.Q[key] = self.Q.get(key, 0.0) + value
                return value

            self._nodes += 1

            value = self._plan(eng, nxt, me, depth)

            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value

            return value

        priors = self._policy_prior(node, select, key)

        best_action = None
        best_score = -float("inf")

        for idx, prior in priors.items():
            child_key = (key, idx)

            score = self._ucb_score(
                key,
                child_key,
                prior,
            )

            if score > best_score:
                best_score = score
                best_action = idx

        if best_action is None:
            value = self._eval(state, me)

            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value

            return value

        try:
            nxt = eng.search_step(
                ss.searchId,
                [best_action],
            )
        except Exception:
            value = self._eval(state, me)

            self.N[key] = self.N.get(key, 0) + 1
            self.Q[key] = self.Q.get(key, 0.0) + value

            return value

        self._nodes += 1

        # -------- Dynamic depth extension --------

        next_depth = depth - 1

        remaining = (
            self._deadline - time.monotonic()
        )

        if (
            len(select.options) <= 3
            and remaining > 0.25
            and depth < self.max_depth + 2
        ):
            next_depth = depth

        value = self._plan(
            eng,
            nxt,
            me,
            next_depth,
        )

        self.N[key] = self.N.get(key, 0) + 1
        self.Q[key] = self.Q.get(key, 0.0) + value

        child_key = (key, best_action)

        self.N[child_key] = self.N.get(child_key, 0) + 1
        self.Q[child_key] = self.Q.get(child_key, 0.0) + value

        return value
    
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
    
    
    def _adaptive_beam_width(self, select) -> int:
        n = len(select.options)
        if n <= 4:
            return n
        if n <= 8:
            return min(6, n)
        if n <= 15:
            return min(8, n)
        return min(10, n)

    # ========================= CHANGE 5 =========================
    # Replace _candidate_main_options()
    # ===========================================================

    def _candidate_main_options(self, node, select):
        """
        Produce the candidate actions that will actually be searched.
        Unlike the previous implementation, this is no longer purely rule-based.
        It combines
            • rule policy
            • learned priors
            • historical search value
            • action diversity
        before applying the adaptive beam width.
        """
        if not select.options:
            return []
        try:
            rules_pick = rules._choose_main(
                node,
                select,
                self.gamedata,
            )
        except Exception:
            rules_pick = None
        priors = self._policy_prior(
            node,
            select,
        )
        scored = []
        for option in select.options:
            score = priors.get(option.index, 0.0)
            if option.index == rules_pick:
                score += 0.50
            action_key = ("candidate", option.index)
            visits = self.N.get(action_key, 0)
            if visits:
                score += min(
                    math.log1p(visits) * 0.15,
                    0.60,
                )
            # encourage diversity
            if option.type == OptionType.ATTACK:
                score += 0.20
            elif option.type == OptionType.ABILITY:
                score += 0.15
            elif option.type == OptionType.EVOLVE:
                score += 0.12
            scored.append(
                (
                    score,
                    option.index,
                )
            )
        scored.sort(reverse=True)
        beam = self._adaptive_beam_width(select)
        ordered = []
        for _, idx in scored:
            if idx not in ordered:
                ordered.append(idx)
            if len(ordered) >= beam:
                break
        return ordered
    
    
    def _extract_features(self, state: dict, me: int):
        return self.feature_extractor.extract(state, me)
    
    # --- leaf evaluation --------------------------------------------------
    def _eval(self, state: dict, me: int) -> float:
        """
        Evaluate a public board state.
        Results are cached using the public state key so repeated
        transpositions reuse the same evaluation instead of repeatedly
        extracting features and invoking the evaluator.
        """
        key = self._state_key(state, me)
        if key is not None:
            cached = self.transposition.get(key)
            if cached is not None:
                return cached
        features = self.feature_extractor.extract(state, me)
        result = self.evaluator.evaluate(features)
        if self.previous_features is not None:
            self.evaluator.temporal_difference_update(
                self.previous_features,
                features,
                reward=result.score,
            )
        self.previous_features = features.copy()
        if key is not None:
            self.transposition[key] = result.score
        return result.score

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
