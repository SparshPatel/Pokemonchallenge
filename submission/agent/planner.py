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
# evaluation.py is kept for offline training utilities.
# Runtime evaluation uses ValueNet + heuristic fallback.
from .value_net import (
    ValueNet,
    extract_features,
)
from .evaluation import (
    DynamicWeightGenerator,
    MatchLearner,
    Evaluator,
    EvaluationContext,
)
import dataclasses
import importlib
import math
import random
import time
from collections import Counter
from . import rules
from .adapter import Select, current_state, extract_select, your_index
from .enums import OptionType, SelectType
from .gamedata import GameData
from dataclasses import dataclass, field
# Modules that may expose the search surface, in priority order.
_ENGINE_MODULES = ("cg.api", "cg", "api")
_SEARCH_FUNCS = ("search_begin", "search_step", "search_end", "to_observation_class")
TERMINAL_WIN = 100000.0
@dataclass(slots=True)
class PlannerCache:
    """
    All planner caches in one place.
    feature:
        state -> FeatureVector
    evaluation:
        state -> scalar evaluation
    search:
        (state, depth) -> backed-up search value
    """
    feature: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)

    def clear(self):
        self.feature.clear()
        self.evaluation.clear()
        self.search.clear()
@dataclass(slots=True)
class TreeNode:
    """
    Statistics stored for one MCTS node.
    """
    visits: int = 0
    virtual_visits: int = 0
    value_sum: float = 0.0
    priors: dict[int, float] = field(default_factory=dict)
    children: dict[int, int] = field(default_factory=dict)
    expanded: bool = False
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
@dataclass(slots=True)
class SearchContext:
    """
    Per-search mutable state.
    A new context is created for every determinization so no search
    accidentally reuses caches from a previous simulation.
    """
    cache: PlannerCache = field(default_factory=PlannerCache)
    tree: dict[int, TreeNode] = field(default_factory=dict)
    deadline: float = 0.0
    nodes: int = 0

    def reset(self):
        self.cache.clear()
        self.tree.clear()
        self.nodes = 0
    
class TurnPlanner:
    def __init__(
        self,
        cards,
        gamedata: GameData | None = None,
        your_deck_ids: list[int] | None = None,
        opponent_deck_ids: list[int] | None = None,
        max_think_s: float = 0.6,
        max_depth: int = 6,
        n_determinizations: int = 2,
        max_nodes: int = 2000,
        opp_response: bool | None = None,
        max_opp_steps: int = 40,
        seed: int = 0,
    ):
        self.cards = cards
        self.gamedata = gamedata or GameData.load()
        self.your_deck_ids = list(your_deck_ids or [])
        self.opponent_deck_ids = list(
            opponent_deck_ids or self.your_deck_ids
        )
        self.max_think_s = max_think_s
        self.max_depth = max_depth
        self.k = n_determinizations
        self.max_nodes = max_nodes
        self.max_opp_steps = max_opp_steps
        self.opp_response = opp_response
        self._engine = self._locate_engine()
        self.ctx = SearchContext()
        self.opponent_embedding = None
        # -------------------------------
        # Learned value network (optional)
        # -------------------------------
                # -------------------------------
        # Learned models
        # -------------------------------
        try:
            from .value_net import ValueNet
            from .weight_model import WeightNet
            self.value_net = ValueNet()
            self.weight_net = WeightNet()
        except Exception:
            self.value_net = None
            self.weight_net = None
        self.use_value_net = (
            self.value_net is not None
            and getattr(self.value_net, "available", False)
        )
        self.use_weight_net = (
            self.weight_net is not None
            and getattr(self.weight_net, "available", False)
        )
        # -----------------------------------------
        # Heuristic fallback evaluator
        # -----------------------------------------
        self.weight_generator = DynamicWeightGenerator()
        self.match_learner = MatchLearner(
            self.weight_generator,
        )
        self.evaluator = Evaluator(
            self.weight_generator,
            self.match_learner,
        )
        self.rng = random.Random(seed)
        self.cpuct = 1.4
        self.min_pw = 2
        self.pw_alpha = 0.50
        
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
    
    def _active(self, player):
        return _active(player)

    def _prizes_left(self, player):
        return _prizes_left(player)

    def _best_affordable_dmg(self, attacker, defender, gd):
        return _best_affordable_dmg(attacker, defender, gd)

    def _can_attack(self, pkmn, gd):
        return _can_attack(pkmn, gd)

    def _energy_in_play(self, player):
        return _energy_in_play(player)

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
        wildly across determinizations.
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
        self.ctx.deadline = min(
            deadline,
            time.monotonic() + self.max_think_s,
        )
        aggregated: dict[int, list[float]] = {}
        for det in determinizations:
            if (
                self.ctx.nodes >= self.max_nodes
                or time.monotonic() >= self.ctx.deadline
            ):
                break
            values = self._search_one(
                eng,
                obs_cls,
                det,
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
                variance = (
                    sum((v - mean) ** 2 for v in vals)
                    / len(vals)
                )
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
        me,
    ) -> dict[int, float] | None:
        deadline = self.ctx.deadline
        try:
            ss = eng.search_begin(
                obs_cls,
                *det,
                False,
            )
        except Exception:
            return None
        self.ctx.reset()
        self.ctx.deadline = deadline
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
    ):
        root_obs = _as_obs_dict(root_ss)
        if root_obs is None:
            return {}
        root_select = extract_select(root_obs)
        if root_select is None or not root_select.options:
            return {}
        root_id = root_ss.searchId
        self._expand_node(
            root_id,
            root_obs,
            root_select,
        )
        while (
            self.ctx.nodes < self.max_nodes
            and time.monotonic() < self.ctx.deadline
        ):
            self._plan(
                eng,
                root_ss,
                me,
                self.max_depth,
                [root_id],
            )
        root = self._node(root_id)
        values = {}
        for action, child_id in root.children.items():
            child = self._node(child_id)
            if child.visits == 0:
                continue
            values[action] = child.value
        return values

    def _state_key(
        self,
        state: dict,
        me: int,
    ):
        """
        Hashable public representation of a board state.
        Hidden information is ignored, but every public resource that
        influences legal actions or evaluation is preserved so search
        transpositions remain correct.
        """
        players = state.get("players")
        if not isinstance(players, list) or len(players) < 2:
            return None
        def encode_pokemon(mon):
            if not isinstance(mon, dict):
                return None
            return (
                mon.get("id"),
                mon.get("hp"),
                len(mon.get("energies") or []),
                tuple(sorted(mon.get("tools") or [])),
                tuple(sorted(mon.get("specialConditions") or [])),
                mon.get("damage", 0),
            )
        def encode_player(player):
            active = encode_pokemon(
                _active(player)
            )
            bench = tuple(
                sorted(
                    encode_pokemon(b)
                    for b in (player.get("bench") or [])
                    if isinstance(b, dict)
                )
            )
            return (
                _prizes_left(player),
                player.get("handCount", 0),
                player.get("deckCount", 0),
                player.get("discardCount", len(player.get("discard") or [])),
                player.get("supporterPlayed", False),
                active,
                bench,
                _energy_in_play(player),
            )
        return (
            encode_player(players[me]),
            encode_player(players[1 - me]),
            state.get("turn", 0),
            state.get("turnPlayer", me),
            state.get("stadium"),
        )
        
    def _node(
        self,
        search_id,
    ) -> TreeNode:
        return self.ctx.tree.setdefault(
            search_id,
            TreeNode(),
        )

    def _backup(
        self,
        path,
        value,
    ):
        for sid in reversed(path):
            node = self._node(sid)
            node.visits += 1
            node.value_sum += value
            
    def _apply_virtual_loss(
        self,
        path,
    ):
        """
        Reserve this path while a simulation is traversing it.
        """
        for sid in path:
            node = self._node(sid)
            node.virtual_visits += 1

    def _revert_virtual_loss(
        self,
        path,
    ):
        """
        Undo the temporary virtual visit counts.
        """
        for sid in path:
            node = self._node(sid)
            if node.virtual_visits:
                node.virtual_visits -= 1

    def _expand_node(
        self,
        search_id,
        node_dict,
        select,
    ):
        node = self._node(search_id)
        if node.expanded:
            return
        node.priors = self._policy_prior(
            node_dict,
            select,
            search_id,
        )
        node.expanded = True

    def _ucb_score(
        self,
        parent_id,
        child_id,
        prior,
    ):
        parent = self._node(parent_id)
        child = self._node(child_id)
        parent_visits = (
            parent.visits
            + parent.virtual_visits
        )
        child_visits = (
            child.visits
            + child.virtual_visits
        )
        exploration = (
            self.cpuct
            * prior
            * math.sqrt(parent_visits + 1)
            / (1 + child_visits)
        )
        return child.value + exploration
    
    def _score_option(
        self,
        option,
        rules_pick,
        search_id,
    ):
        score = 1.0
        score += self._TYPE_PRIORITY.get(
            option.type,
            0,
        )
        if option.index == rules_pick:
            score += 5.0
        if (
            search_id is not None
            and search_id in self.ctx.tree
        ):
            parent = self._node(search_id)
            child_id = parent.children.get(
                option.index
            )
            if child_id is not None:
                child = self._node(child_id)
                if child.visits > 0:
                    q = child.value
                    score += 2.5 * math.tanh(
                        q / 75.0
                    )
                    score += min(
                        math.log1p(child.visits),
                        2.5,
                    )
        emb = self.opponent_embedding
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
    
    def _normalize_priors(
        self,
        scores,
    ):
        total = sum(scores.values())
        if total <= 0:
            n = max(len(scores), 1)
            return {
                k: 1.0 / n
                for k in scores
            }
        return {
            k: v / total
            for k, v in scores.items()
        }
        
    def _progressive_width(
        self,
        search_id,
        max_children,
    ):
        """
        Progressive widening.
        The number of children allowed to exist grows with the number
        of visits to the node.
        width = min_pw + visits^alpha
        """
        node = self._node(search_id)
        width = int(
            self.min_pw
            + (node.visits ** self.pw_alpha)
        )
        return min(
            width,
            max_children,
        )
    
    def _policy_prior(
        self,
        node,
        select,
        search_id=None,
    ):
        """
        Compute policy priors for every legal action.
        Progressive widening is handled during expansion rather than by
        discarding actions here.
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
            scores[option.index] = self._score_option(
                option,
                rules_pick,
                search_id,
            )
        return self._normalize_priors(scores)
     
    # ========================= CHANGE 4 =========================
    # Replace _plan() completely
    # ============================================================
    def _check_terminal(
        self,
        state,
        me,
    ):
        result = state.get("result", -1)
        if isinstance(result, int) and result >= 0:
            return self._terminal_value(result, me)
        return None
    
    def _check_cutoff(
        self,
        state,
        select,
        me,
        depth,
    ):
        if (
            depth <= 0
            or self.ctx.nodes >= self.max_nodes
            or time.monotonic() >= self.ctx.deadline
        ):
            return self._eval(state, me)
        if select is None or not select.options:
            return self._eval(state, me)
        return None
    
    def _handle_opponent(
        self,
        eng,
        ss,
        state,
        me,
    ):
        """
        Handle opponent decisions.
        If opponent search is enabled, perform a shallow search for the
        opponent rather than blindly following the rule policy.
        """
        acting = state.get("yourIndex", me)
        if acting == me:
            return None
        if not self.opp_response:
            return self._eval(state, me)
        node = _as_obs_dict(ss)
        if node is None:
            return self._eval(state, me)
        select = extract_select(node)
        if (
            select is None
            or not select.options
        ):
            return self._eval(state, me)
        if select.select_type != SelectType.MAIN:
            return self._drive_to_my_turn(
                eng,
                ss,
                me,
            )
        value = self._expand_search_node(
            eng,
            ss,
            node,
            state,
            select,
            1 - me,
            2,
            [ss.searchId],
        )
        return -value
        
    def _handle_forced_selection(
        self,
        eng,
        ss,
        node,
        state,
        select,
        me,
        depth,
        path,
    ):
        if select.select_type == SelectType.MAIN:
            return None
        try:
            choice = rules.choose(
                node,
                select,
                self.gamedata,
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
            nxt = eng.search_step(
                ss.searchId,
                choice,
            )
        except Exception:
            return self._eval(
                state,
                me,
            )
        self.ctx.nodes += 1
        return self._plan(
            eng,
            nxt,
            me,
            depth,
            path,
        )    
        
    def _select_best_action(
        self,
        search_id,
    ):
        """
        Select the child with the highest PUCT score.
        """
        node = self._node(search_id)
        best_action = None
        best_score = -float("inf")
        for action, prior in node.priors.items():
            child_id = node.children.get(action)
            if child_id is None:
                score = (
                    self.cpuct
                    * prior
                    * math.sqrt(node.visits + 1)
                )
            else:
                score = self._ucb_score(
                    search_id,
                    child_id,
                    prior,
                )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    
    def _select_child(
        self,
        search_id,
    ):
        """
        Select the already-expanded child with the highest PUCT score.
        Returns (action, child_id).
        """
        node = self._node(search_id)
        best_action = None
        best_child = None
        best_score = -float("inf")
        for action, child_id in node.children.items():
            prior = node.priors[action]
            score = self._ucb_score(
                search_id,
                child_id,
                prior,
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_id
        return best_action, best_child
    
    def _step_child(
        self,
        eng,
        ss,
        action,
    ):
        """
        Execute one search step.
        """
        try:
            nxt = eng.search_step(
                ss.searchId,
                [action],
            )
        except Exception:
            return None
        self.ctx.nodes += 1
        return nxt

    def _expand_search_node(
        self,
        eng,
        ss,
        node,
        state,
        select,
        me,
        depth,
        path,
    ):
        """
        Expand a search node using progressive widening and PUCT.
        """
        self._expand_node(
            ss.searchId,
            node,
            select,
        )
        node_stats = self._node(ss.searchId)
        allowed_width = self._progressive_width(
            ss.searchId,
            len(node_stats.priors),
        )
        # Expand unexplored children first
        ordered_actions = sorted(
            node_stats.priors.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for action, _ in ordered_actions:
            if len(node_stats.children) >= allowed_width:
                break
            if action not in node_stats.children:
                nxt = self._step_child(
                    eng,
                    ss,
                    action,
                )
                if nxt is None:
                   continue
                child_id = nxt.searchId
                node_stats.children[action] = child_id
                child_node = _as_obs_dict(nxt)
                if child_node is not None:
                    child_select = extract_select(child_node)
                    if child_select is not None:
                        self._expand_node(
                            child_id,
                            child_node,
                            child_select,
                        )
                return self._plan(
                    eng,
                    nxt,
                    me,
                    depth - 1,
                    path + [child_id],
                )
        # Otherwise descend into the best existing child.
        best_action, _ = self._select_child(
            ss.searchId,
        )
        if best_action is None:
            return self._eval(
                state,
                me,
            )
        nxt = self._step_child(
            eng,
            ss,
            best_action,
        )
        if nxt is None:
            return self._eval(
                state,
                me,
            )
        return self._plan(
            eng,
            nxt,
            me,
            depth - 1,
            path + [nxt.searchId],
        )
    
    def _plan(
        self,
        eng,
        ss,
        me,
        depth,
        path,
    ):
        node = _as_obs_dict(ss)
        if node is None:
            return 0.0
        state = node.get("current")
        if not isinstance(state, dict):
            return 0.0
        key = self._state_key(state, me)
        if key is None:
            key = ("leaf", id(state))
        cached = self.ctx.cache.search.get(
            (key, depth)
        )
        
        if cached is not None:
            self._backup(path, cached)
            return cached
        select = extract_select(node)
        # --------------------------------------------------
        # Terminal state
        # --------------------------------------------------
        value = self._check_terminal(
            state,
            me,
        )
        if value is not None:
            self.ctx.cache.search[(key, depth)] = value
            self._backup(path, value)
            return value
        # --------------------------------------------------
        # Cutoff
        # --------------------------------------------------
        value = self._check_cutoff(
            state,
            select,
            me,
            depth,
        )
        if value is not None:
            self.ctx.cache.search[(key, depth)] = value
            self._backup(path, value)
            return value
        # --------------------------------------------------
        # Opponent turn
        # --------------------------------------------------
        value = self._handle_opponent(
            eng,
            ss,
            state,
            me,
        )
        if value is not None:
            self.ctx.cache.search[(key, depth)] = value
            self._backup(path, value)
            return value
        # --------------------------------------------------
        # Forced selections
        # --------------------------------------------------
        value = self._handle_forced_selection(
            eng,
            ss,
            node,
            state,
            select,
            me,
            depth,
            path,
        )
        if value is not None:
            self.ctx.cache.search[(key, depth)] = value
            self._backup(path, value)
            return value
        # --------------------------------------------------
        # Expand node
        # --------------------------------------------------
        # --------------------------------------------------
        # Expand node
        # --------------------------------------------------
        self._apply_virtual_loss(path)
        try:
            value = self._expand_search_node(
                eng,
                ss,
                node,
                state,
                select,
                me,
                depth,
                path,
            )
        finally:
            self._revert_virtual_loss(path)
        self.ctx.cache.search[
            (key, depth)
        ] = value
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
            if time.monotonic() >= self.ctx.deadline or self.ctx.nodes >= self.max_nodes:
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
            self.ctx.nodes += 1
        node = _as_obs_dict(ss)
        state = node.get("current") if node else None
        if isinstance(state, dict):
            return self._eval(state, me)
        return 0.0

    def _terminal_value(self, result: int, me: int) -> float:
        if result == me:
            return TERMINAL_WIN
        if result == 2:
            return 0.0
        return -TERMINAL_WIN

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
    
    # --- leaf evaluation --------------------------------------------------
    def _eval(
        self,
        state,
        me,
    ) -> float:
        """
        Evaluate a board state.
        Pipeline:
            GameState
                ↓
            FeatureExtractor
                ↓
            ValueNet (if available)
                ↓
            Handcrafted evaluator (fallback)
        All results are cached within the current search context.
        """
        key = self._state_key(
            state,
            me,
        )
        if key is not None:
            cached = self.ctx.cache.evaluation.get(key)
            if cached is not None:
                return cached
        features = extract_features(
            state,
            me,
            self.gamedata,
            self,
        )
        # --------------------------------------------------
        # Preferred evaluation:
        # learned value network
        # --------------------------------------------------
        score = None
        if (
            getattr(self, "use_value_net", False)
            and self.value_net is not None
        ):
            try:
                win_prob = self.value_net.predict(
                    features
                )
                # convert probability into symmetric score
                score = (
                    (win_prob - 0.5)
                    * 2.0
                    * TERMINAL_WIN
                )
            except Exception:
                score = None
        # --------------------------------------------------
        # Fallback:
        # handcrafted evaluator
        # --------------------------------------------------
        # --------------------------------------------------
        # Fallback:
        # deterministic heuristic evaluation
        # --------------------------------------------------
        if score is None:
            context = EvaluationContext(
                game_phase=(
                    1.0
                    - (
                        features[1]
                        + features[2]
                    )
                    / 12.0
                ),
                prize_diff=features[0],
                search_depth=self.max_depth,
                search_confidence=1.0,
                opponent_embedding=self.opponent_embedding,
            )
            if (
                self.use_weight_net
                and self.weight_net is not None
            ):
                try:
                    weights = self.weight_net.predict(
                        features,
                        context,
                    )
                    self.weight_generator.set_runtime_weights(
                        weights
                    )
                except Exception:
                    pass
            score = self.evaluator.evaluate(
                features,
                context,
            ).score
            
        # Cache evaluation for this search.
        if key is not None:
            self.ctx.cache.evaluation[key] = score
        return score

    # --- determinization --------------------------------------------------
    def _build_my_pool(
        self,
        player,
    ):
        unseen = Counter(
            self.your_deck_ids
        )
        for cid in _visible_ids(player):
            if unseen.get(cid, 0):
                unseen[cid] -= 1
        return list(
            unseen.elements()
        )
    
    def _build_opponent_pool(
        self,
    ):
        pool = list(
            self.opponent_deck_ids
        )
        self.rng.shuffle(pool)
        return pool
    
    def _sample_determinization(
        self,
        my_pool_base,
        opp_pool,
        counts,
        deck_given,
        fb,
        opp_active_needed,
    ):
        (
            my_deck_n,
            my_prize_n,
            opp_deck_n,
            opp_prize_n,
            opp_hand_n,
        ) = counts
        my_pool = list(my_pool_base)
        self.rng.shuffle(my_pool)
        if deck_given:
            your_deck = []
        else:
            your_deck = my_pool[:my_deck_n]
        your_prize = my_pool[
            my_deck_n:
            my_deck_n + my_prize_n
        ]
        your_deck = (
            []
            if deck_given
            else _pad(
                your_deck,
                my_deck_n,
                fb,
            )
        )
        your_prize = _pad(
            your_prize,
            my_prize_n,
            fb,
        )
        opp_deck = _pad(
            opp_pool[:opp_deck_n],
            opp_deck_n,
            fb,
        )
        opp_prize = _pad(
            opp_pool[
                opp_deck_n:
                opp_deck_n + opp_prize_n
            ],
            opp_prize_n,
            fb,
        )
        opp_hand = _pad(
            opp_pool[
                opp_deck_n + opp_prize_n:
                opp_deck_n + opp_prize_n + opp_hand_n
            ],
            opp_hand_n,
            fb,
        )
        opp_active = []
        if opp_active_needed:
            cid = self._first_basic(
                opp_hand + opp_deck
            )
            if cid is None:
                return None
            opp_active = [cid]
        return (
            your_deck,
            your_prize,
            opp_deck,
            opp_prize,
            opp_hand,
            opp_active,
        )
    
    def _build_determinizations(
        self,
        obs_dict,
        state,
        me,
    ):
        players = state.get("players") or []
        if len(players) < 2:
            return []
        mp = players[me]
        op = players[1 - me]
        sel = obs_dict.get("select")
        deck_given = (
            isinstance(sel, dict)
            and sel.get("deck") is not None
        )
        counts = (
            int(mp.get("deckCount") or 0),
            len(mp.get("prize") or []),
            int(op.get("deckCount") or 0),
            len(op.get("prize") or []),
            int(op.get("handCount") or 0),
        )
        fb = (
            self.your_deck_ids[0]
            if self.your_deck_ids
            else 0
        )
        my_pool = self._build_my_pool(mp)
        opp_active_needed = (
            bool(op.get("active"))
            and op["active"][0] is None
        )
        dets = []
        for _ in range(self.k):
            opp_pool = self._build_opponent_pool()
            det = self._sample_determinization(
                my_pool,
                opp_pool,
                counts,
                deck_given,
                fb,
                opp_active_needed,
            )
            if det is not None:
                dets.append(det)
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