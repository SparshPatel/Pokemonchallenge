"""
planner.py
Lightweight orchestration layer.
All search logic has been moved into:
    determinization.py
    search_tree.py
    search_expansion.py
    search_rollout.py
    opponent_model.py
    policy_prior.py
Planner now only coordinates those modules.
"""
from __future__ import annotations
import dataclasses
import importlib
import random
import time
from . import rules
from .adapter import (
    Select,
    current_state,
    extract_select,
    your_index,
)
from .enums import (
    SelectType,
)
from .gamedata import GameData
from .board_value import BoardEvaluator
from .value_net import (
    ValueNet,
    extract_features,
)
from .determinization import Determinizer
from .search_tree import SearchTree
from .policy_prior import PolicyPrior
from .opponent_model import OpponentModel
from .search_expansion import SearchExpansion
from .search_rollout import SearchRollout
_ENGINE_MODULES = (
    "cg.api",
    "cg",
    "api",
)
_SEARCH_FUNCS = (
    "search_begin",
    "search_step",
    "search_end",
    "to_observation_class",
)
TERMINAL_WIN = 100000.0

class TurnPlanner:
    def __init__(
        self,
        cards,
        gamedata: GameData | None = None,
        your_deck_ids=None,
        opponent_deck_ids=None,
        max_think_s=0.60,
        max_depth=6,
        n_determinizations=2,
        max_nodes=2000,
        opp_response=False,
        seed=0,
    ):
        self.cards = cards
        self.gamedata = gamedata or GameData.load()
        self.your_deck_ids = list(your_deck_ids or [])
        self.opponent_deck_ids = list(
            opponent_deck_ids
            or self.your_deck_ids
        )
        self.max_think_s = max_think_s
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.k = n_determinizations
        self.opp_response = opp_response
        self.rng = random.Random(seed)
        self.cpuct = 1.40
        self.dirichlet_alpha = 0.30
        self.dirichlet_epsilon = 0.25
        self.min_pw = 2
        self.pw_alpha = 0.50
        self.Select = Select
        self.SelectType = SelectType
        self.rules = rules
        self._engine = self._locate_engine()
        try:
            self.value_net = ValueNet()
            self.use_value_net = getattr(
                self.value_net,
                "available",
                False,
            )
        except Exception:
            self.value_net = None
            self.use_value_net = False
        self.board_evaluator = BoardEvaluator(
            self.gamedata,
        )
        self.tree = SearchTree()
        self.determinizer = Determinizer(
            self.gamedata,
            self.your_deck_ids,
            self.opponent_deck_ids,
            seed,
        )
        self.policy_prior = PolicyPrior(self)
        self.opponent_model = OpponentModel(
            self,
            self.tree,
        )
        self.expansion = SearchExpansion(
            self,
            self.tree,
            self.policy_prior,
        )
        self.rollout = SearchRollout(
            self,
            self.tree,
            self.opponent_model,
        )

    def _locate_engine(self):
        for name in _ENGINE_MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if all(
                hasattr(mod, fn)
                for fn in _SEARCH_FUNCS
            ):
                return mod
        return None

    def available(self):
        return self._engine is not None

    def choose(
        self,
        obs_dict,
        select,
        deadline,
    ):
        if self._engine is None:
            return None
        if select.select_type != SelectType.MAIN:
            return None
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return None
        me = your_index(obs_dict)
        obs_cls = self._engine.to_observation_class(
            obs_dict,
        )
        determinizations = self.determinizer.build(
            obs_dict,
            state,
            me,
            self.k,
        )
        if not determinizations:
            return None
        self.tree.ctx.deadline = min(
            deadline,
            time.monotonic() + self.max_think_s,
        )
        aggregated = {}
        for det in determinizations:
            values = self._search_one(
                obs_cls,
                det,
                me,
            )
            if not values:
                continue
            for action, value in values.items():
                aggregated.setdefault(
                    action,
                    [],
                ).append(value)
        if not aggregated:
            return None
        best_action = None
        best_score = float("-inf")
        for action, values in aggregated.items():
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = (
                    sum(
                        (v - mean) ** 2
                        for v in values
                    )
                    / len(values)
                )
            else:
                variance = 0.0
            confidence_penalty = (
                0.15 * (variance ** 0.5)
            )
            score = mean - confidence_penalty
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            return None
        return [best_action]

    # ---------------------------------------------------------
    def _search_one(
        self,
        obs_cls,
        determinization,
        me,
    ):
        try:
            ss = self._engine.search_begin(
                obs_cls,
                *determinization,
                False,
            )
        except Exception:
            return None
        self.tree.clear()
        self.tree.ctx.deadline = min(
            self.tree.ctx.deadline,
            time.monotonic() + self.max_think_s,
        )
        self._root_search_id = ss.searchId
        try:
            values = self._expand_root(
                ss,
                me,
            )
        finally:
            self._root_search_id = None
            try:
                self._engine.search_end()
            except Exception:
                pass
        return values

    # ---------------------------------------------------------
    def _expand_root(
        self,
        root_search_state,
        me,
    ):
        root_obs = self._as_obs_dict(
            root_search_state,
        )
        if root_obs is None:
            return {}
        root_select = extract_select(root_obs)
        if (
            root_select is None
            or not root_select.options
        ):
            return {}
        root_id = root_search_state.searchId
        self.expansion.expand_node(
            root_id,
            root_obs,
            root_select,
        )
        while (
            self.tree.ctx.nodes < self.max_nodes
            and time.monotonic()
            < self.tree.ctx.deadline
        ):
            self.rollout.search(
                self._engine,
                root_search_state,
                me,
                self.max_depth,
                [root_id],
            )
        root = self.tree.node(root_id)
        values = {}
        total_visits = max(
            1,
            sum(
                self.tree.node(cid).visits
                for cid in root.children.values()
            ),
        )
        for action, child_id in root.children.items():
            child = self.tree.node(child_id)
            if child.visits == 0:
                continue
            visit_fraction = (
                child.visits / total_visits
            )
            values[action] = (
                0.70 * child.value
                + 0.30
                * visit_fraction
                * TERMINAL_WIN
            )
        return values

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
    ):
        return self.expansion.expand_search(
            engine,
            search_state,
            node,
            state,
            select,
            me,
            depth,
            path,
            self.rollout,
        )
        
        # ---------------------------------------------------------
    def _eval(
        self,
        state,
        me,
    ):
        key = self._state_key(
            state,
            me,
        )
        if key is not None:
            cached = (
                self.tree.ctx.cache.evaluation.get(key)
            )
            if cached is not None:
                return cached
        features = None
        if key is not None:
            features = (
                self.tree.ctx.cache.feature.get(key)
            )
        if features is None:
            features = extract_features(
                state,
                me,
                self.gamedata,
                self,
            )
            if key is not None:
                self.tree.ctx.cache.feature[
                    key
                ] = features
        score = None
        if (
            self.use_value_net
            and self.value_net is not None
        ):
            try:
                win_prob = self.value_net.predict(
                    features,
                )
                score = (
                    (win_prob - 0.5)
                    * 2.0
                    * TERMINAL_WIN
                )
            except Exception:
                score = None
        if score is None:
            try:
                score = (
                    self.board_evaluator.evaluate(
                        state,
                        me,
                    )
                )
            except Exception:
                score = None
        if score is None:
            score = 0.0
        if key is not None:
            self.tree.ctx.cache.evaluation[
                key
            ] = score
        return score

    # ---------------------------------------------------------
    def _terminal_value(
        self,
        result,
        me,
    ):
        if result == me:
            return TERMINAL_WIN
        if result == 2:
            return 0.0
        return -TERMINAL_WIN

    # ---------------------------------------------------------
    def _state_key(
        self,
        state,
        me,
    ):
        players = state.get("players")
        if (
            not isinstance(players, list)
            or len(players) < 2
        ):
            return None
        
        def encode_pokemon(mon):
            if not isinstance(mon, dict):
                return None
            return (
                mon.get("id"),
                mon.get("hp"),
                mon.get("damage", 0),
                len(mon.get("energies") or []),
                tuple(
                    sorted(
                        mon.get("tools") or []
                    )
                ),
                tuple(
                    sorted(
                        mon.get(
                            "specialConditions"
                        )
                        or []
                    )
                ),
            )

        def encode_player(player):
            active = encode_pokemon(
                _active(player),
            )
            bench = tuple(
                sorted(
                    (
                        encode_pokemon(mon)
                        for mon in (
                            player.get("bench")
                            or []
                        )
                        if isinstance(mon, dict)
                    ),
                    key=str,
                )
            )
            return (
                _prizes_left(player),
                player.get("deckCount", 0),
                player.get("handCount", 0),
                player.get(
                    "discardCount",
                    len(
                        player.get("discard")
                        or []
                    ),
                ),
                player.get(
                    "supporterPlayed",
                    False,
                ),
                active,
                bench,
                _energy_in_play(player),
            )
        return (
            encode_player(players[me]),
            encode_player(players[1 - me]),
            state.get("turnPlayer"),
            state.get("turn", 0),
            state.get("stadium"),
            state.get("result", -1),
        )

    # ---------------------------------------------------------
    @staticmethod
    def _as_obs_dict(search_state):
        obs = getattr(
            search_state,
            "observation",
            None,
        )
        if obs is None:
            return None
        try:
            return dataclasses.asdict(obs)
        except Exception:
            return None
        
# ----------------------------------------------------------------------
# Generic helper functions
# ----------------------------------------------------------------------
def _active(player):
    arr = player.get("active") or []
    if arr and isinstance(arr[0], dict):
        return arr[0]
    return None

def _prizes_left(player):
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    return 6

def _energy_in_play(player):
    total = 0
    mons = [_active(player)]
    mons.extend(player.get("bench") or [])
    for mon in mons:
        if not isinstance(mon, dict):
            continue
        total += len(mon.get("energies") or [])
    return total

def _best_affordable_dmg(
    attacker,
    defender,
    gamedata,
):
    cid = attacker.get("id")
    if cid is None:
        return 0
    attached = attacker.get("energies") or []
    defender_id = None
    if isinstance(defender, dict):
        defender_id = defender.get("id")
    best = 0
    for attack_id in gamedata.card_attacks.get(
        cid,
        [],
    ):
        if not gamedata.can_pay(
            gamedata.attack_cost(attack_id),
            attached,
        ):
            continue
        damage = gamedata.attack_damage(
            attack_id,
        )
        if defender_id is not None:
            damage = gamedata.effective_damage(
                cid,
                damage,
                defender_id,
            )
        if damage > best:
            best = damage
    return best

def _can_attack(
    pokemon,
    gamedata,
):
    cid = pokemon.get("id")
    if cid is None:
        return False
    attached = pokemon.get("energies") or []
    for attack_id in gamedata.card_attacks.get(
        cid,
        [],
    ):
        if gamedata.can_pay(
            gamedata.attack_cost(attack_id),
            attached,
        ):
            return True
    return False