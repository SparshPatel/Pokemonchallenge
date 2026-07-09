"""
Information Set Monte Carlo Tree Search (ISMCTS).
Although this file retains the historical name ``pimc.py`` for compatibility
with the rest of the project, the implementation is a true Information Set
Monte Carlo Tree Search rather than flat Partially Observable Monte Carlo.
The algorithm operates as follows:
    Belief update
          ↓
    Sample determinization
          ↓
    Selection
          ↓
    Expansion
          ↓
    Simulation
          ↓
    Backpropagation
The search tree is shared across determinizations, allowing statistics to
accumulate over multiple samples from the opponent belief state.
"""
from __future__ import annotations
import dataclasses
import importlib
import random
import time
import math
from collections import Counter
from .evaluation import (
    PlannerFeatureExtractor,
    WeightGenerator,
    Evaluator,
    DynamicWeightGenerator,
    EvaluationContext,
)
from . import rules
from .adapter import (
    Select,
    current_state,
    extract_select,
    your_index,
)
from .belief import (
    BeliefState,
    CandidateDeck,
)
from .cards import CardStats
from .gamedata import GameData
from .search import SearchNode
_ENGINE_MODULES = (
    "cg.api",
    "cg.sim",
    "cg",
    "api",
    "sim",
)
_SEARCH_FUNCS = (
    "search_begin",
    "search_step",
    "search_release",
)
class PIMCSearcher:
    """
    Information Set Monte Carlo Tree Search.
    The class name is preserved for backwards compatibility with the
    remainder of the project.
    """
    def __init__(
        self,
        cards: CardStats,
        your_deck_ids: list[int] | None = None,
        candidate_decks: list[CandidateDeck] | None = None,
        n_determinizations: int = 8,
        seed: int = 0,
        max_think_s: float = 0.5,
        max_candidates: int = 12,
        exploration: float = 1.414,
    ):
        self.cards = cards
        self.gamedata = GameData.load()
        self.feature_extractor = PlannerFeatureExtractor(
            self.gamedata,
        )
        self.weight_generator = DynamicWeightGenerator()
        self.evaluator = Evaluator(
            self.weight_generator,
        )
        self.your_deck_ids = list(your_deck_ids or [])
        self.k = n_determinizations
        self.max_think_s = max_think_s
        self.max_candidates = max_candidates
        self.exploration = exploration
        self.rng = random.Random(seed)
        self._engine = self._locate_engine()
        decks = candidate_decks or self._default_candidates()
        self.belief = (
            BeliefState(candidates=decks)
            if decks
            else None
        )
        
    def _locate_engine(self):
        for module_name in _ENGINE_MODULES:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            if all(
                hasattr(module, fn)
                for fn in _SEARCH_FUNCS
            ):
                return module
        return None
    def available(self) -> bool:
        return self._engine is not None

    def _default_candidates(self) -> list[CandidateDeck]:
        if not self.your_deck_ids:
            return []
        return [
            CandidateDeck(
                name="mirror",
                card_ids=list(self.your_deck_ids),
            )
        ]
        
    def choose(
        self,
        obs_dict,
        select: Select,
        deadline: float,
    ) -> list[int] | None:
        """
        Choose an action using Information Set Monte Carlo Tree Search.
        One search tree is shared across many sampled determinizations of the
        hidden game state.
        """
        if self._engine is None:
            return None
        if not select.options:
            return None
        deadline = min(
            deadline,
            time.monotonic() + self.max_think_s,
        )
        candidates = self._candidate_actions(select)
        if len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]
        if not candidates:
            return None
        # Update opponent belief using newly revealed information.
        self._update_belief(obs_dict)
        # Sample determinizations from the current belief.
        determinizations = self._build_determinizations(obs_dict)
        if not determinizations:
            return None
        # Root of the search tree.
        root = SearchNode()
        root.untried_actions = candidates.copy()
        # Main ISMCTS loop.
        while time.monotonic() < deadline:
            determinization = self.rng.choice(
                determinizations
            )
            self._mcts_iteration(
                root,
                obs_dict,
                determinization,
                deadline,
            )
        # Nothing explored.
        if not root.children:
            return None
        # Robust Child:
        # choose the action visited the most.
        best = max(
            root.children.values(),
            key=lambda c: (
                c.visits,
                c.value,
            ),
        )
        return best.action
    
    def _mcts_iteration(
        self,
        root: SearchNode,
        obs_dict,
        determinization,
        deadline,
    ):
        '''Execute one complete ISMCTS iteration.'''
        leaf, search_state = self._select(
            root,
            obs_dict,
            determinization,
        )
        if leaf is None or search_state is None:
            return
        reward = self._simulate(
            search_state,
            obs_dict,
            deadline,
        )
        leaf.backup(reward)
        # Free the CABT search state.
        try:
            self._engine.search_release(
                search_state.searchId,
            )
        except Exception:
            pass
        
    def _select(
        self,
        root: SearchNode,
        obs_dict,
        determinization,
    ):
        """
        Selection + Expansion.
        Reconstruct the determinized CABT state, descend the current search
        tree using UCB, expand one previously unseen action, and return the
        resulting leaf node together with the corresponding SearchState.
        """
        eng = self._engine
        if eng is None:
            return None, None
        try:
            observation = eng.to_observation_class(obs_dict)
        except Exception:
            return None, None
        args = self._search_args(
            obs_dict,
            determinization,
        )
        if args is None:
            return None, None
        (
            your_deck,
            your_prize,
            opp_deck,
            opp_prize,
            opp_hand,
            opp_active,
        ) = args
        try:
            state = eng.search_begin(
                observation,
                your_deck,
                your_prize,
                opp_deck,
                opp_prize,
                opp_hand,
                opp_active,
                False,
            )
        except Exception:
            return None, None
        node = root
        while True:
            # Terminal node.
            if node.terminal:
                return node, state
            # Expand one unvisited action.
            if not node.fully_expanded():
                action = node.untried_actions.pop()
                try:
                    state = eng.search_step(
                        state.searchId,
                        action,
                    )
                except Exception:
                    return None, None
                child = node.add_child(action)
                try:
                    obs_dict2 = dataclasses.asdict(
                        state.observation
                    )
                except Exception:
                    return child, state
                select = extract_select(obs_dict2)
                if (
                    select is not None
                    and select.options
                ):
                    child.untried_actions = (
                        self._candidate_actions(
                            select
                        )
                    )
                else:
                    child.terminal = True
                return child, state
            # Tree policy.
            child = node.best_child(
                self.exploration,
            )
            if child is None:
                return node, state
            try:
                state = eng.search_step(
                    state.searchId,
                    child.action,
                )
            except Exception:
                return None, None
            node = child
            
    def _simulate(
        self,
        search_state,
        root_obs_dict,
        deadline,
        max_depth: int = 48,
    ) -> float:
        """
        Roll out using a lightweight policy.
        The rollout intentionally avoids spending large amounts of time inside
        rules.choose(). ISMCTS benefits from many fast playouts rather than a few
        expensive ones.
        """
        eng = self._engine
        yi = your_index(root_obs_dict)
        state = search_state
        for _ in range(max_depth):
            if time.monotonic() >= deadline:
                break
            observation = state.observation
            current = getattr(
                observation,
                "current",
                None,
            )
            if current is None:
                break
            result = getattr(
                current,
                "result",
                -1,
            )
            # Terminal position.
            if isinstance(result, int) and result >= 0:
                if result == yi:
                    return 1.0
                if result == 2:
                    return 0.0
                return -1.0
            try:
                obs_dict = dataclasses.asdict(
                    observation
                )
            except Exception:
                break
            select = extract_select(obs_dict)
            if (
                select is None
                or not select.options
            ):
                break
            # Cheap rollout policy.
            if len(select.options) == 1:
                action = [0]
            else:
                try:
                    action = rules.choose(
                        obs_dict,
                        select,
                        self.gamedata,
                    )
                except Exception:
                    k = max(
                        select.min_count,
                        1,
                    )
                    action = list(range(k))
            try:
                state = eng.search_step(
                    state.searchId,
                    action,
                )
            except Exception:
                break
        return self._terminal_value(
            state,
            yi,
        )

    def _candidate_actions(
        self,
        select: Select,
    ) -> list[list[int]]:
        """
        Generate a diverse subset of legal actions.
        Small action spaces are enumerated exactly.
        Large action spaces are sampled without constructing every possible
        combination.
        """
        import itertools
        n = len(select.options)
        if n == 0:
            return [[]]
        indices = list(range(n))
        min_k = max(0, select.min_count)
        max_k = min(select.max_count, n)
        # Most CABT decisions are single-choice.
        if max_k <= 1:
            return [[i] for i in indices]
        actions: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        # Exact enumeration when still small.
        MAX_ENUM = 128
        for k in range(min_k, max_k + 1):
            total = math.comb(n, k)
            if total <= MAX_ENUM:
                for combo in itertools.combinations(indices, k):
                    actions.append(list(combo))
                continue
            # Otherwise randomly sample combinations.
            attempts = 0
            target = min(
                self.max_candidates,
                total,
            )
            while (
                len(actions) < target
                and attempts < target * 10
            ):
                combo = tuple(
                    sorted(
                        self.rng.sample(indices, k)
                    )
                )
                attempts += 1
                if combo in seen:
                    continue
                seen.add(combo)
                actions.append(list(combo))
        return actions
    
    def _update_belief(
        self,
        obs_dict,
    ) -> None:
        """
        Update the opponent belief using newly revealed public cards.
        """
        if self.belief is None:
            return
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return
        yi = your_index(obs_dict)
        opponent = _opponent_player(
            state,
            yi,
        )
        if not isinstance(opponent, dict):
            return
        for card_id in _visible_opponent_card_ids(opponent):
            try:
                self.belief.observe(card_id)
            except Exception:
                continue
            
    def _build_determinizations(
        self,
        obs_dict,
    ):
        """
        Sample determinizations consistent with the current belief.
        """
        if self.belief is None:
            return [None]
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return [None]
        yi = your_index(obs_dict)
        opponent = _opponent_player(
            state,
            yi,
        )
        if not isinstance(opponent, dict):
            return [None]
        hand_size = int(
            opponent.get("handCount") or 0
        )
        deck_size = int(
            opponent.get("deckCount") or 0
        )
        prize_size = len(
            [
                p
                for p in (
                    opponent.get("prize") or []
                )
                if p is None
            ]
        )
        if prize_size == 0:
            prize_size = 6
        try:
            samples = self.belief.sample_determinizations(
                self.k,
                hand_size,
                deck_size,
                prize_size,
                self.rng,
            )
            return samples or [None]
        except Exception:
            return [None]
        
    def _terminal_value(
        self,
        search_state,
        your_index_value,
    ) -> float:
        """
        Evaluate a leaf node using the same ValueNet/Evaluator pipeline as the
        planner. This keeps both search systems scoring positions identically.
        """
        observation = getattr(
            search_state,
            "observation",
            None,
        )
        if observation is None:
            return 0.0
        try:
            obs_dict = dataclasses.asdict(
                observation
            )
        except Exception:
            return 0.0
        try:
            from .evaluation import EvaluationContext
            features = self.feature_extractor.extract(
                obs_dict,
                self.your_deck_ids,
            )
            context = EvaluationContext(
                phase="search",
                turn_number=0,
                prizes_remaining=0,
                opponent_prizes_remaining=0,
                game_state={},
            )
            result = self.evaluator.evaluate(
                features,
                context,
            )
            return math.tanh(
                result.score / 60.0
            )
        except Exception:
            pass
        # Conservative fallback if feature extraction fails.
        current = getattr(
            observation,
            "current",
            None,
        )
        players = getattr(
            current,
            "players",
            None,
        )
        if (
            not isinstance(players, list)
            or len(players) < 2
        ):
            return 0.0
        me = players[your_index_value]
        opponent = players[
            1 - your_index_value
        ]
        my_remaining = sum(
            1
            for prize in (
                getattr(me, "prize", [])
                or []
            )
            if prize is None
        )
        opponent_remaining = sum(
            1
            for prize in (
                getattr(opponent, "prize", [])
                or []
            )
            if prize is None
        )
        return (
            opponent_remaining
            - my_remaining
        ) / 10.0
        
    def _search_args(
        self,
        obs_dict,
        determinization,
    ):
        """
        Construct the hidden zones required by CABT's search_begin().
        """
        if determinization is None:
            return None
        state = current_state(obs_dict)
        if not isinstance(state, dict):
            return None
        yi = your_index(obs_dict)
        me = _player(
            state,
            yi,
        )
        opponent = _opponent_player(
            state,
            yi,
        )
        select = (
            obs_dict.get("select")
            if isinstance(obs_dict, dict)
            else None
        )
        deck_visible = (
            isinstance(select, dict)
            and select.get("deck") is not None
        )
        my_deck_size = int(
            me.get("deckCount") or 0
        )
        my_prize_size = len(
            me.get("prize") or []
        )
        your_deck, your_prize = self._self_hidden(
            obs_dict,
            my_deck_size,
            my_prize_size,
        )
        if deck_visible:
            your_deck = []
        elif len(your_deck) < my_deck_size:
            return None
        if len(your_prize) < my_prize_size:
            return None
        opponent_deck = _pad(
            list(determinization.deck),
            int(opponent.get("deckCount") or 0),
            self.your_deck_ids,
        )
        opponent_prize = _pad(
            list(determinization.prize),
            len(opponent.get("prize") or []),
            self.your_deck_ids,
        )
        opponent_hand = _pad(
            list(determinization.hand),
            int(opponent.get("handCount") or 0),
            self.your_deck_ids,
        )
        if (
            opponent_deck is None
            or opponent_prize is None
            or opponent_hand is None
        ):
            return None
        opponent_active = []
        active = opponent.get("active") or []
        if active and active[0] is None:
            card_id = (
                determinization.active
                if determinization.active is not None
                else self._first_basic(
                    determinization,
                )
            )
            if card_id is None:
                return None
            opponent_active = [card_id]
        return (
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
        )
        
    def _self_hidden(
        self,
        obs_dict,
        deck_size: int,
        prize_size: int,
    ):
        """
        Infer our own hidden deck and prize cards from the registered deck.
        """
        if not self.your_deck_ids:
            return [], []
        unseen = Counter(
            self.your_deck_ids
        )
        state = current_state(obs_dict)
        if isinstance(state, dict):
            me = _player(
                state,
                your_index(obs_dict),
            )
            for card_id in _visible_self_card_ids(me):
                if unseen.get(card_id, 0) > 0:
                    unseen[card_id] -= 1
        remaining = list(
            unseen.elements()
        )
        self.rng.shuffle(
            remaining
        )
        prizes = remaining[:prize_size]
        deck = remaining[prize_size:]
        fallback = self.your_deck_ids[0]
        while len(prizes) < prize_size:
            prizes.append(fallback)
        while len(deck) < deck_size:
            deck.append(fallback)
        return deck, prizes
    
    def _first_basic(
        self,
        determinization,
    ):
        """
        Return the first Basic Pokémon found in the sampled hidden cards.
        """
        for card_id in (
            list(determinization.hand)
            + list(determinization.deck)
            + list(determinization.prize)
        ):
            if self.gamedata.is_basic_pokemon(
                card_id
            ):
                return card_id
        return None
    
# ----------------------------------------------------------------------
# State helper functions
# ----------------------------------------------------------------------
def _pad(
    values: list[int],
    target_size: int,
    fallback: list[int],
) -> list[int] | None:
    """
    Pad or trim a list so it contains exactly target_size card ids.
    """
    if target_size <= 0:
        return []
    if len(values) >= target_size:
        return values[:target_size]
    if not fallback:
        return None
    result = list(values)
    filler = fallback[0]
    while len(result) < target_size:
        result.append(filler)
    return result

def _players(state) -> list:
    """
    Return the player list from the state dictionary.
    """
    players = (
        state.get("players")
        if isinstance(state, dict)
        else None
    )
    if isinstance(players, list):
        return players
    return []

def _player(
    state,
    index,
):
    """
    Safely retrieve one player.
    """
    players = _players(state)
    if (
        0 <= index < len(players)
        and isinstance(players[index], dict)
    ):
        return players[index]
    return {}

def _opponent_player(
    state,
    your_index_value,
):
    """
    Return the opponent's player dictionary.
    """
    return _player(
        state,
        1 - your_index_value,
    )
    
def _card_id(card) -> int | None:
    """
    Extract the integer Card ID from a card dictionary.
    """
    if not isinstance(card, dict):
        return None
    card_id = card.get("cardId")
    if card_id is None:
        card_id = card.get("id")
    if isinstance(card_id, int):
        return card_id
    return None

def _pokemon_card_ids(mon) -> list[int]:
    """
    Return every public card belonging to one Pokémon:
    Pokémon card itself,
    attached Energy,
    attached Tool,
    previous evolution chain.
    """
    ids: list[int] = []
    if not isinstance(mon, dict):
        return ids
    pokemon_id = mon.get("id")
    if isinstance(pokemon_id, int):
        ids.append(pokemon_id)
    for key in (
        "energyCards",
        "tools",
    ):
        for card in mon.get(key) or []:
            cid = _card_id(card)
            if cid is not None:
                ids.append(cid)
    for card in mon.get("preEvolution") or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    return ids

def _visible_opponent_card_ids(opponent) -> list[int]:
    """
    Collect every opponent card that is public knowledge.
    """
    ids: list[int] = []
    for card in opponent.get("discard") or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    for area in (
        "active",
        "bench",
    ):
        for pokemon in opponent.get(area) or []:
            ids.extend(
                _pokemon_card_ids(pokemon)
            )
    return ids

def _visible_self_card_ids(me) -> list[int]:
    """
    Collect every one of our own visible cards.
    """
    ids: list[int] = []
    for card in me.get("hand") or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    for card in me.get("discard") or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    for area in (
        "active",
        "bench",
    ):
        for pokemon in me.get(area) or []:
            ids.extend(
                _pokemon_card_ids(pokemon)
            )
    return ids