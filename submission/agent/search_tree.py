"""
Generic AlphaZero/MCTS search tree.
This module intentionally contains NO Pokémon-specific logic.
Responsibilities
----------------
• Tree node storage
• Search statistics
• Transposition table
• Planner caches
• Virtual loss
• Backpropagation
The planner owns search flow.
This module owns search state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
# ---------------------------------------------------------
# Planner caches
# ---------------------------------------------------------
@dataclass(slots=True)
class PlannerCache:
    """
    Shared caches used during one search.
    feature:
        state -> feature vector
    evaluation:
        state -> scalar evaluation
    search:
        (state_key, depth) -> backed-up value
    """
    feature: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)
    def clear(self) -> None:
        self.feature.clear()
        self.evaluation.clear()
        self.search.clear()

# ---------------------------------------------------------
# Tree Node
# ---------------------------------------------------------
@dataclass(slots=True)
class TreeNode:
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

# ---------------------------------------------------------
# Search Context
# ---------------------------------------------------------
@dataclass(slots=True)
class SearchContext:
    cache: PlannerCache = field(default_factory=PlannerCache)
    tree: dict[int, TreeNode] = field(default_factory=dict)
    transposition: dict = field(default_factory=dict)
    deadline: float = 0.0
    nodes: int = 0
    def reset(self):
        self.cache.clear()
        self.tree.clear()
        self.transposition.clear()
        self.nodes = 0

# ---------------------------------------------------------
# Search Tree
# ---------------------------------------------------------
class SearchTree:
    """
    Generic search-tree manager.
    This class knows nothing about Pokémon.
    It only manages statistics.
    """
    def __init__(self):
        self.ctx = SearchContext()
    # -----------------------------------------------------
    
    def node(self, search_id: int) -> TreeNode:
        node = self.ctx.tree.get(search_id)
        if node is None:
            node = TreeNode()
            self.ctx.tree[search_id] = node
        return node

    # -----------------------------------------------------
    def lookup_transposition(
        self,
        key,
        search_id,
    ):
        existing = self.ctx.transposition.get(key)
        if existing is None:
            self.ctx.transposition[key] = search_id
            return search_id
        if existing != search_id:
            self.ctx.tree.setdefault(
                existing,
                self.node(search_id),
            )
            return existing
        return existing

    # -----------------------------------------------------
    def apply_virtual_loss(
        self,
        path,
    ):
        for sid in path:
            self.node(sid).virtual_visits += 1
    # -----------------------------------------------------

    def revert_virtual_loss(
        self,
        path,
    ):
        for sid in path:
            node = self.node(sid)
            if node.virtual_visits > 0:
                node.virtual_visits -= 1
    # -----------------------------------------------------

    def backup(
        self,
        path,
        value,
        discount: float = 0.995,
    ):
        for sid in reversed(path):
            node = self.node(sid)
            node.visits += 1
            node.value_sum += value
            value *= discount

    # -----------------------------------------------------
    def clear(self):
        self.ctx.reset()