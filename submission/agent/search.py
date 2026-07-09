from __future__ import annotations
import math
from dataclasses import dataclass, field

@dataclass
class SearchNode:
    """
    Generic Monte Carlo Tree Search node.
    This class intentionally stores only tree statistics.
    It contains no CABT state.
    """
    parent: "SearchNode | None" = None
    action: list[int] | None = None
    children: dict[
        tuple[int, ...],
        "SearchNode",
    ] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    untried_actions: list[list[int]] = field(
        default_factory=list,
    )
    terminal: bool = False
    depth: int = 0

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(
        self,
        action: list[int],
    ) -> "SearchNode":
        child = SearchNode(
            parent=self,
            action=action,
            depth=self.depth + 1,
        )
        self.children[tuple(action)] = child
        return child

    def ucb_score(
        self,
        exploration: float = math.sqrt(2.0),
    ) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return (
            self.value
            + exploration
            * math.sqrt(
                math.log(
                    max(1, self.parent.visits)
                )
                / self.visits
            )
        )

    def best_child(
        self,
        exploration: float = math.sqrt(2.0),
    ) -> "SearchNode":
        """
        Return the child with the highest UCB score.
        Break ties randomly so insertion order does not bias search.
        """
        import random
        best_score = float("-inf")
        best = []
        for child in self.children.values():
            score = child.ucb_score(exploration)
            if score > best_score:
                best_score = score
                best = [child]
            elif score == best_score:
                best.append(child)
        return random.choice(best)

    def backup(
        self,
        reward: float,
    ):
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

@dataclass
class SearchResult:
    """
    Optional statistics produced by a search.
    """
    action: list[int]
    simulations: int
    tree_size: int
    max_depth: int
    root_value: float

def tree_size(root: SearchNode) -> int:
    """
    Count nodes in the tree.
    """
    stack = [root]
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(
            node.children.values()
        )
    return count