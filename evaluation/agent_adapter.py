from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
# =====================================================================
# Agent Adapter
# =====================================================================
@dataclass(slots=True)
class AgentAdapter:
    """
    Generic adapter around the agent being evaluated.
    The tournament engine does not need to know how the agent internally
    works. It only needs a callable that receives the current observation
    and returns the selected option indices.
    Expected callable:
        agent_fn(observation) -> list[int]
    The observation is the raw dictionary returned by the CG engine.
    The returned list contains indices into observation["select"]["option"].
    """
    name: str
    agent_fn: Callable[
        [dict[str, Any]],
        list[int],
    ]
    def choose(
        self,
        observation: dict[str, Any],
    ) -> list[int]:
        """
        Ask the evaluated agent to choose an action.
        The adapter deliberately performs only generic validation.
        It does not interpret Pokémon-specific actions.
        """
        if observation is None:
            raise ValueError(
                "Agent received a None observation."
            )
        result = self.agent_fn(
            observation
        )
        if not isinstance(
            result,
            list,
        ):
            raise TypeError(
                "Agent must return list[int]. "
                f"Received {type(result).__name__}."
            )
        if not all(
            isinstance(
                index,
                int,
            )
            for index in result
        ):
            raise TypeError(
                "Agent selection must be list[int]."
            )
        return result

# =====================================================================
# Factory
# =====================================================================
def from_callable(
    name: str,
    agent_fn: Callable[
        [dict[str, Any]],
        list[int],
    ],
) -> AgentAdapter:
    """
    Construct an AgentAdapter from any compatible callable.
    This is the preferred entry point when the agent is already imported
    as a Python function.
    """
    if not name:
        raise ValueError(
            "Agent name cannot be empty."
        )
    if not callable(agent_fn):
        raise TypeError(
            "agent_fn must be callable."
        )
    return AgentAdapter(
        name=name,
        agent_fn=agent_fn,
    )

# =====================================================================
# Module Loader
# =====================================================================
def load_agent(
    module_name: str,
    function_name: str,
    agent_name: str | None = None,
) -> AgentAdapter:
    """
    Dynamically load an agent function.
    Example:
        adapter = load_agent(
            module_name="submission.agent.agent",
            function_name="agent",
            agent_name="candidate",
        )
    This keeps the evaluation pipeline independent of the concrete
    location of the submitted agent.
    """
    import importlib
    module = importlib.import_module(
        module_name
    )
    agent_fn = getattr(
        module,
        function_name,
        None,
    )
    if agent_fn is None:
        raise AttributeError(
            f"Module '{module_name}' does not expose "
            f"'{function_name}'."
        )
    return from_callable(
        name=(
            agent_name
            if agent_name is not None
            else f"{module_name}.{function_name}"
        ),
        agent_fn=agent_fn,
    )