from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None
    
# -------------------------------------------------------
# Default heuristic weights
# -------------------------------------------------------
DEFAULT_WEIGHTS = np.array(
    [
        3.0,   # prize
        1.8,   # my hp
        -1.8,  # opp hp
        2.2,   # board
        1.6,   # energy
        2.5,   # attack ready
        1.2,   # tempo
        2.8,   # ko pressure
        1.3,   # hand
        0.8,   # bench
        1.0,   # evolution
        0.6,   # supporters
    ],
    dtype=np.float32,
)

# -------------------------------------------------------
# Context passed from planner
# -------------------------------------------------------
@dataclass(slots=True)
class WeightContext:
    game_phase: float
    prize_diff: float
    search_depth: int
    search_confidence: float
    opponent_embedding: Optional[np.ndarray] = None

# -------------------------------------------------------
# Neural model
# -------------------------------------------------------
if nn is not None:
    class WeightNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(DEFAULT_WEIGHTS)),
            )

        def forward(self, x):
            return self.net(x)
else:
    WeightNetwork = None

# -------------------------------------------------------
# WeightModel
# -------------------------------------------------------
class WeightModel:
    """
    Predicts evaluator weights from game context.
    If unavailable, falls back to DEFAULT_WEIGHTS.
    """
    def __init__(
        self,
        checkpoint: str | None = None,
    ):
        self.available = False
        self.device = None
        self.model = None
        if (
            torch is None
            or WeightNetwork is None
        ):
            return
        self.device = torch.device("cpu")
        self.model = WeightNetwork().to(self.device)
        if checkpoint is None:
            checkpoint = (
                Path(__file__).parent
                / "models"
                / "weight_model.pt"
            )
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            try:
                state = torch.load(
                    checkpoint,
                    map_location=self.device,
                )
                self.model.load_state_dict(state)
                self.model.eval()
                self.available = True
            except Exception:
                self.available = False

    def _build_input(
        self,
        features,
        context: WeightContext,
    ):
        x = np.zeros(
            16,
            dtype=np.float32,
        )
        x[0] = context.game_phase
        x[1] = context.search_depth
        x[2] = context.search_confidence
        x[3] = context.prize_diff
        n = min(
            len(features),
            8,
        )
        x[4:4+n] = features[:n]
        if context.opponent_embedding is not None:
            emb = np.asarray(
                context.opponent_embedding,
                dtype=np.float32,
            )
            m = min(
                len(emb),
                4,
            )
            x[12:12+m] = emb[:m]
        return x

    def predict(
        self,
        features,
        context: WeightContext,
    ):
        if (
            not self.available
            or self.model is None
        ):
            return DEFAULT_WEIGHTS.copy()
        x = self._build_input(
            features,
            context,
        )
        with torch.no_grad():
            tensor = (
                torch.from_numpy(x)
                .unsqueeze(0)
                .to(self.device)
            )
            weights = (
                self.model(tensor)
                .cpu()
                .numpy()[0]
            )
        return weights.astype(np.float32)