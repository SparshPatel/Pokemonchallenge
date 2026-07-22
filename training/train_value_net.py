from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from replay_parser import ReplayParser
from replay_loader import ReplayLoader
from dataset_builder import DatasetBuilder
from feature_extractor import FeatureExtractor
# =====================================================================
# Configuration
# =====================================================================
REPLAY_FOLDER = Path("../replays")
MODEL_OUT = Path("../submission/agent/value_net.pt")
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 5e-4
GAME_VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42
# ---------------------------------------------------------------------
# Discount factor search
# ---------------------------------------------------------------------
INITIAL_GAMMA = 0.995
GAMMA_SEARCH = [
    0.90,
    0.93,
    0.95,
    0.97,
    0.985,
    0.99,
    0.995,
    0.998,
]
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# =====================================================================
# Network
# =====================================================================
class ValueNet(nn.Module):
    """
    Runtime-compatible ValueNet.
    Architecture:
        input
          ↓
        Linear
          ↓
        tanh
          ↓
        Linear
          ↓
        scalar
    This architecture intentionally matches
    submission/agent/value_net.py exactly.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )
        
    def forward(
        self,
        x,
    ):
        return self.network(x)

# =====================================================================
# Discount Optimizer
# =====================================================================
class DiscountOptimizer:
    def __init__(
        self,
        candidates=None,
    ):
        self.candidates = (
            candidates
            if candidates is not None
            else GAMMA_SEARCH
        )
        self.best_gamma = INITIAL_GAMMA
        self.best_loss = float("inf")

    def __iter__(
        self,
    ):
        return iter(
            self.candidates
        )

    def update(
        self,
        gamma: float,
        validation_loss: float,
    ):
        if validation_loss < self.best_loss:
            self.best_loss = (
                validation_loss
            )
            self.best_gamma = gamma

    def summary(
        self,
    ):
        print()
        print(
            "=" * 70
        )
        print(
            "Discount Optimizer"
        )
        print(
            "=" * 70
        )
        print()
        print(
            "Best Gamma :",
            self.best_gamma,
        )
        print(
            "Best Loss  :",
            self.best_loss,
        )
        print()

# =====================================================================
# Dataset
# =====================================================================
class TrainingDataset:
    def __init__(
        self,
        gamma: float,
    ):
        parser = ReplayParser(
            REPLAY_FOLDER
        )
        parser.parse()
        loader = ReplayLoader()
        loader.load_from_parser(
            parser
        )
        builder = DatasetBuilder()
        samples = builder.build(
            loader.games
        )
        # -------------------------------------------------------------
        # Assign discounted return.
        # -------------------------------------------------------------
        grouped = {}
        for sample in samples:
            grouped.setdefault(
                sample.game_index,
                [],
            ).append(
                sample
            )
        for game_samples in grouped.values():
            game_samples.sort(
                key=lambda s: s.turn_index
            )
            terminal_reward = (
                game_samples[-1].reward
            )
            value = terminal_reward
            for sample in reversed(
                game_samples
            ):
                sample.reward = value
                value *= gamma
        # -------------------------------------------------------------
        # Extract features.
        # -------------------------------------------------------------
        extractor = FeatureExtractor()
        features = (
            extractor.extract_dataset(
                samples
            )
        )
        X, y = (
            extractor.feature_matrix(
                features
            )
        )
        self.X = np.asarray(
            X,
            dtype=np.float32,
        )
        self.y = np.asarray(
            y,
            dtype=np.float32,
        )
        self.samples = len(
            self.X
        )
        self.feature_dim = (
            self.X.shape[1]
        )
        # -------------------------------------------------------------
        # Game IDs for leakage-free splitting.
        # -------------------------------------------------------------
        self.game_ids = np.asarray(
            [
                sample.game_index
                for sample in samples
            ],
            dtype=np.int64,
        )
        print()
        print(
            "=" * 70
        )
        print(
            "Training Dataset"
        )
        print(
            "=" * 70
        )
        print()
        print(
            "Samples :",
            self.samples,
        )
        print(
            "Features:",
            self.feature_dim,
        )
        print(
            "Games   :",
            len(
                np.unique(
                    self.game_ids
                )
            ),
        )
        print()
        print(
            "Targets"
        )
        print(
            " Min :",
            float(
                self.y.min()
            ),
        )
        print(
            " Max :",
            float(
                self.y.max()
            ),
        )
        print(
            " Mean:",
            float(
                self.y.mean()
            ),
        )

# =====================================================================
# Trainer
# =====================================================================
class Trainer:
    def __init__(
        self,
        gamma: float,
    ):
        random.seed(
            RANDOM_SEED
        )
        np.random.seed(
            RANDOM_SEED
        )
        torch.manual_seed(
            RANDOM_SEED
        )
        self.gamma = gamma
        self.dataset = TrainingDataset(
            gamma=gamma
        )
        # -------------------------------------------------------------
        # Split by GAME, not individual samples.
        # -------------------------------------------------------------
        unique_games = np.unique(
            self.dataset.game_ids
        )
        rng = np.random.default_rng(
            RANDOM_SEED
        )
        rng.shuffle(
            unique_games
        )
        split = int(
            (
                1.0
                - GAME_VALIDATION_SPLIT
            )
            * len(
                unique_games
            )
        )
        train_games = set(
            unique_games[
                :split
            ]
        )
        valid_games = set(
            unique_games[
                split:
            ]
        )
        train_mask = np.asarray(
            [
                gid in train_games
                for gid
                in self.dataset.game_ids
            ],
            dtype=bool,
        )
        valid_mask = np.asarray(
            [
                gid in valid_games
                for gid
                in self.dataset.game_ids
            ],
            dtype=bool,
        )
        train_x = torch.tensor(
            self.dataset.X[
                train_mask
            ],
            dtype=torch.float32,
        )
        train_y = torch.tensor(
            self.dataset.y[
                train_mask
            ],
            dtype=torch.float32,
        ).unsqueeze(
            1
        )
        valid_x = torch.tensor(
            self.dataset.X[
                valid_mask
            ],
            dtype=torch.float32,
        )
        valid_y = torch.tensor(
            self.dataset.y[
                valid_mask
            ],
            dtype=torch.float32,
        ).unsqueeze(
            1
        )
        print()
        print(
            "Game Split"
        )
        print(
            "Train games:",
            len(train_games),
        )
        print(
            "Valid games:",
            len(valid_games),
        )
        print(
            "Train samples:",
            len(train_x),
        )
        print(
            "Valid samples:",
            len(valid_x),
        )
        # -------------------------------------------------------------
        # Data loaders.
        # -------------------------------------------------------------
        self.train_loader = DataLoader(
            TensorDataset(
                train_x,
                train_y,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        self.valid_loader = DataLoader(
            TensorDataset(
                valid_x,
                valid_y,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        # -------------------------------------------------------------
        # Runtime-compatible model.
        # -------------------------------------------------------------
        self.model = ValueNet(
            self.dataset.feature_dim,
            hidden_dim=32,
        ).to(
            DEVICE
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )
        self.loss_fn = nn.MSELoss()

    # =================================================================
    # Train
    # =================================================================
    def train_epoch(
        self,
    ):
        self.model.train()
        total = 0.0
        count = 0
        for x, y in self.train_loader:
            x = x.to(
                DEVICE
            )
            y = y.to(
                DEVICE
            )
            pred = self.model(
                x
            )
            loss = self.loss_fn(
                pred,
                y,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            count += 1
        return (
            total
            / max(
                count,
                1,
            )
        )
    # =================================================================
    # Validate
    # =================================================================
    @torch.no_grad()
    def validate(
        self,
    ):
        self.model.eval()
        total = 0.0
        count = 0
        for x, y in self.valid_loader:
            x = x.to(
                DEVICE
            )
            y = y.to(
                DEVICE
            )
            pred = self.model(
                x
            )
            loss = self.loss_fn(
                pred,
                y,
            )
            total += loss.item()
            count += 1
        return (
            total
            / max(
                count,
                1,
            )
        )
        
    # =================================================================
    # Export
    # =================================================================
    def export_runtime_model(
        self,
    ):
        self.model.eval()
        state = (
            self.model.state_dict()
        )
        # PyTorch Linear:
        # weight shape:
        #   [out_features, in_features]
        # Runtime numpy model expects:
        #   [in_features, out_features]
        W1 = (
            state[
                "network.0.weight"
            ]
            .detach()
            .cpu()
            .numpy()
            .T
        )
        b1 = (
            state[
                "network.0.bias"
            ]
            .detach()
            .cpu()
            .numpy()
        )
        W2 = (
            state[
                "network.2.weight"
            ]
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1
            )
        )
        b2 = (
            state[
                "network.2.bias"
            ]
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1
            )
        )
        npz_path = (
            MODEL_OUT.with_suffix(
                ".npz"
            )
        )
        np.savez(
            npz_path,
            kind="mlp",
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
        )
        print()
        print(
            "Exported runtime model:"
        )
        print(
            npz_path
        )
        print()
        print(
            "W1:",
            W1.shape,
        )
        print(
            "b1:",
            b1.shape,
        )
        print(
            "W2:",
            W2.shape,
        )
        print(
            "b2:",
            b2.shape,
        )

    # =================================================================
    # Fit
    # =================================================================
    def fit(
        self,
    ):
        best_loss = float(
            "inf"
        )
        best_state = None
        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            train_loss = (
                self.train_epoch()
            )
            valid_loss = (
                self.validate()
            )
            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.6f} | "
                f"Valid {valid_loss:.6f}"
            )
            if valid_loss < best_loss:
                best_loss = (
                    valid_loss
                )
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in self.model.state_dict().items()
                }
        # -------------------------------------------------------------
        # Restore best validation checkpoint.
        # -------------------------------------------------------------
        if best_state is not None:
            self.model.load_state_dict(
                best_state
            )
        MODEL_OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            self.model.state_dict(),
            MODEL_OUT,
        )
        self.export_runtime_model()
        print()
        print(
            "Best validation loss:",
            best_loss,
        )
        print(
            "Saved PyTorch model:",
            MODEL_OUT,
        )
        return best_loss
    
# =====================================================================
# Gamma Search
# =====================================================================
class GammaSearch:
    def __init__(
        self,
    ):
        self.optimizer = (
            DiscountOptimizer()
        )

    def run(
        self,
    ):
        global_best_loss = float(
            "inf"
        )
        global_best_gamma = (
            INITIAL_GAMMA
        )
        print()
        print(
            "=" * 70
        )
        print(
            "Gamma Search"
        )
        print(
            "=" * 70
        )
        for gamma in self.optimizer:
            print()
            print(
                "=" * 70
            )
            print(
                f"Training gamma = {gamma:.4f}"
            )
            print(
                "=" * 70
            )
            trainer = Trainer(
                gamma
            )
            loss = trainer.fit()
            self.optimizer.update(
                gamma,
                loss,
            )
            if loss < global_best_loss:
                global_best_loss = (
                    loss
                )
                global_best_gamma = (
                    gamma
                )
        print()
        self.optimizer.summary()
        print(
            "Selected Gamma :",
            global_best_gamma,
        )
        print(
            "Validation Loss:",
            global_best_loss,
        )
        print(
            "Model saved to :",
            MODEL_OUT,
        )

# =====================================================================
# Main
# =====================================================================
def main():
    GammaSearch().run()

if __name__ == "__main__":
    main()