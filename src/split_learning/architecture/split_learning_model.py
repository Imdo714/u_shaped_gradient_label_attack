from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..f_model.client_front_f_model import ClientFrontFModel
from ..g_model.server_middle_g_model import ServerMiddleGModel
from ..h_model.client_tail_h_model import ClientTailHModel


@dataclass
class SplitLearningModel:
    """Container that makes the U-shaped f -> g -> h ownership explicit."""

    f_model: ClientFrontFModel
    g_model: ServerMiddleGModel
    h_model: ClientTailHModel
    cut_config: str = "middle"

    # Compatibility properties preserve older attribute access.
    @property
    def client_front(self) -> ClientFrontFModel:
        return self.f_model

    @property
    def server_middle(self) -> ServerMiddleGModel:
        return self.g_model

    @property
    def client_tail(self) -> ClientTailHModel:
        return self.h_model

    def to(self, device: torch.device | str) -> "SplitLearningModel":
        self.f_model.to(device)
        self.g_model.to(device)
        self.h_model.to(device)
        return self

    def train(self, mode: bool = True) -> "SplitLearningModel":
        self.f_model.train(mode)
        self.g_model.train(mode)
        self.h_model.train(mode)
        return self

    def eval(self) -> "SplitLearningModel":
        return self.train(False)

    def predict(self, x: Tensor) -> Tensor:
        """Run ordinary inference without simulating communication boundaries."""
        return self.h_model(self.g_model(self.f_model(x)))

    def state_dict(self) -> dict[str, object]:
        """Return checkpoint state using the established keys for compatibility."""
        return {
            "client_front": self.f_model.state_dict(),
            "server_middle": self.g_model.state_dict(),
            "client_tail": self.h_model.state_dict(),
            "cut_config": self.cut_config,
            "num_classes": self.h_model.num_classes,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.f_model.load_state_dict(state["client_front"])  # type: ignore[arg-type]
        self.g_model.load_state_dict(state["server_middle"])  # type: ignore[arg-type]
        self.h_model.load_state_dict(state["client_tail"])  # type: ignore[arg-type]

    def save(self, path: str | Path, **metadata: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "metadata": metadata}, path)


def build_split_learning_model(
    cut_config: str = "middle",
    num_classes: int = 2,
) -> SplitLearningModel:
    """Build the client f model, server g model, and client h model."""
    return SplitLearningModel(
        f_model=ClientFrontFModel(cut_config),
        g_model=ServerMiddleGModel(cut_config),
        h_model=ClientTailHModel(num_classes),
        cut_config=cut_config,
    )


def _infer_num_classes(state: dict[str, object]) -> int:
    saved = state.get("num_classes")
    if saved is not None:
        return int(saved)
    tail_state = state["client_tail"]
    if not isinstance(tail_state, dict):
        raise ValueError("Invalid client_tail checkpoint state")
    for key, value in reversed(list(tail_state.items())):
        if key.endswith(".weight") and isinstance(value, Tensor) and value.ndim == 2:
            return int(value.shape[0])
    raise ValueError("Cannot infer the number of classes from the checkpoint")


def load_split_learning_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SplitLearningModel, dict]:
    """Restore f/g/h models while accepting checkpoints created before this refactor."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    model = build_split_learning_model(
        str(state.get("cut_config", "middle")),
        num_classes=_infer_num_classes(state),
    )
    model.load_state_dict(state)
    model.to(device)
    return model, checkpoint.get("metadata", {})


# Compatibility aliases for external code using the earlier public API.
SplitModel = SplitLearningModel
build_split_model = build_split_learning_model
load_split_model = load_split_learning_model

__all__ = [
    "SplitLearningModel",
    "build_split_learning_model",
    "load_split_learning_model",
    "SplitModel",
    "build_split_model",
    "load_split_model",
]
