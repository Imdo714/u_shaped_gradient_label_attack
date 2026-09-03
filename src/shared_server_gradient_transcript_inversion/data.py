from __future__ import annotations

from torch import Tensor, zeros_like
from torch.utils.data import Dataset

from .conditions import SignalCondition


class ConditionedTranscriptDataset(Dataset):
    """Present exactly one signal condition while retaining a common schema.

    The reused decoder has fixed ``u`` and ``dL/dz`` branches.  A hidden branch
    receives an all-zero tensor, so dimensions and parameter counts stay equal
    across A/B/C without exposing the omitted observation.
    """

    def __init__(self, dataset: Dataset, condition: SignalCondition) -> None:
        if len(dataset) < 1:
            raise ValueError("transcript dataset is empty")
        self.dataset = dataset
        self.condition = condition
        sample = dataset[0]
        if condition.use_z and not isinstance(sample.get("smashed_z"), Tensor):
            raise ValueError(
                "condition D requires records captured with --capture-z"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        if not self.condition.use_u:
            item["server_output_u"] = zeros_like(item["server_output_u"])
        if not self.condition.use_grad_z:
            item["grad_g_to_f"] = zeros_like(item["grad_g_to_f"])
        if not self.condition.use_z:
            item.pop("smashed_z", None)
        return item


__all__ = ["ConditionedTranscriptDataset"]
