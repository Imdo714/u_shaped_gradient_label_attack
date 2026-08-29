from __future__ import annotations

import random
from enum import Enum

import torch
from torch.utils.data import Dataset


class PairingMode(str, Enum):
    EXACT = "exact"
    CLASS_SHUFFLED = "class-shuffled"
    GLOBAL_SHUFFLED = "global-shuffled"


def _deranged(indices: list[int], generator: random.Random) -> list[int]:
    if len(indices) < 2:
        return indices.copy()
    candidate = indices.copy()
    for _ in range(100):
        generator.shuffle(candidate)
        if all(left != right for left, right in zip(indices, candidate)):
            return candidate
    return indices[1:] + indices[:1]


class PairingAblationDataset(Dataset):
    """P0-P3 paired controls with deterministic pair assignment.

    The source must expose evaluator targets.  ``paired_fraction`` implements
    the limited-pair P1 condition; class/global shuffling implement P2/P3.
    """

    def __init__(
        self,
        source: Dataset,
        mode: PairingMode | str = PairingMode.EXACT,
        paired_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 < paired_fraction <= 1.0:
            raise ValueError("paired_fraction must be in (0, 1]")
        self.source = source
        self.mode = PairingMode(mode)
        generator = random.Random(seed)
        count = max(1, round(len(source) * paired_fraction))
        selected = list(range(len(source)))
        generator.shuffle(selected)
        self.source_indices = sorted(selected[:count])
        self.target_indices = self._make_target_indices(generator)

    def _make_target_indices(self, generator: random.Random) -> list[int]:
        if self.mode == PairingMode.EXACT:
            return self.source_indices.copy()
        if self.mode == PairingMode.GLOBAL_SHUFFLED:
            return _deranged(self.source_indices, generator)

        groups: dict[int, list[int]] = {}
        for index in self.source_indices:
            item = self.source[index]
            if "true_label" not in item:
                raise KeyError("class-shuffled pairing requires true_label")
            label = int(item["true_label"])
            groups.setdefault(label, []).append(index)
        mapping: dict[int, int] = {}
        for indices in groups.values():
            mapping.update(zip(indices, _deranged(indices, generator)))
        return [mapping[index] for index in self.source_indices]

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self.source_indices[index]
        target_index = self.target_indices[index]
        source_item = dict(self.source[source_index])
        target_item = self.source[target_index]
        if "target_image" not in target_item:
            raise KeyError("pairing ablations require a target-enabled source dataset")
        source_item["target_image"] = target_item["target_image"]
        source_item["paired_true_label"] = target_item["true_label"]
        source_item["source_index"] = torch.tensor(source_index)
        source_item["target_index"] = torch.tensor(target_index)
        source_item["pair_is_exact"] = torch.tensor(source_index == target_index)
        return source_item


__all__ = ["PairingAblationDataset", "PairingMode"]
