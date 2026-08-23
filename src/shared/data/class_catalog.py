from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ClassCatalog:
    """Immutable semantic-class registry shared by training and attack stages."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) < 2:
            raise ValueError("At least two classes are required")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Class names must be unique: {self.names}")
        if tuple(sorted(self.names)) != self.names:
            raise ValueError(
                "Class names must be alphabetically sorted to match torchvision ImageFolder"
            )
        for name in self.names:
            if not name or name != name.strip() or Path(name).name != name:
                raise ValueError(f"Invalid class name: {name!r}")

    @classmethod
    def discover(cls, data_dir: str | Path, split: str = "train") -> "ClassCatalog":
        root = Path(data_dir) / split
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {root}")
        names = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
        return cls(names)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "ClassCatalog":
        return cls(tuple(names))

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.names)}

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(name.upper() for name in self.names)

    def label(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as error:
            raise ValueError(f"Unknown class {name!r}; expected one of {self.names}") from error

    def anchor_paths(self, anchor_dir: str | Path) -> tuple[Path, ...]:
        root = Path(anchor_dir)
        return tuple(root / name / f"{name}_anchor.jpg" for name in self.names)


def checkpoint_class_catalog(
    metadata: Mapping[str, object], num_classes: int
) -> ClassCatalog:
    raw_names = metadata.get("class_names")
    if raw_names is None:
        config = metadata.get("config", {})
        if isinstance(config, Mapping):
            raw_names = config.get("class_names")
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        catalog = ClassCatalog.from_names([str(name) for name in raw_names])
        if catalog.num_classes != num_classes:
            raise ValueError(
                f"Checkpoint has {num_classes} outputs but {catalog.num_classes} class names"
            )
        return catalog
    if num_classes == 2:
        return ClassCatalog(("cat", "dog"))
    raise ValueError("Checkpoint is missing class_names metadata")
