from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    """All non-source project paths derived from one workspace root."""

    root: Path = Path("workspace")

    @property
    def data_root(self) -> Path:
        return self.root / "data"

    @property
    def dataset(self) -> Path:
        return self.data_root / "dataset"

    @property
    def anchors(self) -> Path:
        return self.data_root / "anchors"

    @property
    def class_config(self) -> Path:
        return self.data_root / "dataset_classes.json"

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    @property
    def checkpoints(self) -> Path:
        return self.results_root / "checkpoints"

    @property
    def transcripts(self) -> Path:
        return self.results_root / "transcripts"

    @property
    def reports(self) -> Path:
        return self.results_root / "reports"

    @property
    def runs(self) -> Path:
        return self.results_root / "runs"

    @property
    def tests(self) -> Path:
        return self.root / "tests"


DEFAULT_WORKSPACE_PATHS = WorkspacePaths()

__all__ = ["WorkspacePaths", "DEFAULT_WORKSPACE_PATHS"]
