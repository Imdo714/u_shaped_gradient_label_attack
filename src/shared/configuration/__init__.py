"""Experiment configuration shared by training and experiments."""

from .experiment_config import ExperimentConfig
from .workspace_paths import DEFAULT_WORKSPACE_PATHS, WorkspacePaths

__all__ = ["ExperimentConfig", "WorkspacePaths", "DEFAULT_WORKSPACE_PATHS"]
