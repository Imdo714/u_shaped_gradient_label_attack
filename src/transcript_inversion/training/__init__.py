"""Trainers for paired controls and architecture-agnostic ABTR."""

from .abtr_trainer import ABTRConfig, ABTRTrainer
from .paired_decoder_trainer import PairedDecoderConfig, PairedDecoderTrainer
from .tail_simulator_trainer import TailSimulatorConfig, TailSimulatorTrainer

__all__ = [
    "ABTRConfig",
    "ABTRTrainer",
    "PairedDecoderConfig",
    "PairedDecoderTrainer",
    "TailSimulatorConfig",
    "TailSimulatorTrainer",
]
