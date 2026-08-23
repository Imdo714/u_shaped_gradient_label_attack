"""Explicit forward activations and backward gradient exchange."""

from .gradient_exchange import (
    COMMUNICATION_DIAGRAM,
    GradientExchangeResult,
    StepResult,
    explicit_split_step,
    frozen_gradient_observation,
    observe_frozen_gradient_exchange,
    run_gradient_exchange_step,
)

__all__ = [
    "COMMUNICATION_DIAGRAM",
    "GradientExchangeResult",
    "run_gradient_exchange_step",
    "observe_frozen_gradient_exchange",
    "StepResult",
    "explicit_split_step",
    "frozen_gradient_observation",
]
