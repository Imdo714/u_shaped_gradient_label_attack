from __future__ import annotations

import pytest
import torch

from src.shared_pretrained_encoder_drift_attack.models import decoder_for_condition
from src.shared_pretrained_encoder_drift_attack.pipeline.run_drift_attack import (
    ATTACK_CONDITIONS,
    build_parser,
)


@pytest.mark.parametrize("condition", ATTACK_CONDITIONS)
def test_drift_attack_decoder_conditions_have_expected_inputs(condition):
    model = decoder_for_condition(
        condition,
        z_channels=8,
        image_size=16,
        signal_spatial_size=4,
        signal_channels=8,
        base_channels=16,
        min_channels=8,
    )
    z = torch.randn(2, 8, 4, 4)
    gradient = torch.randn(2, 8, 4, 4)
    reconstruction = model(z, gradient)
    assert reconstruction.shape == (2, 3, 16, 16)
    reconstruction.mean().backward()
    assert (model.z_encoder is not None) == (condition != "gradient_only")
    assert (model.gradient_encoder is not None) == (condition != "z_only")


def test_drift_pipeline_defaults_capture_initial_and_final_epochs():
    args = build_parser().parse_args(["--pretrained-autoencoder", "encoder.pt"])
    assert args.num_clients == 3
    assert args.capture_epochs == [0, 1, 5, 10, 20]
    assert args.client_epochs == 20
