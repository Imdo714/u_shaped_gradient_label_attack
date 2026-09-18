from __future__ import annotations

import torch

from src.shared_pretrained_encoder_drift_attack.pipeline.run_online_fair_z_comparison import (
    SIGNAL_MODES,
    _as_gradient_bundle,
    _as_z_bundle,
    build_parser,
)
from src.shared_pretrained_encoder_drift_attack.pipeline.run_online_z_gradient_attack import (
    ZGradientTranscriptBundle,
)


def _bundle() -> ZGradientTranscriptBundle:
    return ZGradientTranscriptBundle(
        z=torch.randn(3, 4, 8, 8),
        grad_z=torch.randn(3, 4, 8, 8),
        labels=torch.tensor([0, 1, 2]),
        sample_ids=("a", "b", "c"),
        rounds=torch.tensor([1, 5, 20]),
        server_steps=torch.tensor([0, 10, 20]),
        u=torch.randn(3, 8, 4, 4),
    )


def test_signal_projections_reuse_the_same_records() -> None:
    source = _bundle()
    gradient = _as_gradient_bundle(source)
    z_only = _as_z_bundle(source)
    assert gradient.grad_z.data_ptr() == source.grad_z.data_ptr()
    assert gradient.u.data_ptr() == source.u.data_ptr()
    assert z_only.z.data_ptr() == source.z.data_ptr()
    assert gradient.sample_ids == z_only.sample_ids == source.sample_ids
    assert torch.equal(gradient.labels, z_only.labels)


def test_fair_comparison_defaults_to_all_three_signal_views() -> None:
    parser = build_parser()
    args = parser.parse_args(["--pretrained-autoencoder", "checkpoint.pt"])
    assert tuple(args.signal_modes) == SIGNAL_MODES
    assert args.output.endswith("animal5_online_fair_z_comparison")
