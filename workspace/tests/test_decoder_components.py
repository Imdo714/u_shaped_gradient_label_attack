from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.decoder.data.gradient_label_predictor import GradientLabelPredictor
from src.decoder.evaluation.image_comparison_writer import ReconstructionComparisonWriter
from src.decoder.losses.reconstruction_loss import ReconstructionLoss, structural_similarity
from src.decoder.models.label_conditioned_decoder import DecoderConfig, LabelConditionedDecoder
from src.decoder.pipeline.run_reconstruction_experiment import (
    build_parser,
    resolved_decoder_options,
)


def test_label_conditioned_decoder_combines_all_observations():
    config = DecoderConfig(
        z_channels=32,
        u_channels=64,
        gradient_channels=64,
        num_classes=3,
        image_size=64,
    )
    model = LabelConditionedDecoder(config)
    output = model(
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 64, 8, 8),
        torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.7, 0.2]]),
    )
    assert output.shape == (2, 3, 64, 64)
    assert torch.all((0.0 <= output) & (output <= 1.0))
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gradient_predictor_supports_hard_and_soft_conditions():
    centroids = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    predictor = GradientLabelPredictor(centroids, {0: 2, 1: 0, 2: 1}, 3)
    hard = predictor.predict(torch.tensor([0.9, 0.1]), soft=False)
    soft = predictor.predict(torch.tensor([0.9, 0.1]), soft=True)
    assert hard.label == 2
    assert hard.probabilities.tolist() == [0.0, 0.0, 1.0]
    assert soft.probabilities.sum().item() == pytest.approx(1.0)
    assert soft.probabilities[2] > soft.probabilities[0]


def test_gradient_predictor_rejects_incomplete_cluster_mapping():
    with pytest.raises(ValueError, match="must cover"):
        GradientLabelPredictor(torch.eye(3), {0: 0, 1: 1}, 3)


def test_structural_similarity_is_one_for_identical_images():
    image = torch.rand(2, 3, 16, 16)
    assert structural_similarity(image, image).item() == pytest.approx(1.0, abs=1e-5)


def test_comparison_writer_saves_comparisons_grid_and_separate_images(tmp_path):
    writer = ReconstructionComparisonWriter(
        tmp_path, ("cat", "dog", "pug"), max_grid_images=2
    )
    writer.save(
        "sample_one",
        torch.zeros(3, 8, 8),
        torch.ones(3, 8, 8),
        true_label=0,
        inferred_label=1,
    )
    grid_path = writer.finalize()

    assert (tmp_path / "originals/sample_one.png").is_file()
    assert (tmp_path / "reconstructions/sample_one.png").is_file()
    comparison_path = tmp_path / "comparisons/sample_one.png"
    assert comparison_path.is_file()
    assert grid_path == tmp_path / "comparison_grid.png"
    assert grid_path.is_file()
    with Image.open(comparison_path) as comparison:
        assert comparison.size == (16, 42)


def test_strong_decoder_configuration_forwards():
    config = DecoderConfig(
        z_channels=32,
        u_channels=64,
        gradient_channels=64,
        num_classes=3,
        image_size=64,
        signal_spatial_size=16,
        signal_channels=64,
        label_channels=32,
        decoder_base_channels=256,
        decoder_min_channels=32,
        decoder_refinement_blocks=1,
    )
    model = LabelConditionedDecoder(config)
    output = model(
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 64, 8, 8),
        torch.tensor([[0.2, 0.7, 0.1]]),
    )
    assert output.shape == (1, 3, 64, 64)


def test_edge_and_perceptual_losses_are_reported():
    reconstruction = torch.rand(2, 3, 16, 16, requires_grad=True)
    target = torch.rand(2, 3, 16, 16)
    loss_function = ReconstructionLoss(
        l1_weight=1.0,
        ssim_weight=0.75,
        edge_weight=0.15,
        perceptual_weight=0.25,
    )
    loss, metrics = loss_function(reconstruction, target)
    assert set(metrics) == {"loss", "l1", "ssim", "edge", "perceptual"}
    assert metrics["edge"] >= 0.0
    assert metrics["perceptual"] >= 0.0
    loss.backward()
    assert reconstruction.grad is not None


def test_decoder_presets_and_cli_override_resolve_expected_values():
    parser = build_parser()
    baseline = resolved_decoder_options(parser.parse_args([]))
    strong = resolved_decoder_options(
        parser.parse_args(
            ["--decoder-preset", "strong", "--edge-weight", "0.2"]
        )
    )
    assert baseline["signal_spatial_size"] == 8
    assert baseline["edge_weight"] == 0.0
    assert strong["signal_spatial_size"] == 16
    assert strong["decoder_base_channels"] == 256
    assert strong["edge_weight"] == 0.2
