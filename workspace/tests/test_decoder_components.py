from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.decoder.data.gradient_label_predictor import GradientLabelPredictor
from src.decoder.evaluation.image_comparison_writer import ReconstructionComparisonWriter
from src.decoder.losses.reconstruction_loss import structural_similarity
from src.decoder.models.label_conditioned_decoder import DecoderConfig, LabelConditionedDecoder


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


def test_comparison_writer_saves_only_comparisons_and_grid(tmp_path):
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

    assert not (tmp_path / "originals").exists()
    assert not (tmp_path / "reconstructions").exists()
    comparison_path = tmp_path / "comparisons/sample_one.png"
    assert comparison_path.is_file()
    assert grid_path == tmp_path / "comparison_grid.png"
    assert grid_path.is_file()
    with Image.open(comparison_path) as comparison:
        assert comparison.size == (16, 42)
