from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.client_received_transcript_attack.models.decoder import (
    ClientReceivedDecoder,
    ClientReceivedDecoderConfig,
)
from src.client_received_transcript_attack.models.multiscale_decoder import (
    MultiscaleDecoderConfig,
    MultiscalePixelShuffleDecoder,
)
from src.client_received_transcript_attack.evaluation.v2_comparison import (
    evaluate_decoder_v2_comparison,
)
from src.client_received_transcript_refiner.evaluation.evaluator import evaluate_refiner
from src.client_received_transcript_refiner.models.refiner import (
    ResidualUNetRefiner,
    ResidualUNetRefinerConfig,
    TranscriptConditionedRefinerConfig,
    TranscriptConditionedResidualUNetRefiner,
)
from src.client_received_transcript_refiner.pipeline.common import (
    require_disjoint_datasets,
)
from src.client_received_transcript_refiner.training.trainer import (
    RefinerTrainingConfig,
    train_conditioned_refiner,
    train_refiner,
)


class _TranscriptDataset(Dataset):
    def __init__(self, prefix: str, samples: int = 4) -> None:
        generator = torch.Generator().manual_seed(7)
        self.rows = [
            {"transcript_id": f"{prefix}-{index}"} for index in range(samples)
        ]
        self.items = [
            {
                "transcript_id": row["transcript_id"],
                "server_output_u": torch.randn(5, 4, 4, generator=generator),
                "grad_g_to_f": torch.randn(7, 8, 8, generator=generator),
                "target_image": torch.rand(3, 16, 16, generator=generator),
                "coarse_reconstruction": torch.rand(
                    3, 16, 16, generator=generator
                ),
                "true_label": torch.tensor(index % 2, dtype=torch.long),
            }
            for index, row in enumerate(self.rows)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _decoder(use_label_head: bool = False) -> ClientReceivedDecoder:
    return ClientReceivedDecoder(
        ClientReceivedDecoderConfig(
            u_channels=5,
            grad_z_channels=7,
            num_classes=2,
            image_size=16,
            signal_spatial_size=4,
            signal_channels=8,
            decoder_base_channels=16,
            decoder_min_channels=8,
            refinement_blocks=0,
            use_label_head=use_label_head,
            label_channels=4,
        )
    )


def _refiner() -> ResidualUNetRefiner:
    return ResidualUNetRefiner(
        ResidualUNetRefinerConfig(
            image_channels=3,
            base_channels=8,
            bottleneck_blocks=1,
            max_residual=0.1,
        )
    )


def _conditioned_refiner() -> TranscriptConditionedResidualUNetRefiner:
    return TranscriptConditionedResidualUNetRefiner(
        TranscriptConditionedRefinerConfig(
            u_channels=5,
            grad_z_channels=7,
            image_channels=3,
            base_channels=8,
            condition_channels=4,
            condition_spatial_size=4,
            bottleneck_blocks=1,
            max_residual=0.1,
        )
    )


def test_refiner_starts_as_identity_and_bounds_pixel_corrections():
    model = _refiner()
    coarse = torch.rand(2, 3, 16, 16)
    refined, residual = model(coarse)
    torch.testing.assert_close(refined, coarse)
    assert torch.count_nonzero(residual) == 0

    with torch.no_grad():
        model.residual_head.bias.fill_(100.0)
    _, residual = model(coarse)
    assert float(residual.detach().abs().max()) <= 0.1 + 1e-6


def test_conditioned_refiner_starts_as_identity_and_uses_received_shapes():
    model = _conditioned_refiner()
    coarse = torch.rand(2, 3, 16, 16)
    refined, residual = model(
        coarse,
        torch.randn(2, 5, 4, 4),
        torch.randn(2, 7, 8, 8),
    )
    torch.testing.assert_close(refined, coarse)
    assert torch.count_nonzero(residual) == 0


def test_refiner_training_updates_only_refiner_and_writes_checkpoint(tmp_path):
    decoder = _decoder()
    refiner = _refiner()
    train_dataset = _TranscriptDataset("train")
    validation_dataset = _TranscriptDataset("validation")
    decoder_before = {
        key: value.detach().clone() for key, value in decoder.state_dict().items()
    }
    head_before = refiner.residual_head.weight.detach().clone()

    checkpoint, history = train_refiner(
        refiner,
        decoder,
        train_dataset,
        validation_dataset,
        tmp_path,
        RefinerTrainingConfig(
            epochs=1,
            batch_size=2,
            ssim_weight=0.0,
            edge_weight=0.0,
            perceptual_weight=0.0,
            residual_weight=0.0,
            low_frequency_weight=0.0,
        ),
        torch.device("cpu"),
    )

    assert checkpoint.is_file()
    assert len(history) == 2
    assert history[0]["epoch"] == 0
    assert not torch.equal(head_before, refiner.residual_head.weight.detach())
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(value, decoder_before[key])
    assert not any(parameter.requires_grad for parameter in decoder.parameters())


def test_conditioned_refiner_training_writes_typed_checkpoint(tmp_path):
    refiner = _conditioned_refiner()
    checkpoint, history = train_conditioned_refiner(
        refiner,
        _TranscriptDataset("train"),
        _TranscriptDataset("validation"),
        tmp_path,
        RefinerTrainingConfig(
            epochs=1,
            batch_size=2,
            ssim_weight=0.0,
            edge_weight=0.0,
            perceptual_weight=0.0,
            residual_weight=0.0,
            low_frequency_weight=0.0,
        ),
        torch.device("cpu"),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["refiner_type"] == "transcript_conditioned"
    assert len(history) == 2
    assert history[0]["epoch"] == 0


def test_refiner_evaluation_writes_three_way_outputs_and_deltas(tmp_path):
    dataset = _TranscriptDataset("holdout", samples=2)
    summary = evaluate_refiner(
        _decoder(use_label_head=True),
        _refiner(),
        dataset,
        tmp_path,
        torch.device("cpu"),
        ("cat", "dog"),
        batch_size=2,
        max_grid_images=2,
    )

    assert summary["samples"] == 2
    assert summary["delta_refined_minus_coarse"]["mse"] == 0.0
    assert (tmp_path / "originals/holdout-0.png").is_file()
    assert (tmp_path / "coarse_reconstructions/holdout-0.png").is_file()
    assert (tmp_path / "refined_reconstructions/holdout-0.png").is_file()
    assert (tmp_path / "comparisons/holdout-0.png").is_file()
    assert (tmp_path / "refinement_comparison_grid.png").is_file()
    assert (tmp_path / "refinement_metrics.csv").is_file()
    assert (tmp_path / "refinement_summary.json").is_file()


def test_conditioned_refiner_evaluation_keeps_three_way_output(tmp_path):
    summary = evaluate_refiner(
        _decoder(use_label_head=True),
        _conditioned_refiner(),
        _TranscriptDataset("conditioned-holdout", samples=2),
        tmp_path,
        torch.device("cpu"),
        ("cat", "dog"),
        batch_size=2,
        max_grid_images=2,
    )
    assert summary["refiner_type"] == "transcript_conditioned"
    assert (tmp_path / "refinement_comparison_grid.png").is_file()


def test_decoder_v2_evaluation_writes_four_way_comparison(tmp_path):
    v2 = MultiscalePixelShuffleDecoder(
        MultiscaleDecoderConfig(
            u_channels=5,
            grad_z_channels=7,
            num_classes=2,
            image_size=16,
            signal_spatial_size=4,
            signal_channels=8,
            decoder_base_channels=16,
            decoder_min_channels=8,
            refinement_blocks=0,
            use_label_head=True,
            label_channels=4,
        )
    )
    summary = evaluate_decoder_v2_comparison(
        _decoder(use_label_head=True),
        v2,
        _TranscriptDataset("v2-holdout", samples=2),
        tmp_path,
        torch.device("cpu"),
        ("cat", "dog"),
        batch_size=2,
        max_grid_images=2,
    )
    assert summary["samples"] == 2
    assert set(summary) >= {"baseline", "v2", "postprocessed"}
    assert (tmp_path / "decoder_v2_comparison_grid.png").is_file()
    assert (tmp_path / "baseline_reconstructions/v2-holdout-0.png").is_file()
    assert (tmp_path / "v2_reconstructions/v2-holdout-0.png").is_file()
    assert (tmp_path / "v2_postprocessed/v2-holdout-0.png").is_file()


def test_dataset_overlap_guard_rejects_seen_transcript_ids():
    first = _TranscriptDataset("shared", samples=2)
    second = _TranscriptDataset("shared", samples=1)
    try:
        require_disjoint_datasets(first, second, "test datasets")
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("expected transcript overlap to be rejected")
