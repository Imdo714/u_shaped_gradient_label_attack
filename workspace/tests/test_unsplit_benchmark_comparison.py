from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.shared_pretrained_encoder_drift_attack.pipeline import run_unsplit_comparison
from src.shared_pretrained_encoder_drift_attack.pipeline.run_unsplit_comparison import build_parser
from src.shared_pretrained_encoder_drift_attack.unsplit_benchmark import (
    BenchmarkReconstructionDecoder,
    LayerwiseBenchmarkNet,
    first_example_per_class_indices,
    observe_transcript,
    run_unsplit_attack,
)


class _LabelDataset(Dataset):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.zeros(1, 2, 2), self.labels[index]


class _TinyMnistDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        return torch.rand(1, 28, 28, generator=generator), index % 10


def test_official_demo_target_rule_selects_first_example_of_each_class():
    dataset = _LabelDataset([2, 0, 2, 1, 0, 1])
    assert first_example_per_class_indices(dataset, 3) == [1, 3, 0]


def test_mnist_layerwise_model_exposes_matching_cut_gradient():
    model = LayerwiseBenchmarkNet("mnist")
    images = torch.rand(2, 1, 28, 28)
    labels = torch.tensor([0, 1])
    z, gradient = observe_transcript(model, images, labels, split_depth=2)
    assert z.shape == (2, 8, 12, 12)
    assert gradient.shape == z.shape
    assert model.forward_from(z, split_depth=2).shape == (2, 10)


def test_cifar_layerwise_model_matches_public_split_shape():
    model = LayerwiseBenchmarkNet("cifar10")
    images = torch.rand(2, 3, 32, 32)
    z = model.forward_to(images, split_depth=4)
    assert z.shape == (2, 64, 16, 16)
    assert model.forward_from(z, split_depth=4).shape == (2, 10)


def test_native_resolution_decoders_accept_all_transcript_conditions():
    z = torch.randn(2, 8, 12, 12)
    gradient = torch.randn_like(z)
    for condition in ("z_only", "gradient_only", "z_gradient"):
        decoder = BenchmarkReconstructionDecoder(8, 1, 28, condition)
        reconstruction = decoder(z, gradient)
        assert reconstruction.shape == (2, 1, 28, 28)
        assert torch.isfinite(reconstruction).all()


def test_unsplit_attack_smoke_returns_input_shaped_finite_result():
    victim = LayerwiseBenchmarkNet("mnist")
    clone = LayerwiseBenchmarkNet("mnist")
    target = torch.rand(1, 1, 28, 28)
    target_z = victim.forward_to(target, split_depth=1).detach()
    result = run_unsplit_attack(
        clone,
        1,
        target_z,
        target.shape,
        main_iters=1,
        input_iters=1,
        model_iters=1,
    )
    assert result.reconstruction.shape == target.shape
    assert torch.isfinite(result.reconstruction).all()
    assert result.runtime_seconds >= 0


def test_comparison_parser_uses_public_unsplit_iteration_defaults():
    args = build_parser().parse_args(["--dataset", "mnist"])
    assert args.unsplit_main_iters == 1000
    assert args.unsplit_input_iters == 100
    assert args.unsplit_model_iters == 100
    assert args.max_targets == 10


def test_comparison_pipeline_writes_shared_target_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_unsplit_comparison,
        "load_benchmark_datasets",
        lambda *args, **kwargs: (_TinyMnistDataset(40), _TinyMnistDataset(10)),
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            "mnist",
            "--output",
            str(tmp_path),
            "--pretrain-samples",
            "10",
            "--aux-samples",
            "10",
            "--victim-samples",
            "10",
            "--pretrain-epochs",
            "1",
            "--warmup-epochs",
            "1",
            "--victim-epochs",
            "1",
            "--attack-epochs",
            "1",
            "--batch-size",
            "5",
            "--max-targets",
            "1",
            "--unsplit-main-iters",
            "1",
            "--unsplit-input-iters",
            "1",
            "--unsplit-model-iters",
            "1",
            "--device",
            "cpu",
            "--no-download",
        ]
    )
    output = run_unsplit_comparison.run(args)
    assert (output / "target_manifest.csv").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "paper_reference.csv").is_file()
    assert (output / "victim_and_drift_metrics.csv").is_file()
    assert (output / "final" / "comparison_grid.png").is_file()
