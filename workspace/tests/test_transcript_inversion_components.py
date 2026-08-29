from __future__ import annotations

import csv

import torch
from torch import nn

from src.transcript_inversion.data.pairing import PairingAblationDataset, PairingMode
from src.transcript_inversion.data.transcript_dataset import PublicTargetDataset, TranscriptDataset
from src.transcript_inversion.data.transcript_writer import (
    EvaluatorTargetWriter,
    ServerTranscriptWriter,
)
from src.transcript_inversion.losses.gradient_matching import GradientMatchingLoss
from src.transcript_inversion.models.decoder import (
    BidirectionalTranscriptDecoder,
    DecoderConfig,
)
from src.transcript_inversion.models.simulators import FrontSimulator, TailGradientSimulator
from src.transcript_inversion.pipeline.common import load_server_middle_only
from src.transcript_inversion.training.abtr_trainer import ABTRConfig, ABTRTrainer


def _write_combined_manifest(root, sample_ids=("a/unsafe", "b", "c", "d")):
    attacker = ServerTranscriptWriter(root)
    evaluator = EvaluatorTargetWriter(root)
    for index, sample_id in enumerate(sample_ids):
        label = index % 2
        attacker.write(
            sample_id,
            torch.randn(4, 4, 4),
            torch.randn(8, 4, 4),
            torch.randn(8, 4, 4),
            torch.randn(4, 4, 4),
            torch.nn.functional.one_hot(torch.tensor(label), 2).float(),
            label,
            0.9,
        )
        evaluator.write(sample_id, torch.rand(3, 16, 16), label)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "attacker_record", "evaluator_target")
        )
        writer.writeheader()
        for left, right in zip(attacker.rows, evaluator.rows):
            writer.writerow({**left, "evaluator_target": right["evaluator_target"]})
    return manifest


def test_transcript_dataset_keeps_targets_out_of_strict_attacker_view(tmp_path):
    manifest = _write_combined_manifest(tmp_path)
    strict_item = TranscriptDataset(manifest, include_target=False)[0]
    evaluation_item = TranscriptDataset(manifest, include_target=True)[0]

    assert "target_image" not in strict_item
    assert strict_item["has_grad_g_to_f"]
    assert evaluation_item["target_image"].shape == (3, 16, 16)
    assert not (tmp_path / "attacker_records" / "a" / "unsafe.npz").exists()


def test_public_view_never_requires_attacker_records(tmp_path):
    evaluator = EvaluatorTargetWriter(tmp_path)
    evaluator.write("public", torch.rand(3, 8, 8), 1)
    manifest = evaluator.finalize()
    item = PublicTargetDataset(manifest)[0]
    assert set(item) == {"sample_id", "image", "label"}


def test_pairing_controls_generate_expected_pair_types(tmp_path):
    source = TranscriptDataset(_write_combined_manifest(tmp_path), include_target=True)
    exact = PairingAblationDataset(source, PairingMode.EXACT)
    global_shuffled = PairingAblationDataset(source, PairingMode.GLOBAL_SHUFFLED)
    class_shuffled = PairingAblationDataset(source, PairingMode.CLASS_SHUFFLED)

    assert all(bool(exact[index]["pair_is_exact"]) for index in range(len(exact)))
    assert all(
        not bool(global_shuffled[index]["pair_is_exact"])
        for index in range(len(global_shuffled))
    )
    assert all(
        int(class_shuffled[index]["true_label"])
        == int(class_shuffled[index]["paired_true_label"])
        for index in range(len(class_shuffled))
    )


def test_bidirectional_decoder_uses_all_four_transcript_signals():
    model = BidirectionalTranscriptDecoder(
        DecoderConfig(
            z_channels=4,
            u_channels=8,
            num_classes=2,
            image_size=16,
            signal_spatial_size=4,
            decoder_base_channels=32,
            decoder_min_channels=8,
        )
    )
    output = model(
        torch.randn(2, 4, 4, 4),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 4, 4, 4),
        torch.eye(2),
    )
    assert output.shape == (2, 3, 16, 16)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_gradient_matching_backpropagates_through_tail_simulator():
    tail = TailGradientSimulator(8, 2, width=8)
    server_output = torch.randn(2, 8, 4, 4, requires_grad=True)
    objective = nn.CrossEntropyLoss(reduction="sum")(
        tail(server_output), torch.tensor([0, 1])
    )
    predicted = torch.autograd.grad(objective, server_output, create_graph=True)[0]
    loss, metrics = GradientMatchingLoss()(predicted, torch.randn_like(predicted))
    loss.backward()
    assert metrics["gradient_loss"] >= 0.0
    assert any(parameter.grad is not None for parameter in tail.parameters())


def test_server_loader_does_not_require_client_checkpoint_states(tmp_path):
    from src.split_learning.g_model.server_middle_g_model import ServerMiddleGModel

    server = ServerMiddleGModel("late")
    checkpoint = tmp_path / "server.pt"
    torch.save(
        {"model": {"cut_config": "late", "server_middle": server.state_dict()}}, checkpoint
    )
    restored, cut_config = load_server_middle_only(checkpoint, torch.device("cpu"))
    assert cut_config == "late"
    assert isinstance(restored, ServerMiddleGModel)


def test_strict_unpaired_abtr_runs_without_real_targets(tmp_path):
    real_batch = {
        "smashed_z": torch.randn(2, 4, 4, 4),
        "server_output_u": torch.randn(2, 8, 4, 4),
        "grad_h_to_g": torch.randn(2, 8, 4, 4),
        "grad_g_to_f": torch.randn(2, 4, 4, 4),
        "has_grad_g_to_f": torch.ones(2, dtype=torch.bool),
        "label_condition": torch.eye(2),
    }
    public_batch = {
        "image": torch.rand(2, 3, 16, 16),
        "label": torch.tensor([0, 1]),
    }
    front = FrontSimulator(4, (4, 4), width=8)
    server = nn.Conv2d(4, 8, 1)
    tail = TailGradientSimulator(8, 2, width=8)
    decoder = BidirectionalTranscriptDecoder(
        DecoderConfig(
            z_channels=4,
            u_channels=8,
            num_classes=2,
            image_size=16,
            signal_spatial_size=4,
            signal_channels=8,
            label_channels=4,
            decoder_base_channels=16,
            decoder_min_channels=4,
        )
    )
    trainer = ABTRTrainer(ABTRConfig(epochs=1), torch.device("cpu"), 2)
    history = trainer.fit(
        front,
        server,
        tail,
        decoder,
        [real_batch],
        [public_batch],
        tmp_path / "abtr",
    )
    assert len(history) == 1
    assert (tmp_path / "abtr" / "abtr_models.pt").is_file()
