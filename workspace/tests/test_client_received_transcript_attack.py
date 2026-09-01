from __future__ import annotations

from pathlib import Path
import socket

import numpy as np
import torch

from src.client_received_transcript_attack.data.collector import (
    ReceivedTranscript,
    collect_received_transcripts,
)
from src.client_received_transcript_attack.data.dataset import (
    ATTACKER_KEYS,
    ClientReceivedTranscriptDataset,
)
from src.client_received_transcript_attack.models.decoder import (
    ClientReceivedDecoder,
    ClientReceivedDecoderConfig,
)
from src.client_received_transcript_attack.pipeline.run_experiment import build_parser
from src.client_received_transcript_attack.pipeline.train_from_transcripts import (
    build_parser as build_transcript_training_parser,
)
from src.client_received_transcript_attack.rpc.export_roles import export_role_checkpoints
from src.client_received_transcript_attack.rpc.protocol import receive_message, send_message
from src.client_received_transcript_attack.training.trainer import (
    AttackTrainingConfig,
    train_client_received_decoder,
)
from src.split_learning.architecture.split_learning_model import build_split_learning_model


class _FakeProvider:
    provider_name = "test-black-box-provider"

    def observe(self, image: torch.Tensor, label: torch.Tensor) -> ReceivedTranscript:
        batch_size = image.shape[0]
        label_scale = label.float().view(batch_size, 1, 1, 1) + 1.0
        return ReceivedTranscript(
            server_output_u=torch.ones(batch_size, 5, 4, 4, device=image.device)
            * label_scale,
            grad_g_to_f=torch.ones(batch_size, 7, 8, 8, device=image.device)
            * label_scale,
        )


def _collect_fake_records(root: Path):
    loader = [
        (
            torch.randn(4, 3, 16, 16),
            torch.tensor([0, 1, 0, 1]),
            ["sample-a", "sample-b", "sample-c", "sample-d"],
        )
    ]
    return collect_received_transcripts(
        _FakeProvider(), loader, root, "unit_test", torch.device("cpu")
    )


def test_collector_persists_only_u_and_grad_z_in_attacker_records(tmp_path):
    manifests = _collect_fake_records(tmp_path)
    with manifests.attacker.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
        first_row = handle.readline().strip().split(",")
    assert header == "transcript_id,attacker_record"
    attacker_record = manifests.attacker.parent / first_row[1]
    with np.load(attacker_record, allow_pickle=False) as record:
        assert set(record.files) == ATTACKER_KEYS
        assert "target_image" not in record.files
        assert "true_label" not in record.files
        assert "smashed_z" not in record.files
        assert "grad_h_to_g" not in record.files


def test_dataset_joins_physically_separated_targets(tmp_path):
    manifests = _collect_fake_records(tmp_path)
    attacker_view = ClientReceivedTranscriptDataset(manifests.attacker)
    paired_view = ClientReceivedTranscriptDataset(manifests.attacker, manifests.evaluator)
    assert set(attacker_view[0]) == {
        "transcript_id",
        "server_output_u",
        "grad_g_to_f",
    }
    assert paired_view[0]["server_output_u"].shape == (5, 4, 4)
    assert paired_view[0]["grad_g_to_f"].shape == (7, 8, 8)
    assert paired_view[0]["target_image"].shape == (3, 16, 16)
    assert paired_view[0]["true_label"].ndim == 0


def test_decoder_uses_only_received_signals_and_backpropagates_to_both_encoders():
    model = ClientReceivedDecoder(
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
            use_label_head=True,
            label_channels=4,
        )
    )
    reconstruction, label_logits = model(
        torch.randn(2, 5, 4, 4), torch.randn(2, 7, 8, 8)
    )
    assert reconstruction.shape == (2, 3, 16, 16)
    assert label_logits is not None and label_logits.shape == (2, 2)
    reconstruction.mean().backward()
    assert any(parameter.grad is not None for parameter in model.u_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.grad_z_encoder.parameters())


def test_public_pair_training_writes_best_decoder_checkpoint(tmp_path):
    manifests = _collect_fake_records(tmp_path / "records")
    dataset = ClientReceivedTranscriptDataset(manifests.attacker, manifests.evaluator)
    model = ClientReceivedDecoder(
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
        )
    )
    before = next(model.u_encoder.parameters()).detach().clone()
    checkpoint, history = train_client_received_decoder(
        model,
        dataset,
        dataset,
        tmp_path / "checkpoints",
        AttackTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            ssim_weight=0.0,
            edge_weight=0.0,
            perceptual_weight=0.0,
        ),
        torch.device("cpu"),
    )
    after = next(model.u_encoder.parameters()).detach()
    assert checkpoint.is_file()
    assert len(history) == 1
    assert not torch.equal(before, after)


def test_pipeline_defaults_to_twenty_holdouts_and_one_hundred_epochs():
    args = build_parser().parse_args([])
    assert args.holdout_count == 20
    assert args.epochs == 100
    assert args.victim_split == "test"


def test_rpc_protocol_round_trips_tensors_without_pickle():
    sender, receiver = socket.socketpair()
    try:
        expected = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        send_message(sender, "forward_result", "request-1", {"server_output_u": expected})
        message = receive_message(receiver)
    finally:
        sender.close()
        receiver.close()
    assert message.message_type == "forward_result"
    assert message.request_id == "request-1"
    np.testing.assert_array_equal(message.arrays["server_output_u"], expected)


def test_role_export_keeps_server_and_client_weights_mutually_exclusive(tmp_path):
    source = tmp_path / "full.pt"
    model = build_split_learning_model("middle", num_classes=3)
    model.save(source, class_names=["cat", "dog", "pug"])
    server_path, client_path = export_role_checkpoints(source, tmp_path / "roles")
    server = torch.load(server_path, map_location="cpu", weights_only=False)
    client = torch.load(client_path, map_location="cpu", weights_only=False)
    assert set(server) == {"role", "cut_config", "server_middle"}
    assert "client_front" not in server and "client_tail" not in server
    assert set(client) == {
        "role",
        "cut_config",
        "num_classes",
        "class_names",
        "client_front",
        "client_tail",
    }
    assert "server_middle" not in client


def test_checkpoint_free_attack_training_cli_has_no_victim_checkpoint_argument():
    parser = build_transcript_training_parser()
    destinations = {action.dest for action in parser._actions}
    assert "checkpoint" not in destinations
    assert "server_role_checkpoint" not in destinations
    assert "client_role_checkpoint" not in destinations
