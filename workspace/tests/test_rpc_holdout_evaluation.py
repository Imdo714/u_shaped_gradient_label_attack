from __future__ import annotations

from src.client_received_transcript_attack.pipeline.evaluate_rpc_holdout import (
    build_parser,
)


def test_holdout_only_evaluation_defaults_to_100_without_training_arguments():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--server-role-checkpoint",
            "server.pt",
            "--client-role-checkpoint",
            "client.pt",
            "--decoder-checkpoint",
            "decoder.pt",
            "--output",
            "results",
        ]
    )

    assert args.holdout_count == 100
    assert args.max_grid_images == 100
    assert not hasattr(args, "epochs")
