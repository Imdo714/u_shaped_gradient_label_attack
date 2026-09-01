from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data.dataset import ClientReceivedTranscriptDataset
from ..evaluation.evaluator import evaluate_client_received_decoder
from ..models.decoder import ClientReceivedDecoder, ClientReceivedDecoderConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained attack decoder in an evaluator-only process."
    )
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--attacker-manifest", required=True)
    parser.add_argument("--evaluator-manifest", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(args: argparse.Namespace) -> dict[str, float | int | bool]:
    device = _device(args.device)
    checkpoint = torch.load(args.decoder_checkpoint, map_location=device, weights_only=False)
    config = ClientReceivedDecoderConfig(**checkpoint["decoder_config"])
    if config.num_classes != len(args.class_names):
        raise ValueError("decoder class count does not match --class-names")
    decoder = ClientReceivedDecoder(config).to(device)
    decoder.load_state_dict(checkpoint["model"])
    dataset = ClientReceivedTranscriptDataset(
        args.attacker_manifest, args.evaluator_manifest
    )
    summary = evaluate_client_received_decoder(
        decoder,
        dataset,
        args.output,
        device,
        tuple(args.class_names),
        batch_size=args.batch_size,
        max_grid_images=args.max_grid_images,
    )
    with (Path(args.output) / "evaluation_run_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "loaded_victim_checkpoint": False,
                "evaluation_summary": summary,
            },
            handle,
            indent=2,
            default=str,
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
