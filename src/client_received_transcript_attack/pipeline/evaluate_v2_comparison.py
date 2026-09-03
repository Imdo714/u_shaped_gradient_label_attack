from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..data.dataset import ClientReceivedTranscriptDataset
from ..evaluation.v2_comparison import evaluate_decoder_v2_comparison
from ..models.factory import load_decoder_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original / baseline decoder / multiscale decoder v2 / "
            "deterministic visual enhancement on the same unseen holdout."
        )
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--attacker-manifest", required=True)
    parser.add_argument("--evaluator-manifest", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--chroma-denoise", type=float, default=0.15)
    parser.add_argument("--sharpen-amount", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=1.03)
    parser.add_argument("--saturation", type=float, default=1.03)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    device = _device(args.device)
    baseline, baseline_state = load_decoder_checkpoint(
        args.baseline_checkpoint, device
    )
    v2, v2_state = load_decoder_checkpoint(args.v2_checkpoint, device)
    baseline_type = baseline_state.get("decoder_type", "baseline_bilinear")
    v2_type = v2_state.get("decoder_type", "baseline_bilinear")
    if baseline_type != "baseline_bilinear":
        raise ValueError("--baseline-checkpoint must contain the bilinear decoder")
    if v2_type != "multiscale_pixelshuffle":
        raise ValueError("--v2-checkpoint must contain the multiscale PixelShuffle decoder")
    if baseline.config.num_classes != len(args.class_names):
        raise ValueError("baseline class count does not match --class-names")
    if v2.config.num_classes != len(args.class_names):
        raise ValueError("v2 class count does not match --class-names")
    if baseline.config.image_size != v2.config.image_size:
        raise ValueError("baseline and v2 decoder image sizes differ")

    dataset = ClientReceivedTranscriptDataset(
        args.attacker_manifest, args.evaluator_manifest
    )
    summary = evaluate_decoder_v2_comparison(
        baseline,
        v2,
        dataset,
        args.output,
        device,
        tuple(args.class_names),
        batch_size=args.batch_size,
        max_grid_images=args.max_grid_images,
        chroma_denoise=args.chroma_denoise,
        sharpen_amount=args.sharpen_amount,
        contrast=args.contrast,
        saturation=args.saturation,
    )
    output = Path(args.output)
    with (output / "evaluation_run_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "baseline_decoder_type": baseline_type,
                "v2_decoder_type": v2_type,
                "baseline_sha256": _sha256(args.baseline_checkpoint),
                "v2_sha256": _sha256(args.v2_checkpoint),
                "loaded_split_learning_checkpoint": False,
                "postprocessing_is_deterministic": True,
                "postprocessing_uses_generative_prior": False,
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
