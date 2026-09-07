from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from ...decoder.data.image_scaling import denormalize_image
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ..models import ReconstructionDecoder, ReconstructionDecoderConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Centrally pretrain the common client encoder as an autoencoder."
    )
    parser.add_argument("--data", default="workspace/data/dataset")
    parser.add_argument(
        "--output",
        default="workspace/results/shared_pretrained_encoder_drift_attack/autoencoder",
    )
    parser.add_argument("--cut-config", choices=("early", "middle", "late"), default="middle")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument("--min-channels", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value: str) -> torch.device:
    return torch.device(
        "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
    )


def _epoch(encoder, decoder, loader, optimizer, device) -> float:
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    total = 0.0
    samples = 0
    for images, _, _ in loader:
        images = images.to(device)
        target = denormalize_image(images)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction = decoder(encoder(images))
            l1 = nn.functional.l1_loss(reconstruction, target)
            mse = nn.functional.mse_loss(reconstruction, target)
            loss = l1 + 0.5 * mse
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * images.shape[0]
        samples += images.shape[0]
    return total / samples


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = _device(args.device)
    catalog = ClassCatalog.discover(args.data)
    train_loader = make_loader(
        args.data, "train", args.image_size, args.batch_size, args.num_workers,
        shuffle=True, augment=True, class_names=catalog.names,
    )
    val_loader = make_loader(
        args.data, "val", args.image_size, args.batch_size, args.num_workers,
        shuffle=False, class_names=catalog.names,
    )
    encoder = ClientFrontFModel(args.cut_config).to(device)
    with torch.no_grad():
        sample = next(iter(train_loader))[0][:1].to(device)
        z_channels = int(encoder(sample).shape[1])
    decoder_config = ReconstructionDecoderConfig(
        z_channels=z_channels,
        image_size=args.image_size,
        use_z=True,
        signal_spatial_size=args.signal_spatial_size,
        signal_channels=args.signal_channels,
        base_channels=args.base_channels,
        min_channels=args.min_channels,
    )
    decoder = ReconstructionDecoder(decoder_config).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()], lr=args.learning_rate
    )
    best_loss = float("inf")
    best_epoch = 0
    best_encoder = copy.deepcopy(encoder.state_dict())
    best_decoder = copy.deepcopy(decoder.state_dict())
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(encoder, decoder, train_loader, optimizer, device)
        val_loss = _epoch(encoder, decoder, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"AE epoch {epoch:03d}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "pretrained_autoencoder_best.pt"
    torch.save(
        {
            "encoder": best_encoder,
            "autoencoder_decoder": best_decoder,
            "decoder_config": decoder_config.to_dict(),
            "cut_config": args.cut_config,
            "image_size": args.image_size,
            "class_names": list(catalog.names),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "note": "autoencoder decoder is not the attack decoder",
        }, checkpoint,
    )
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(
        json.dumps({**vars(args), "resolved_device": str(device), "checkpoint": str(checkpoint)}, indent=2),
        encoding="utf-8",
    )
    print(f"Pretrained encoder checkpoint: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
