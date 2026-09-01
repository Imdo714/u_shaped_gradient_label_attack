from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...decoder.data.image_scaling import denormalize_image
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from .protocol import receive_message, send_message


class NormalClientModels:
    """Normal client-owned f/h models; no server g weights are loaded."""

    def __init__(self, front: nn.Module, tail: nn.Module, class_names: tuple[str, ...]) -> None:
        self.front = front
        self.tail = tail
        self.class_names = class_names

    def eval(self) -> "NormalClientModels":
        self.front.eval()
        self.tail.eval()
        return self


def load_client_role(path: str | Path, device: torch.device) -> NormalClientModels:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("role") != "client":
        raise ValueError("expected a client-role checkpoint")
    if "server_middle" in checkpoint:
        raise ValueError("client-role checkpoint contains forbidden server weights")
    cut_config = str(checkpoint["cut_config"])
    num_classes = int(checkpoint["num_classes"])
    raw_names = checkpoint.get("class_names")
    class_names = (
        tuple(str(name) for name in raw_names)
        if raw_names is not None
        else tuple(f"class_{index}" for index in range(num_classes))
    )
    if len(class_names) != num_classes:
        raise ValueError("client-role class names do not match num_classes")
    front = ClientFrontFModel(cut_config)
    tail = ClientTailHModel(num_classes)
    front.load_state_dict(checkpoint["client_front"])
    tail.load_state_dict(checkpoint["client_tail"])
    front.to(device)
    tail.to(device)
    return NormalClientModels(front, tail, class_names).eval()


def _transcript_id(collection_name: str, sample_id: str) -> str:
    digest = hashlib.sha256(
        f"{collection_name}\0{sample_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"transcript_{digest}"


def _sample_ids(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise ValueError("sample ID file must be a CSV containing a sample_id column")
    return {row["sample_id"] for row in rows}


def collect_as_normal_client(
    client_models: NormalClientModels,
    loader,
    proxy_host: str,
    proxy_port: int,
    output_dir: str | Path,
    collection_name: str,
    device: torch.device,
    max_samples: int | None = None,
) -> Path:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    output = Path(output_dir)
    evaluator_dir = output / "evaluator_targets"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, str]] = []

    with socket.create_connection((proxy_host, proxy_port), timeout=30) as connection:
        connection.settimeout(None)
        for images, labels, sample_ids in loader:
            for index, sample_id in enumerate(sample_ids):
                if max_samples is not None and len(rows) >= max_samples:
                    break
                image = images[index : index + 1].to(device)
                label = labels[index : index + 1].to(device)
                request_id = _transcript_id(collection_name, str(sample_id))
                with torch.no_grad():
                    smashed_z = client_models.front(image)
                send_message(
                    connection,
                    "forward",
                    request_id,
                    {"smashed_z": smashed_z.detach().cpu().numpy()},
                )
                forward_result = receive_message(connection)
                if (
                    forward_result.message_type != "forward_result"
                    or forward_result.request_id != request_id
                    or set(forward_result.arrays) != {"server_output_u"}
                ):
                    raise RuntimeError("invalid forward response from server relay")
                server_output_u = torch.from_numpy(
                    forward_result.arrays["server_output_u"]
                ).to(device)
                server_output_u = server_output_u.detach().requires_grad_(True)
                loss = criterion(client_models.tail(server_output_u), label)
                grad_h_to_g = torch.autograd.grad(loss, server_output_u)[0]
                send_message(
                    connection,
                    "backward",
                    request_id,
                    {"grad_h_to_g": grad_h_to_g.detach().cpu().numpy()},
                )
                backward_result = receive_message(connection)
                if (
                    backward_result.message_type != "backward_result"
                    or backward_result.request_id != request_id
                    or set(backward_result.arrays) != {"grad_g_to_f"}
                ):
                    raise RuntimeError("invalid backward response from server relay")

                filename = f"{request_id}.npz"
                np.savez_compressed(
                    evaluator_dir / filename,
                    target_image=denormalize_image(image[0]).cpu().numpy(),
                    true_label=np.int64(label.item()),
                )
                rows.append(
                    {
                        "transcript_id": request_id,
                        "evaluator_target": f"evaluator_targets/{filename}",
                    }
                )
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError("normal client did not process any samples")
    manifest = output / "evaluator_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transcript_id", "evaluator_target"])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "client_collection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "collection_name": collection_name,
                "samples": len(rows),
                "loaded_model_roles": ["client_front", "client_tail"],
                "loaded_server_weights": False,
                "sent_to_server": ["z", "dL/du"],
                "received_from_server": ["u", "dL/dz"],
            },
            handle,
            indent=2,
        )
    print(f"Normal client processed {len(rows)} samples", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a normal f/h client through the authorized research proxy."
    )
    parser.add_argument("--client-role-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sample-id-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()
    device = _device(args.device)
    models = load_client_role(args.client_role_checkpoint, device)
    catalog = ClassCatalog.discover(args.data)
    if models.class_names != catalog.names:
        raise ValueError(
            f"client role classes {models.class_names} do not match dataset {catalog.names}"
        )
    loader = make_loader(
        args.data,
        args.split,
        args.image_size,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        class_names=catalog.names,
        include_sample_ids=_sample_ids(args.sample_id_file),
    )
    collect_as_normal_client(
        models,
        loader,
        args.proxy_host,
        args.proxy_port,
        args.output,
        args.collection_name,
        device,
        args.max_samples,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "NormalClientModels",
    "build_parser",
    "collect_as_normal_client",
    "load_client_role",
]
