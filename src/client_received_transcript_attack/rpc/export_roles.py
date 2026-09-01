from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch



def _infer_num_classes(state: dict[str, object]) -> int:
    saved = state.get("num_classes")
    if saved is not None:
        return int(saved)
    tail_state = state.get("client_tail")
    if not isinstance(tail_state, dict):
        raise ValueError("source checkpoint has no valid client_tail state")
    for key, value in reversed(list(tail_state.items())):
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[0])
    raise ValueError("cannot infer num_classes from source checkpoint")


def export_role_checkpoints(
    source_checkpoint: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    source = Path(source_checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    metadata = checkpoint.get("metadata", {})
    cut_config = str(state.get("cut_config", "middle"))
    num_classes = _infer_num_classes(state)
    class_names = metadata.get("class_names")
    if class_names is None and isinstance(metadata.get("config"), dict):
        class_names = metadata["config"].get("class_names")

    server_path = output / "server_role.pt"
    client_path = output / "client_role.pt"
    torch.save(
        {
            "role": "server",
            "cut_config": cut_config,
            "server_middle": state["server_middle"],
        },
        server_path,
    )
    torch.save(
        {
            "role": "client",
            "cut_config": cut_config,
            "num_classes": num_classes,
            "class_names": list(class_names) if class_names is not None else None,
            "client_front": state["client_front"],
            "client_tail": state["client_tail"],
        },
        client_path,
    )
    with (output / "role_export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_checkpoint": str(source),
                "server_checkpoint": str(server_path),
                "client_checkpoint": str(client_path),
                "cut_config": cut_config,
                "num_classes": num_classes,
                "server_keys": ["server_middle"],
                "client_keys": ["client_front", "client_tail"],
                "attacker_decoder_receives_checkpoint": False,
            },
            handle,
            indent=2,
        )
    return server_path, client_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export mutually exclusive server and normal-client role checkpoints."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server_path, client_path = export_role_checkpoints(args.checkpoint, args.output)
    print(f"Server role: {server_path.resolve()}")
    print(f"Client role: {client_path.resolve()}")


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "export_role_checkpoints"]
