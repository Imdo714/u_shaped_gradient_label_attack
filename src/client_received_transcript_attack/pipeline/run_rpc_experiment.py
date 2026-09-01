from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ...decoder.data.holdout_selection import (
    HoldoutRecord,
    select_holdout_records,
    write_holdout_records,
)
from ...shared.data.class_catalog import ClassCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run process-separated U-shaped SL traffic, passive u/dL-dz capture, "
            "checkpoint-free attack training, and isolated evaluation."
        )
    )
    parser.add_argument("--server-role-checkpoint", required=True)
    parser.add_argument("--client-role-checkpoint", required=True)
    parser.add_argument("--data", default="workspace/data/dataset")
    parser.add_argument("--output", default=None)
    parser.add_argument("--aux-train-split", default="train")
    parser.add_argument("--aux-validation-split", default="val")
    parser.add_argument("--victim-split", choices=("test", "new_holdout"), default="test")
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--holdout-start-index", type=int, default=0)
    parser.add_argument("--holdout-labels", nargs="+", default=None)
    parser.add_argument("--max-aux-train-samples", type=int, default=None)
    parser.add_argument("--max-aux-validation-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--use-label-head", action="store_true")
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--refinement-blocks", type=int, default=1)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--perceptual-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ready-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=14400.0)
    return parser


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def _wait_ready(path: Path, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"background process exited with code {return_code} before becoming ready"
            )
        time.sleep(0.1)
    raise TimeoutError(f"process did not create ready file within {timeout}s: {path}")


def _balanced_holdouts(
    data_dir: str | Path,
    split: str,
    class_names: tuple[str, ...],
    labels: tuple[str, ...],
    count: int,
    start_index: int,
) -> list[HoldoutRecord]:
    if count < 1:
        raise ValueError("holdout_count must be positive")
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        raise ValueError(f"unknown holdout labels: {unknown}")
    base_count, remainder = divmod(count, len(labels))
    records: list[HoldoutRecord] = []
    for index, label in enumerate(labels):
        class_count = base_count + int(index < remainder)
        if class_count:
            records.extend(
                select_holdout_records(
                    data_dir,
                    split,
                    class_names,
                    class_count,
                    labels=(label,),
                    start_index=start_index,
                )
            )
    return records


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _start_logged(command: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=_creation_flags(),
    )
    return process, handle


def _run_checked(command: list[str], timeout: float) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=timeout,
        creationflags=_creation_flags(),
    )


def _optional_argument(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def run(args: argparse.Namespace) -> Path:
    if args.aux_train_split == args.aux_validation_split:
        raise ValueError("auxiliary train and validation splits must be different")
    if args.victim_split in (args.aux_train_split, args.aux_validation_split):
        raise ValueError("victim split must differ from auxiliary splits")
    catalog = ClassCatalog.discover(args.data)
    selected_labels = tuple(args.holdout_labels) if args.holdout_labels else catalog.names
    holdouts = _balanced_holdouts(
        args.data,
        args.victim_split,
        catalog.names,
        selected_labels,
        args.holdout_count,
        args.holdout_start_index,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output)
        if args.output
        else Path("workspace/results/client_received_transcript_attack") / f"rpc_{stamp}"
    )
    runtime = output / "runtime"
    observations = output / "observations"
    runtime.mkdir(parents=True, exist_ok=True)
    observations.mkdir(parents=True, exist_ok=True)
    holdout_path = write_holdout_records(holdouts, output / "holdout_records.csv")
    python = sys.executable
    server_port = _free_loopback_port()
    server_ready = runtime / "server.ready.json"
    server_ready.unlink(missing_ok=True)
    server_command = [
        python,
        "-m",
        "src.client_received_transcript_attack.rpc.server",
        "--server-role-checkpoint",
        str(args.server_role_checkpoint),
        "--port",
        str(server_port),
        "--expected-connections",
        "3",
        "--device",
        str(args.device),
        "--ready-file",
        str(server_ready),
    ]
    server_process, server_log = _start_logged(server_command, runtime / "server.log")
    active_processes: list[subprocess.Popen] = [server_process]
    process_handles: list[object] = [server_log]
    client_commands: list[list[str]] = []
    proxy_commands: list[list[str]] = []

    try:
        _wait_ready(server_ready, server_process, args.ready_timeout_seconds)
        collections = (
            (
                "aux_train",
                args.aux_train_split,
                args.max_aux_train_samples,
                None,
            ),
            (
                "aux_validation",
                args.aux_validation_split,
                args.max_aux_validation_samples,
                None,
            ),
            ("victim_holdout", args.victim_split, None, holdout_path),
        )
        for collection_name, split, max_samples, sample_id_file in collections:
            print(f"[RPC] Collecting {collection_name} through passive proxy...", flush=True)
            collection_output = observations / collection_name
            proxy_port = _free_loopback_port()
            proxy_ready = runtime / f"{collection_name}_proxy.ready.json"
            proxy_ready.unlink(missing_ok=True)
            proxy_command = [
                python,
                "-m",
                "src.client_received_transcript_attack.rpc.proxy",
                "--listen-port",
                str(proxy_port),
                "--server-port",
                str(server_port),
                "--output",
                str(collection_output),
                "--ready-file",
                str(proxy_ready),
            ]
            expected_samples = len(holdouts) if collection_name == "victim_holdout" else max_samples
            _optional_argument(proxy_command, "--expected-samples", expected_samples)
            proxy_commands.append(proxy_command)
            proxy_process, proxy_log = _start_logged(
                proxy_command, runtime / f"{collection_name}_proxy.log"
            )
            active_processes.append(proxy_process)
            process_handles.append(proxy_log)
            _wait_ready(proxy_ready, proxy_process, args.ready_timeout_seconds)

            client_command = [
                python,
                "-m",
                "src.client_received_transcript_attack.rpc.client",
                "--client-role-checkpoint",
                str(args.client_role_checkpoint),
                "--data",
                str(args.data),
                "--split",
                str(split),
                "--collection-name",
                collection_name,
                "--proxy-port",
                str(proxy_port),
                "--output",
                str(collection_output),
                "--image-size",
                str(args.image_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                str(args.device),
            ]
            _optional_argument(client_command, "--max-samples", max_samples)
            _optional_argument(client_command, "--sample-id-file", sample_id_file)
            client_commands.append(client_command)
            _run_checked(client_command, args.process_timeout_seconds)
            proxy_return = proxy_process.wait(timeout=args.process_timeout_seconds)
            if proxy_return:
                raise RuntimeError(
                    f"proxy for {collection_name} exited with code {proxy_return}"
                )
            active_processes.remove(proxy_process)

        server_return = server_process.wait(timeout=args.process_timeout_seconds)
        if server_return:
            raise RuntimeError(f"server exited with code {server_return}")
        active_processes.remove(server_process)

        training_output = output / "attack_training"
        trainer_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.pipeline.train_from_transcripts",
            "--train-attacker-manifest",
            str(observations / "aux_train" / "attacker_manifest.csv"),
            "--train-target-manifest",
            str(observations / "aux_train" / "evaluator_manifest.csv"),
            "--validation-attacker-manifest",
            str(observations / "aux_validation" / "attacker_manifest.csv"),
            "--validation-target-manifest",
            str(observations / "aux_validation" / "evaluator_manifest.csv"),
            "--class-names",
            *catalog.names,
            "--output",
            str(training_output),
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--classification-weight",
            str(args.classification_weight),
            "--signal-spatial-size",
            str(args.signal_spatial_size),
            "--signal-channels",
            str(args.signal_channels),
            "--decoder-base-channels",
            str(args.decoder_base_channels),
            "--decoder-min-channels",
            str(args.decoder_min_channels),
            "--refinement-blocks",
            str(args.refinement_blocks),
            "--l1-weight",
            str(args.l1_weight),
            "--ssim-weight",
            str(args.ssim_weight),
            "--edge-weight",
            str(args.edge_weight),
            "--perceptual-weight",
            str(args.perceptual_weight),
            "--gradient-clip-norm",
            str(args.gradient_clip_norm),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
        ]
        if args.use_label_head:
            trainer_command.append("--use-label-head")
        forbidden_role_paths = {
            str(Path(args.server_role_checkpoint)),
            str(Path(args.client_role_checkpoint)),
        }
        if any(value in trainer_command for value in forbidden_role_paths):
            raise RuntimeError("attack trainer command unexpectedly contains a role checkpoint")
        print("[RPC] Training decoder in a checkpoint-free attacker process...", flush=True)
        _run_checked(trainer_command, args.process_timeout_seconds)

        decoder_checkpoint = (
            training_output / "checkpoints" / "client_received_decoder_best.pt"
        )
        evaluator_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.pipeline.evaluate_from_transcripts",
            "--decoder-checkpoint",
            str(decoder_checkpoint),
            "--attacker-manifest",
            str(observations / "victim_holdout" / "attacker_manifest.csv"),
            "--evaluator-manifest",
            str(observations / "victim_holdout" / "evaluator_manifest.csv"),
            "--class-names",
            *catalog.names,
            "--output",
            str(output / "evaluation"),
            "--batch-size",
            str(args.batch_size),
            "--max-grid-images",
            str(args.max_grid_images),
            "--device",
            str(args.device),
        ]
        print("[RPC] Evaluating in a separate evaluator process...", flush=True)
        _run_checked(evaluator_command, args.process_timeout_seconds)

        with (output / "process_boundary_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "server_process_loaded": ["server_middle"],
                    "normal_client_process_loaded": ["client_front", "client_tail"],
                    "proxy_loaded_victim_checkpoint": False,
                    "proxy_persisted_signals": ["u", "dL/dz"],
                    "attack_trainer_loaded_victim_checkpoint": False,
                    "attack_trainer_received_victim_targets": False,
                    "evaluator_loaded_victim_checkpoint": False,
                    "transport": "loopback TCP, protocol-aware authorized passive relay",
                    "server_command": server_command,
                    "proxy_commands": proxy_commands,
                    "client_commands": client_commands,
                    "attack_trainer_command": trainer_command,
                    "evaluator_command": evaluator_command,
                },
                handle,
                indent=2,
            )
        with (output / "rpc_run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **vars(args),
                    "resolved_output": str(output),
                    "class_names": list(catalog.names),
                    "holdout_samples": len(holdouts),
                    "decoder_checkpoint": str(decoder_checkpoint),
                },
                handle,
                indent=2,
                default=str,
            )
        print(f"RPC experiment results: {output.resolve()}")
        return output
    finally:
        for process in active_processes:
            if process.poll() is None:
                process.terminate()
        for process in active_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for handle in process_handles:
            handle.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
