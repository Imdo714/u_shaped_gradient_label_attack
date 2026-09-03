from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from ...client_received_transcript_attack.pipeline.run_rpc_experiment import (
    _balanced_holdouts,
)
from ...decoder.data.holdout_selection import write_holdout_records
from ...shared.data.class_catalog import ClassCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect auxiliary and multi-client victim transcripts through one "
            "shared ServerMiddle process."
        )
    )
    parser.add_argument("--server-role-checkpoint", required=True)
    parser.add_argument("--aux-client-checkpoint", required=True)
    parser.add_argument(
        "--victim-client",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Repeat to evaluate transfer to multiple f/h client checkpoints.",
    )
    parser.add_argument("--aux-data", default="workspace/data/dataset_aux_10k")
    parser.add_argument("--victim-data", default="workspace/data/dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aux-train-split", default="train")
    parser.add_argument("--aux-validation-split", default="val")
    parser.add_argument("--victim-split", choices=("test", "new_holdout"), default="new_holdout")
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--holdout-start-index", type=int, default=0)
    parser.add_argument("--holdout-labels", nargs="+", default=["cat", "dog"])
    parser.add_argument("--max-aux-train-samples", type=int, default=None)
    parser.add_argument("--max-aux-validation-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--capture-z", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ready-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=86400.0)
    return parser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _start(command: list[str], log: Path) -> tuple[subprocess.Popen, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=_creation_flags(),
    )
    return process, handle


def _wait_ready(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"process exited with code {return_code} before creating {path}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"process did not create ready file within {timeout}s: {path}")


def _run(command: list[str], timeout: float) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=timeout,
        creationflags=_creation_flags(),
    )


def _client_specs(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"victim client must be NAME=CHECKPOINT, found {value!r}")
        name, checkpoint = value.split("=", 1)
        if not name or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in name
        ):
            raise ValueError(f"invalid victim client name: {name!r}")
        result.append((name, Path(checkpoint)))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("victim client names must be unique")
    return result


def _optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def run(args: argparse.Namespace) -> Path:
    victims = _client_specs(args.victim_client)
    aux_catalog = ClassCatalog.discover(args.aux_data, args.aux_train_split)
    victim_catalog = ClassCatalog.discover(args.victim_data, args.victim_split)
    if aux_catalog.names != victim_catalog.names:
        raise ValueError(
            f"auxiliary classes {aux_catalog.names} differ from victim classes "
            f"{victim_catalog.names}"
        )
    labels = tuple(args.holdout_labels) if args.holdout_labels else victim_catalog.names
    holdouts = _balanced_holdouts(
        args.victim_data,
        args.victim_split,
        victim_catalog.names,
        labels,
        args.holdout_count,
        args.holdout_start_index,
    )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"collection output is not empty: {output}")
    runtime = output / "runtime"
    observations = output / "observations"
    runtime.mkdir(parents=True, exist_ok=True)
    observations.mkdir(parents=True, exist_ok=True)
    holdout_file = write_holdout_records(holdouts, output / "holdout_records.csv")

    python = sys.executable
    server_port = _free_port()
    server_ready = runtime / "server.ready.json"
    total_connections = 2 + len(victims)
    server_command = [
        python,
        "-m",
        "src.client_received_transcript_attack.rpc.server",
        "--server-role-checkpoint",
        str(args.server_role_checkpoint),
        "--port",
        str(server_port),
        "--expected-connections",
        str(total_connections),
        "--device",
        str(args.device),
        "--ready-file",
        str(server_ready),
    ]
    server, server_log = _start(server_command, runtime / "server.log")
    active: list[subprocess.Popen] = [server]
    handles: list[object] = [server_log]
    audit_collections: list[dict[str, object]] = []

    def collect(
        *,
        name: str,
        checkpoint: Path,
        data: str,
        split: str,
        destination: Path,
        max_samples: int | None = None,
        sample_ids: Path | None = None,
        expected_samples: int | None = None,
    ) -> None:
        proxy_port = _free_port()
        proxy_ready = runtime / f"{name}.proxy.ready.json"
        proxy_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.rpc.proxy",
            "--listen-port",
            str(proxy_port),
            "--server-port",
            str(server_port),
            "--output",
            str(destination),
            "--ready-file",
            str(proxy_ready),
        ]
        _optional(proxy_command, "--expected-samples", expected_samples)
        if args.capture_z:
            proxy_command.append("--capture-z")
        proxy, proxy_log = _start(proxy_command, runtime / f"{name}.proxy.log")
        active.append(proxy)
        handles.append(proxy_log)
        _wait_ready(proxy_ready, proxy, args.ready_timeout_seconds)
        client_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.rpc.client",
            "--client-role-checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--split",
            split,
            "--collection-name",
            name,
            "--proxy-port",
            str(proxy_port),
            "--output",
            str(destination),
            "--image-size",
            str(args.image_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
        ]
        _optional(client_command, "--max-samples", max_samples)
        _optional(client_command, "--sample-id-file", sample_ids)
        print(f"[shared g] collecting {name}...", flush=True)
        _run(client_command, args.process_timeout_seconds)
        return_code = proxy.wait(timeout=args.process_timeout_seconds)
        if return_code:
            raise RuntimeError(f"proxy {name} exited with code {return_code}")
        active.remove(proxy)
        audit_collections.append(
            {
                "name": name,
                "client_checkpoint": str(checkpoint),
                "data": str(data),
                "split": split,
                "output": str(destination),
                "proxy_command": proxy_command,
                "client_command": client_command,
            }
        )

    try:
        _wait_ready(server_ready, server, args.ready_timeout_seconds)
        aux_root = observations / "auxiliary_client"
        collect(
            name="aux_train",
            checkpoint=Path(args.aux_client_checkpoint),
            data=args.aux_data,
            split=args.aux_train_split,
            destination=aux_root / "train",
            max_samples=args.max_aux_train_samples,
            expected_samples=args.max_aux_train_samples,
        )
        collect(
            name="aux_validation",
            checkpoint=Path(args.aux_client_checkpoint),
            data=args.aux_data,
            split=args.aux_validation_split,
            destination=aux_root / "validation",
            max_samples=args.max_aux_validation_samples,
            expected_samples=args.max_aux_validation_samples,
        )
        for client_name, checkpoint in victims:
            collect(
                name=f"victim_{client_name}",
                checkpoint=checkpoint,
                data=args.victim_data,
                split=args.victim_split,
                destination=observations / "victim_clients" / client_name,
                sample_ids=holdout_file,
                expected_samples=len(holdouts),
            )
        return_code = server.wait(timeout=args.process_timeout_seconds)
        if return_code:
            raise RuntimeError(f"shared server exited with code {return_code}")
        active.remove(server)

        index = {
            "class_names": list(aux_catalog.names),
            "capture_z": bool(args.capture_z),
            "shared_server_process_count": 1,
            "shared_server_connections": total_connections,
            "server_role_checkpoint": str(args.server_role_checkpoint),
            "auxiliary": {
                "train_attacker_manifest": str(aux_root / "train" / "attacker_manifest.csv"),
                "train_target_manifest": str(aux_root / "train" / "evaluator_manifest.csv"),
                "validation_attacker_manifest": str(aux_root / "validation" / "attacker_manifest.csv"),
                "validation_target_manifest": str(aux_root / "validation" / "evaluator_manifest.csv"),
            },
            "victims": {
                name: {
                    "attacker_manifest": str(
                        observations / "victim_clients" / name / "attacker_manifest.csv"
                    ),
                    "evaluator_manifest": str(
                        observations / "victim_clients" / name / "evaluator_manifest.csv"
                    ),
                }
                for name, _ in victims
            },
            "audit": {
                "server_command": server_command,
                "collections": audit_collections,
                "proxy_loaded_checkpoints": False,
                "stored_signals": (
                    ["z", "u", "dL/dz"] if args.capture_z else ["u", "dL/dz"]
                ),
            },
        }
        with (output / "manifest_index.json").open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
        print(f"Shared-server transcript collection: {output.resolve()}")
        return output
    finally:
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for handle in handles:
            handle.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
