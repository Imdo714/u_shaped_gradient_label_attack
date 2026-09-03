from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ...decoder.data.holdout_selection import write_holdout_records
from ...shared.data.class_catalog import ClassCatalog
from .run_rpc_experiment import (
    _balanced_holdouts,
    _free_loopback_port,
    _run_checked,
    _start_logged,
    _wait_ready,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one isolated victim holdout through the authorized passive RPC "
            "relay and evaluate an existing attack decoder without retraining it."
        )
    )
    parser.add_argument("--server-role-checkpoint", required=True)
    parser.add_argument("--client-role-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--data", default="workspace/data/dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--victim-split", choices=("test", "new_holdout"), default="new_holdout"
    )
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--holdout-start-index", type=int, default=0)
    parser.add_argument("--holdout-labels", nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--capture-z", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grid-images", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ready-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=14400.0)
    return parser


def run(args: argparse.Namespace) -> Path:
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
    output = Path(args.output)
    if (output / "observations" / "victim_holdout" / "attacker_manifest.csv").exists():
        raise FileExistsError(
            f"holdout observation output already exists: {output}; choose a new --output"
        )
    runtime = output / "runtime"
    observations = output / "observations" / "victim_holdout"
    runtime.mkdir(parents=True, exist_ok=True)
    observations.mkdir(parents=True, exist_ok=True)
    holdout_path = write_holdout_records(holdouts, output / "holdout_records.csv")

    python = sys.executable
    server_port = _free_loopback_port()
    server_ready = runtime / "server.ready.json"
    server_command = [
        python,
        "-m",
        "src.client_received_transcript_attack.rpc.server",
        "--server-role-checkpoint",
        str(args.server_role_checkpoint),
        "--port",
        str(server_port),
        "--expected-connections",
        "1",
        "--device",
        str(args.device),
        "--ready-file",
        str(server_ready),
    ]
    server_process, server_log = _start_logged(server_command, runtime / "server.log")
    active_processes: list[subprocess.Popen] = [server_process]
    process_handles: list[object] = [server_log]

    try:
        _wait_ready(server_ready, server_process, args.ready_timeout_seconds)
        proxy_port = _free_loopback_port()
        proxy_ready = runtime / "victim_holdout_proxy.ready.json"
        proxy_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.rpc.proxy",
            "--listen-port",
            str(proxy_port),
            "--server-port",
            str(server_port),
            "--output",
            str(observations),
            "--expected-samples",
            str(len(holdouts)),
            "--ready-file",
            str(proxy_ready),
        ]
        if args.capture_z:
            proxy_command.append("--capture-z")
        proxy_process, proxy_log = _start_logged(
            proxy_command, runtime / "victim_holdout_proxy.log"
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
            str(args.victim_split),
            "--collection-name",
            "victim_holdout",
            "--proxy-port",
            str(proxy_port),
            "--output",
            str(observations),
            "--image-size",
            str(args.image_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--sample-id-file",
            str(holdout_path),
        ]
        print(f"[RPC] Collecting {len(holdouts)} holdout samples...", flush=True)
        _run_checked(client_command, args.process_timeout_seconds)

        proxy_return = proxy_process.wait(timeout=args.process_timeout_seconds)
        if proxy_return:
            raise RuntimeError(f"holdout proxy exited with code {proxy_return}")
        active_processes.remove(proxy_process)
        server_return = server_process.wait(timeout=args.process_timeout_seconds)
        if server_return:
            raise RuntimeError(f"server exited with code {server_return}")
        active_processes.remove(server_process)

        evaluation_output = output / "evaluation"
        evaluator_command = [
            python,
            "-m",
            "src.client_received_transcript_attack.pipeline.evaluate_from_transcripts",
            "--decoder-checkpoint",
            str(args.decoder_checkpoint),
            "--attacker-manifest",
            str(observations / "attacker_manifest.csv"),
            "--evaluator-manifest",
            str(observations / "evaluator_manifest.csv"),
            "--class-names",
            *catalog.names,
            "--output",
            str(evaluation_output),
            "--batch-size",
            str(args.batch_size),
            "--max-grid-images",
            str(args.max_grid_images),
            "--device",
            str(args.device),
        ]
        print("[RPC] Evaluating the existing decoder without retraining...", flush=True)
        _run_checked(evaluator_command, args.process_timeout_seconds)

        with (output / "holdout_evaluation_audit.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "holdout_samples": len(holdouts),
                    "holdout_labels": list(selected_labels),
                    "proxy_persisted_signals": (
                        ["z", "u", "dL/dz"] if args.capture_z else ["u", "dL/dz"]
                    ),
                    "decoder_retrained": False,
                    "proxy_loaded_victim_checkpoint": False,
                    "evaluator_loaded_victim_checkpoint": False,
                    "server_command": server_command,
                    "proxy_command": proxy_command,
                    "client_command": client_command,
                    "evaluator_command": evaluator_command,
                },
                handle,
                indent=2,
            )
        print(f"Holdout evaluation results: {output.resolve()}")
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
