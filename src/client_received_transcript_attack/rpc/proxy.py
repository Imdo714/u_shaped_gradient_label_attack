from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from pathlib import Path

import numpy as np

from ..data.dataset import ATTACKER_KEYS, ATTACKER_KEYS_WITH_Z
from .protocol import receive_message, send_message


def _write_ready_file(path: str | Path | None, host: str, port: int) -> None:
    if path is None:
        return
    ready = Path(path)
    ready.parent.mkdir(parents=True, exist_ok=True)
    with ready.open("w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": host, "port": port}, handle, indent=2)


def observe_and_relay(
    listen_host: str,
    listen_port: int,
    server_host: str,
    server_port: int,
    output_dir: str | Path,
    expected_samples: int | None = None,
    ready_file: str | Path | None = None,
    capture_z: bool = False,
) -> Path:
    if listen_host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the research proxy is restricted to the loopback interface")
    output = Path(output_dir)
    attacker_dir = output / "attacker_records"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    pending_u: dict[str, np.ndarray] = {}
    pending_z: dict[str, np.ndarray] = {}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_host, listen_port))
        listener.listen(1)
        actual_port = int(listener.getsockname()[1])
        _write_ready_file(ready_file, listen_host, actual_port)
        print(f"Passive research proxy listening on {listen_host}:{actual_port}", flush=True)
        client_connection, _ = listener.accept()
        with client_connection, socket.create_connection(
            (server_host, server_port), timeout=30
        ) as server_connection:
            server_connection.settimeout(None)
            while True:
                try:
                    client_message = receive_message(client_connection)
                except EOFError:
                    break
                if capture_z and client_message.message_type == "forward":
                    if set(client_message.arrays) != {"smashed_z"}:
                        raise ValueError("forward message has unexpected arrays")
                    if client_message.request_id in pending_z:
                        raise ValueError(
                            f"duplicate z request ID: {client_message.request_id}"
                        )
                    pending_z[client_message.request_id] = client_message.arrays[
                        "smashed_z"
                    ].copy()
                send_message(
                    server_connection,
                    client_message.message_type,
                    client_message.request_id,
                    client_message.arrays,
                    client_message.metadata,
                )
                server_message = receive_message(server_connection)
                if server_message.request_id != client_message.request_id:
                    raise RuntimeError("server response request ID mismatch")
                if server_message.message_type == "forward_result":
                    if set(server_message.arrays) != {"server_output_u"}:
                        raise ValueError("forward result has unexpected arrays")
                    pending_u[server_message.request_id] = server_message.arrays[
                        "server_output_u"
                    ].copy()
                elif server_message.message_type == "backward_result":
                    if set(server_message.arrays) != {"grad_g_to_f"}:
                        raise ValueError("backward result has unexpected arrays")
                    if server_message.request_id not in pending_u:
                        raise RuntimeError("dL/dz arrived without its paired u")
                    u = pending_u.pop(server_message.request_id)
                    z = None
                    if capture_z:
                        if server_message.request_id not in pending_z:
                            raise RuntimeError("dL/dz arrived without its paired z")
                        z = pending_z.pop(server_message.request_id)
                    filename = f"{server_message.request_id}.npz"
                    payload = {
                        "server_output_u": u[0],
                        "grad_g_to_f": server_message.arrays["grad_g_to_f"][0],
                    }
                    if z is not None:
                        payload["smashed_z"] = z[0]
                    np.savez_compressed(attacker_dir / filename, **payload)
                    expected_keys = (
                        ATTACKER_KEYS_WITH_Z if capture_z else ATTACKER_KEYS
                    )
                    with np.load(attacker_dir / filename, allow_pickle=False) as record:
                        if set(record.files) != expected_keys:
                            raise RuntimeError("proxy persisted a forbidden attacker field")
                    rows.append(
                        {
                            "transcript_id": server_message.request_id,
                            "attacker_record": f"attacker_records/{filename}",
                        }
                    )
                else:
                    raise ValueError(
                        f"unexpected server response type: {server_message.message_type}"
                    )
                send_message(
                    client_connection,
                    server_message.message_type,
                    server_message.request_id,
                    server_message.arrays,
                    server_message.metadata,
                )

    if pending_u:
        raise RuntimeError(f"unpaired u messages remain: {sorted(pending_u)}")
    if pending_z:
        raise RuntimeError(f"unpaired z messages remain: {sorted(pending_z)}")
    if expected_samples is not None and len(rows) != expected_samples:
        raise RuntimeError(
            f"proxy captured {len(rows)} samples, expected {expected_samples}"
        )
    if not rows:
        raise ValueError("proxy did not capture any completed transcript")
    manifest = output / "attacker_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transcript_id", "attacker_record"])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "proxy_capture_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "samples": len(rows),
                "attacker_visible_signals": (
                    ["z", "u", "dL/dz"] if capture_z else ["u", "dL/dz"]
                ),
                "persisted_attacker_keys": sorted(
                    ATTACKER_KEYS_WITH_Z if capture_z else ATTACKER_KEYS
                ),
                "stored_client_to_server_payloads": ["z"] if capture_z else [],
                "loaded_victim_checkpoint": False,
            },
            handle,
            indent=2,
        )
    signal_text = "z/u/dL-dz" if capture_z else "u/dL-dz"
    print(f"Captured {len(rows)} paired {signal_text} transcripts", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relay authorized loopback traffic and persist server-to-client u/dL-dz, "
            "optionally including client-to-server z."
        )
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--capture-z", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    observe_and_relay(
        args.listen_host,
        args.listen_port,
        args.server_host,
        args.server_port,
        args.output,
        args.expected_samples,
        args.ready_file,
        args.capture_z,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "observe_and_relay"]
