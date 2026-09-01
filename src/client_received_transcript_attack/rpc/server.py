from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from .protocol import receive_message, send_message


def load_server_role(path: str | Path, device: torch.device) -> ServerMiddleGModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("role") != "server":
        raise ValueError("expected a server-role checkpoint")
    if "client_front" in checkpoint or "client_tail" in checkpoint:
        raise ValueError("server-role checkpoint contains forbidden client weights")
    model = ServerMiddleGModel(str(checkpoint["cut_config"]))
    model.load_state_dict(checkpoint["server_middle"])
    model.to(device).eval()
    return model


def _write_ready_file(path: str | Path | None, host: str, port: int) -> None:
    if path is None:
        return
    ready = Path(path)
    ready.parent.mkdir(parents=True, exist_ok=True)
    with ready.open("w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": host, "port": port}, handle, indent=2)


def serve(
    server_role_checkpoint: str | Path,
    host: str,
    port: int,
    expected_connections: int,
    device: torch.device,
    ready_file: str | Path | None = None,
) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the research RPC server is restricted to the loopback interface")
    if expected_connections < 1:
        raise ValueError("expected_connections must be positive")
    model = load_server_role(server_role_checkpoint, device)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        actual_port = int(listener.getsockname()[1])
        _write_ready_file(ready_file, host, actual_port)
        print(f"ServerMiddle RPC listening on {host}:{actual_port}", flush=True)
        for connection_index in range(expected_connections):
            connection, address = listener.accept()
            print(
                f"Accepted authorized client relay {connection_index + 1}/"
                f"{expected_connections} from {address}",
                flush=True,
            )
            pending: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            with connection:
                while True:
                    try:
                        message = receive_message(connection)
                    except EOFError:
                        break
                    if message.message_type == "forward":
                        if set(message.arrays) != {"smashed_z"}:
                            raise ValueError("forward message must contain only smashed_z")
                        if message.request_id in pending:
                            raise ValueError(f"duplicate request ID: {message.request_id}")
                        z = torch.from_numpy(message.arrays["smashed_z"]).to(device)
                        z = z.detach().requires_grad_(True)
                        u_server = model(z)
                        pending[message.request_id] = (z, u_server)
                        send_message(
                            connection,
                            "forward_result",
                            message.request_id,
                            {"server_output_u": u_server.detach().cpu().numpy()},
                        )
                    elif message.message_type == "backward":
                        if set(message.arrays) != {"grad_h_to_g"}:
                            raise ValueError("backward message must contain only grad_h_to_g")
                        if message.request_id not in pending:
                            raise ValueError(f"unknown request ID: {message.request_id}")
                        z, u_server = pending.pop(message.request_id)
                        grad_u = torch.from_numpy(message.arrays["grad_h_to_g"]).to(device)
                        grad_z = torch.autograd.grad(
                            u_server, z, grad_outputs=grad_u, retain_graph=False
                        )[0]
                        send_message(
                            connection,
                            "backward_result",
                            message.request_id,
                            {"grad_g_to_f": grad_z.detach().cpu().numpy()},
                        )
                    else:
                        raise ValueError(f"unexpected message type: {message.message_type}")
            if pending:
                raise RuntimeError(
                    f"connection closed with incomplete requests: {sorted(pending)}"
                )
    print("ServerMiddle RPC completed all expected connections", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a loopback-only ServerMiddle process with a server-only checkpoint."
    )
    parser.add_argument("--server-role-checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--expected-connections", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ready-file", default=None)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()
    serve(
        args.server_role_checkpoint,
        args.host,
        args.port,
        args.expected_connections,
        _device(args.device),
        args.ready_file,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "load_server_role", "serve"]
