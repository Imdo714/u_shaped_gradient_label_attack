from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Mapping

import numpy as np


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 1_048_576
MAX_ARRAY_BYTES = 512 * 1024 * 1024
ALLOWED_DTYPES = {"float32", "float64"}


@dataclass(frozen=True)
class Message:
    message_type: str
    request_id: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, str]


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("connection closed while receiving a framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(
    connection: socket.socket,
    message_type: str,
    request_id: str,
    arrays: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> None:
    if not message_type or not request_id:
        raise ValueError("message_type and request_id must be non-empty")
    encoded_arrays: list[tuple[str, np.ndarray]] = []
    descriptors: list[dict[str, object]] = []
    for name, value in (arrays or {}).items():
        array = np.ascontiguousarray(value)
        dtype_name = array.dtype.name
        if dtype_name not in ALLOWED_DTYPES:
            raise ValueError(f"unsupported array dtype: {dtype_name}")
        if array.nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"array {name!r} exceeds the payload limit")
        encoded_arrays.append((name, array))
        descriptors.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
            }
        )
    header = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "type": message_type,
            "request_id": request_id,
            "metadata": dict(metadata or {}),
            "arrays": descriptors,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("message header exceeds the size limit")
    connection.sendall(struct.pack("!Q", len(header)))
    connection.sendall(header)
    for _, array in encoded_arrays:
        connection.sendall(array.tobytes(order="C"))


def receive_message(connection: socket.socket) -> Message:
    header_length = struct.unpack("!Q", _receive_exact(connection, 8))[0]
    if header_length < 2 or header_length > MAX_HEADER_BYTES:
        raise ValueError(f"invalid header length: {header_length}")
    raw_header = _receive_exact(connection, header_length)
    header = json.loads(raw_header.decode("utf-8"))
    if header.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {header.get('version')}")
    message_type = header.get("type")
    request_id = header.get("request_id")
    metadata = header.get("metadata", {})
    descriptors = header.get("arrays", [])
    if not isinstance(message_type, str) or not message_type:
        raise ValueError("invalid message type")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("invalid request ID")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("metadata must contain only string keys and values")
    if not isinstance(descriptors, list):
        raise ValueError("array descriptor list is invalid")

    arrays: dict[str, np.ndarray] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("array descriptor must be an object")
        name = descriptor.get("name")
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        nbytes = descriptor.get("nbytes")
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError("array names must be unique non-empty strings")
        if dtype_name not in ALLOWED_DTYPES:
            raise ValueError(f"unsupported array dtype: {dtype_name}")
        if not isinstance(shape, list) or not all(
            isinstance(dimension, int) and dimension >= 0 for dimension in shape
        ):
            raise ValueError(f"invalid shape for array {name!r}")
        if not isinstance(nbytes, int) or nbytes < 0 or nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"invalid byte count for array {name!r}")
        dtype = np.dtype(dtype_name)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected_bytes:
            raise ValueError(f"byte count does not match shape for array {name!r}")
        payload = _receive_exact(connection, nbytes)
        arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    return Message(message_type, request_id, arrays, dict(metadata))


__all__ = [
    "ALLOWED_DTYPES",
    "MAX_ARRAY_BYTES",
    "MAX_HEADER_BYTES",
    "Message",
    "PROTOCOL_VERSION",
    "receive_message",
    "send_message",
]
