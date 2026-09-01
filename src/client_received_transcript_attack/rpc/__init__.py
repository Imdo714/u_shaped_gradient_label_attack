"""Authorized loopback RPC components for process-separated experiments."""

from .protocol import Message, receive_message, send_message

__all__ = ["Message", "receive_message", "send_message"]
