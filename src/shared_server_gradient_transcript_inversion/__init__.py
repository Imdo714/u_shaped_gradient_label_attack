"""Shared-server gradient transcript inversion experiments.

The package builds on the role-separated RPC implementation in
``src.client_received_transcript_attack``.  Attack training remains
checkpoint-free: only persisted transcript manifests and public targets are
passed to the attacker process.
"""

from .conditions import SignalCondition, condition_from_name
from .data import ConditionedTranscriptDataset

__all__ = ["ConditionedTranscriptDataset", "SignalCondition", "condition_from_name"]
