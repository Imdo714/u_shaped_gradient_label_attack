from __future__ import annotations

from ...split_learning.h_model.client_tail_h_model import ClientTailHModel


class SurrogateHModel(ClientTailHModel):
    """Attacker-owned h clone trained by matching observed dL/du."""


__all__ = ["SurrogateHModel"]
