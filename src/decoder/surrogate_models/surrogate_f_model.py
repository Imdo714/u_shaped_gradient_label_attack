from __future__ import annotations

from ...split_learning.f_model.client_front_f_model import ClientFrontFModel


class SurrogateFModel(ClientFrontFModel):
    """Attacker-owned f clone trained by matching observed smashed data z."""


__all__ = ["SurrogateFModel"]
