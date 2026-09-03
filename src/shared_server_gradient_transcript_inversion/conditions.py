from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalCondition:
    code: str
    name: str
    use_u: bool
    use_grad_z: bool
    use_z: bool = False

    @property
    def visible_signals(self) -> tuple[str, ...]:
        signals: list[str] = []
        if self.use_z:
            signals.append("z")
        if self.use_u:
            signals.append("u")
        if self.use_grad_z:
            signals.append("dL/dz")
        return tuple(signals)


CONDITIONS: dict[str, SignalCondition] = {
    "A": SignalCondition("A", "u_only", use_u=True, use_grad_z=False),
    "B": SignalCondition("B", "grad_z_only", use_u=False, use_grad_z=True),
    "C": SignalCondition("C", "u_grad_z", use_u=True, use_grad_z=True),
    "D": SignalCondition(
        "D", "z_u_grad_z", use_u=True, use_grad_z=True, use_z=True
    ),
}

_ALIASES = {
    condition.code.lower(): condition.code for condition in CONDITIONS.values()
} | {condition.name.lower(): condition.code for condition in CONDITIONS.values()}


def condition_from_name(value: str) -> SignalCondition:
    try:
        return CONDITIONS[_ALIASES[value.strip().lower()]]
    except KeyError as error:
        supported = ", ".join(
            f"{condition.code} ({condition.name})" for condition in CONDITIONS.values()
        )
        raise ValueError(f"unknown signal condition {value!r}; choose {supported}") from error


__all__ = ["CONDITIONS", "SignalCondition", "condition_from_name"]
