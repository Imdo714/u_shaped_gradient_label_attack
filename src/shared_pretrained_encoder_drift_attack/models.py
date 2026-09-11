from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..decoder.data.image_scaling import l2_normalize_gradient


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class ReconstructionDecoderConfig:
    z_channels: int
    image_size: int
    grad_channels: int | None = None
    use_z: bool = True
    use_gradient: bool = False
    signal_spatial_size: int = 16
    signal_channels: int = 64
    base_channels: int = 256
    min_channels: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


class _Adapter(nn.Module):
    def __init__(self, inputs: int, outputs: int, spatial_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(inputs, outputs, 1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = torch.nn.functional.adaptive_avg_pool2d(
            value, (self.spatial_size, self.spatial_size)
        )
        return self.network(value)


class ReconstructionDecoder(nn.Module):
    """Separate z/gradient branches followed by a shared image decoder."""

    def __init__(self, config: ReconstructionDecoderConfig) -> None:
        super().__init__()
        if not config.use_z and not config.use_gradient:
            raise ValueError("at least one transcript signal must be enabled")
        if config.use_gradient and not config.grad_channels:
            raise ValueError("gradient condition requires grad_channels")
        ratio = config.image_size // config.signal_spatial_size
        if (
            config.image_size % config.signal_spatial_size
            or ratio < 1
            or ratio & (ratio - 1)
        ):
            raise ValueError("image_size / signal_spatial_size must be a power of two")
        self.config = config
        self.z_encoder = (
            _Adapter(config.z_channels, config.signal_channels, config.signal_spatial_size)
            if config.use_z
            else None
        )
        self.gradient_encoder = (
            _Adapter(
                int(config.grad_channels),
                config.signal_channels,
                config.signal_spatial_size,
            )
            if config.use_gradient
            else None
        )
        fusion_channels = config.signal_channels * (
            int(config.use_z) + int(config.use_gradient)
        )
        layers: list[nn.Module] = [
            nn.Conv2d(fusion_channels, config.base_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.base_channels), config.base_channels),
            nn.SiLU(),
        ]
        size = config.signal_spatial_size
        channels = config.base_channels
        while size < config.image_size:
            next_channels = max(config.min_channels, channels // 2)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, next_channels, 3, padding=1),
                    nn.GroupNorm(_groups(next_channels), next_channels),
                    nn.SiLU(),
                ]
            )
            channels = next_channels
            size *= 2
        layers.extend((nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid()))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor | None, gradient: Tensor | None = None) -> Tensor:
        features: list[Tensor] = []
        if self.z_encoder is not None:
            if z is None:
                raise ValueError("z is required by this decoder")
            features.append(self.z_encoder(z))
        if self.gradient_encoder is not None:
            if gradient is None:
                raise ValueError("gradient is required by this decoder")
            features.append(self.gradient_encoder(l2_normalize_gradient(gradient)))
        return self.decoder(torch.cat(features, dim=1))


@dataclass(frozen=True)
class LabelInferenceClassifierConfig:
    u_channels: int
    grad_z_channels: int
    num_classes: int
    use_u: bool = True
    use_gradient: bool = True
    signal_spatial_size: int = 8
    signal_channels: int = 64
    hidden_channels: int = 128
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class LabelInferenceClassifier(nn.Module):
    """Infer a private label from client-visible ``u`` and ``dL/dz``."""

    def __init__(self, config: LabelInferenceClassifierConfig) -> None:
        super().__init__()
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not config.use_u and not config.use_gradient:
            raise ValueError("at least one transcript signal must be enabled")
        self.config = config
        self.u_encoder = (
            _Adapter(
                config.u_channels,
                config.signal_channels,
                config.signal_spatial_size,
            )
            if config.use_u
            else None
        )
        self.gradient_encoder = (
            _Adapter(
                config.grad_z_channels,
                config.signal_channels,
                config.signal_spatial_size,
            )
            if config.use_gradient
            else None
        )
        fusion_channels = config.signal_channels * (
            int(config.use_u) + int(config.use_gradient)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(fusion_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.hidden_channels), config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.hidden_channels), config.hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.num_classes),
        )

    def forward(self, u: Tensor | None, grad_z: Tensor | None) -> Tensor:
        features: list[Tensor] = []
        if self.u_encoder is not None:
            if u is None:
                raise ValueError("u is required by this classifier")
            features.append(self.u_encoder(u))
        if self.gradient_encoder is not None:
            if grad_z is None:
                raise ValueError("dL/dz is required by this classifier")
            features.append(self.gradient_encoder(l2_normalize_gradient(grad_z)))
        return self.classifier(torch.cat(features, dim=1))


@dataclass(frozen=True)
class DriftRobustLabelClassifierConfig:
    u_channels: int
    grad_z_channels: int
    num_classes: int
    use_u: bool = False
    use_gradient_norm: bool = True
    signal_spatial_size: int = 8
    signal_channels: int = 64
    hidden_channels: int = 128
    norm_channels: int = 16
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class DriftRobustLabelClassifier(nn.Module):
    """Label classifier with gradient magnitude and optional gated ``u`` logits."""

    def __init__(self, config: DriftRobustLabelClassifierConfig) -> None:
        super().__init__()
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.config = config
        self.gradient_encoder = _Adapter(
            config.grad_z_channels,
            config.signal_channels,
            config.signal_spatial_size,
        )
        self.gradient_refiner = nn.Sequential(
            nn.Conv2d(config.signal_channels, config.signal_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.signal_channels), config.signal_channels),
            nn.SiLU(),
            nn.Conv2d(config.signal_channels, config.signal_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.signal_channels), config.signal_channels),
            nn.SiLU(),
        )
        self.norm_encoder = (
            nn.Sequential(
                nn.Linear(1, config.norm_channels),
                nn.SiLU(),
                nn.Linear(config.norm_channels, config.norm_channels),
                nn.SiLU(),
            )
            if config.use_gradient_norm
            else None
        )
        gradient_features = config.signal_channels + (
            config.norm_channels if config.use_gradient_norm else 0
        )
        self.gradient_head = nn.Sequential(
            nn.Linear(gradient_features, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.num_classes),
        )
        self.u_encoder = (
            _Adapter(
                config.u_channels,
                config.signal_channels,
                config.signal_spatial_size,
            )
            if config.use_u
            else None
        )
        self.u_refiner = (
            nn.Sequential(
                nn.Conv2d(config.signal_channels, config.signal_channels, 3, padding=1),
                nn.GroupNorm(_groups(config.signal_channels), config.signal_channels),
                nn.SiLU(),
                nn.Conv2d(config.signal_channels, config.signal_channels, 3, padding=1),
                nn.GroupNorm(_groups(config.signal_channels), config.signal_channels),
                nn.SiLU(),
            )
            if config.use_u
            else None
        )
        self.u_head = (
            nn.Sequential(
                nn.Linear(config.signal_channels, config.hidden_channels),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_channels, config.num_classes),
            )
            if config.use_u
            else None
        )
        gate_features = gradient_features + (
            config.signal_channels if config.use_u else 0
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(gate_features, config.hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(config.hidden_channels // 2, 1),
                nn.Sigmoid(),
            )
            if config.use_u
            else None
        )
        if self.gate is not None:
            # Start by trusting the empirically stronger gradient head. The
            # optimizer can open the u branch only when it reduces label loss.
            final_linear = self.gate[2]
            if isinstance(final_linear, nn.Linear):
                nn.init.constant_(final_linear.bias, -2.0)

    @staticmethod
    def _pooled(value: Tensor) -> Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)

    @staticmethod
    def gradient_log_norm(gradient: Tensor) -> Tensor:
        # Division keeps ordinary cross-entropy gradient norms near a convenient
        # numerical scale while retaining their relative magnitude information.
        norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
        return (torch.log10(norm) / 10.0).unsqueeze(1)

    def forward_with_gate(
        self, u: Tensor | None, grad_z: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        direction = l2_normalize_gradient(grad_z)
        gradient_vector = self._pooled(
            self.gradient_refiner(self.gradient_encoder(direction))
        )
        gradient_parts = [gradient_vector]
        if self.norm_encoder is not None:
            gradient_parts.append(self.norm_encoder(self.gradient_log_norm(grad_z)))
        gradient_features = torch.cat(gradient_parts, dim=1)
        gradient_logits = self.gradient_head(gradient_features)
        if (
            self.u_encoder is None
            or self.u_refiner is None
            or self.u_head is None
            or self.gate is None
        ):
            return gradient_logits, None
        if u is None:
            raise ValueError("u is required by the gated classifier")
        u_vector = self._pooled(self.u_refiner(self.u_encoder(u)))
        gate = self.gate(torch.cat((gradient_features, u_vector), dim=1))
        logits = gradient_logits + gate * self.u_head(u_vector)
        return logits, gate

    def forward(self, u: Tensor | None, grad_z: Tensor) -> Tensor:
        return self.forward_with_gate(u, grad_z)[0]


def label_classifier_for_condition(
    condition: str,
    u_channels: int,
    grad_z_channels: int,
    num_classes: int,
    signal_spatial_size: int = 8,
    signal_channels: int = 64,
    hidden_channels: int = 128,
) -> LabelInferenceClassifier:
    normalized = condition.lower().replace("+", "_")
    settings = {
        "u_only": (True, False),
        "gradient_only": (False, True),
        "grad_z_only": (False, True),
        "u_gradient": (True, True),
        "u_grad_z": (True, True),
    }
    if normalized not in settings:
        raise ValueError(f"unknown label attack condition: {condition}")
    use_u, use_gradient = settings[normalized]
    return LabelInferenceClassifier(
        LabelInferenceClassifierConfig(
            u_channels=u_channels,
            grad_z_channels=grad_z_channels,
            num_classes=num_classes,
            use_u=use_u,
            use_gradient=use_gradient,
            signal_spatial_size=signal_spatial_size,
            signal_channels=signal_channels,
            hidden_channels=hidden_channels,
        )
    )


def decoder_for_condition(
    condition: str,
    z_channels: int,
    image_size: int,
    signal_spatial_size: int = 16,
    signal_channels: int = 64,
    base_channels: int = 256,
    min_channels: int = 32,
) -> ReconstructionDecoder:
    normalized = condition.lower().replace("+", "_")
    settings = {
        "z_only": (True, False),
        "gradient_only": (False, True),
        "z_gradient": (True, True),
        "z_grad": (True, True),
    }
    if normalized not in settings:
        raise ValueError(f"unknown attack condition: {condition}")
    use_z, use_gradient = settings[normalized]
    return ReconstructionDecoder(
        ReconstructionDecoderConfig(
            z_channels=z_channels,
            grad_channels=z_channels,
            image_size=image_size,
            use_z=use_z,
            use_gradient=use_gradient,
            signal_spatial_size=signal_spatial_size,
            signal_channels=signal_channels,
            base_channels=base_channels,
            min_channels=min_channels,
        )
    )


__all__ = [
    "DriftRobustLabelClassifier",
    "DriftRobustLabelClassifierConfig",
    "LabelInferenceClassifier",
    "LabelInferenceClassifierConfig",
    "ReconstructionDecoder",
    "ReconstructionDecoderConfig",
    "decoder_for_condition",
    "label_classifier_for_condition",
]
