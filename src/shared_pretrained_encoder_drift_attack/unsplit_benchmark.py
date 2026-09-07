from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    channels: int
    image_size: int
    class_names: tuple[str, ...]
    default_split_depth: int
    max_split_depth: int


SPECS = {
    "mnist": BenchmarkSpec(
        "mnist", 1, 28, tuple(str(index) for index in range(10)), 2, 6
    ),
    "fashion_mnist": BenchmarkSpec(
        "fashion_mnist",
        1,
        28,
        (
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ),
        2,
        6,
    ),
    "cifar10": BenchmarkSpec(
        "cifar10",
        3,
        32,
        (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
        4,
        8,
    ),
}


def benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark dataset: {name}") from error


def load_benchmark_datasets(
    name: str, root: str | Path, *, download: bool = True
) -> tuple[Dataset, Dataset]:
    transform = transforms.ToTensor()
    root = str(root)
    if name == "mnist":
        dataset_type = datasets.MNIST
    elif name == "fashion_mnist":
        dataset_type = datasets.FashionMNIST
    elif name == "cifar10":
        dataset_type = datasets.CIFAR10
    else:
        raise ValueError(f"unknown benchmark dataset: {name}")
    return (
        dataset_type(root, train=True, transform=transform, download=download),
        dataset_type(root, train=False, transform=transform, download=download),
    )


def first_example_per_class_indices(
    dataset: Dataset, number_of_classes: int = 10
) -> list[int]:
    """Match the target-selection rule in the official UnSplit demo."""

    result: list[int | None] = [None] * number_of_classes
    for index in range(len(dataset)):
        _, raw_label = dataset[index]
        label = int(raw_label)
        if 0 <= label < number_of_classes and result[label] is None:
            result[label] = index
        if all(value is not None for value in result):
            break
    if any(value is None for value in result):
        missing = [str(i) for i, value in enumerate(result) if value is None]
        raise ValueError("dataset is missing classes: " + ", ".join(missing))
    return [int(value) for value in result]


class LayerwiseBenchmarkNet(nn.Module):
    """Architectures and layer numbering used by the public UnSplit code."""

    def __init__(self, dataset: str) -> None:
        super().__init__()
        self.dataset = dataset
        if dataset in {"mnist", "fashion_mnist"}:
            self.flatten_at = 6
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(1, 8, kernel_size=5),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(8, 16, kernel_size=5),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2, 2),
                    nn.Linear(16 * 4 * 4, 120),
                    nn.ReLU(inplace=False),
                    nn.Linear(120, 84),
                    nn.ReLU(inplace=False),
                    nn.Linear(84, 10),
                ]
            )
        elif dataset == "cifar10":
            self.flatten_at = 15
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2, 2),
                    nn.Linear(4 * 4 * 128, 512),
                    nn.Sigmoid(),
                    nn.Linear(512, 10),
                ]
            )
        else:
            raise ValueError(f"unknown benchmark dataset: {dataset}")

    def _apply_range(self, value: Tensor, start: int, end: int) -> Tensor:
        for index in range(start, end + 1):
            if index == self.flatten_at and value.ndim > 2:
                value = value.flatten(1)
            value = self.layers[index](value)
        return value

    def forward_to(self, images: Tensor, split_depth: int) -> Tensor:
        return self._apply_range(images, 0, split_depth)

    def forward_from(self, smashed: Tensor, split_depth: int) -> Tensor:
        return self._apply_range(smashed, split_depth + 1, len(self.layers) - 1)

    def forward(self, images: Tensor) -> Tensor:
        return self._apply_range(images, 0, len(self.layers) - 1)

    def prefix_parameters(self, split_depth: int) -> list[nn.Parameter]:
        return [
            parameter
            for layer in self.layers[: split_depth + 1]
            for parameter in layer.parameters()
        ]

    def suffix_parameters(self, split_depth: int) -> list[nn.Parameter]:
        return [
            parameter
            for layer in self.layers[split_depth + 1 :]
            for parameter in layer.parameters()
        ]


def observe_transcript(
    model: LayerwiseBenchmarkNet,
    images: Tensor,
    labels: Tensor,
    split_depth: int,
) -> tuple[Tensor, Tensor]:
    z = model.forward_to(images, split_depth).detach().requires_grad_(True)
    logits = model.forward_from(z, split_depth)
    loss = functional.cross_entropy(logits, labels)
    gradient = torch.autograd.grad(loss, z)[0]
    return z.detach(), gradient.detach()


def _groups(channels: int) -> int:
    groups = min(channels, 8)
    while channels % groups:
        groups -= 1
    return groups


class _SignalAdapter(nn.Module):
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

    def forward(self, signal: Tensor) -> Tensor:
        if signal.ndim == 2:
            signal = signal[:, :, None, None]
        signal = functional.adaptive_avg_pool2d(
            signal, (self.spatial_size, self.spatial_size)
        )
        return self.network(signal)


class BenchmarkReconstructionDecoder(nn.Module):
    def __init__(
        self,
        signal_channels: int,
        output_channels: int,
        image_size: int,
        condition: str,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        normalized = condition.lower().replace("+", "_")
        settings = {
            "z_only": (True, False),
            "gradient_only": (False, True),
            "z_gradient": (True, True),
        }
        if normalized not in settings:
            raise ValueError(f"unknown condition: {condition}")
        self.condition = normalized
        self.use_z, self.use_gradient = settings[normalized]
        spatial_size = 7 if image_size == 28 else 8
        self.z_adapter = (
            _SignalAdapter(signal_channels, hidden_channels, spatial_size)
            if self.use_z
            else None
        )
        self.gradient_adapter = (
            _SignalAdapter(signal_channels, hidden_channels, spatial_size)
            if self.use_gradient
            else None
        )
        inputs = hidden_channels * (int(self.use_z) + int(self.use_gradient))
        self.fusion = nn.Sequential(
            nn.Conv2d(inputs, 128, 3, padding=1),
            nn.GroupNorm(_groups(128), 128),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.image_size = image_size

    @staticmethod
    def _normalize_gradient(gradient: Tensor) -> Tensor:
        flat_norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
        return gradient / flat_norm.view(-1, *([1] * (gradient.ndim - 1)))

    def forward(self, z: Tensor | None, gradient: Tensor | None) -> Tensor:
        features: list[Tensor] = []
        if self.z_adapter is not None:
            if z is None:
                raise ValueError("z is required")
            features.append(self.z_adapter(z))
        if self.gradient_adapter is not None:
            if gradient is None:
                raise ValueError("gradient is required")
            features.append(self.gradient_adapter(self._normalize_gradient(gradient)))
        fused = self.fusion(torch.cat(features, dim=1))
        resized = functional.interpolate(
            fused,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.output(resized)


def total_variation(images: Tensor) -> Tensor:
    horizontal = (images[:, :, 1:, :] - images[:, :, :-1, :]).square().mean()
    vertical = (images[:, :, :, 1:] - images[:, :, :, :-1]).square().mean()
    return horizontal + vertical


@dataclass(frozen=True)
class UnSplitResult:
    reconstruction: Tensor
    runtime_seconds: float
    feature_mse: float


def run_unsplit_attack(
    clone: LayerwiseBenchmarkNet,
    split_depth: int,
    target_z: Tensor,
    input_shape: Sequence[int],
    *,
    main_iters: int = 1000,
    input_iters: int = 100,
    model_iters: int = 100,
    learning_rate: float = 1e-3,
    lambda_tv: float = 0.1,
    lambda_l2: float = 1.0,
) -> UnSplitResult:
    """Faithful, efficient transcription of the public UnSplit alternating attack."""

    if min(main_iters, input_iters, model_iters) < 1:
        raise ValueError("UnSplit iteration counts must be positive")
    device = target_z.device
    clone.to(device)
    clone.train()
    prefix_parameters = clone.prefix_parameters(split_depth)
    prediction = torch.full(tuple(input_shape), 0.5, device=device, requires_grad=True)
    input_optimizer = torch.optim.Adam([prediction], lr=learning_rate, amsgrad=True)
    model_optimizer = torch.optim.Adam(prefix_parameters, lr=learning_rate, amsgrad=True)
    started = perf_counter()
    for _ in range(main_iters):
        for parameter in prefix_parameters:
            parameter.requires_grad_(False)
        for _ in range(input_iters):
            input_optimizer.zero_grad(set_to_none=True)
            predicted_z = clone.forward_to(prediction, split_depth)
            loss = (
                functional.mse_loss(predicted_z, target_z)
                + lambda_tv * total_variation(prediction)
                + lambda_l2 * prediction.square().mean()
            )
            loss.backward()
            input_optimizer.step()
        for parameter in prefix_parameters:
            parameter.requires_grad_(True)
        frozen_prediction = prediction.detach()
        for _ in range(model_iters):
            model_optimizer.zero_grad(set_to_none=True)
            predicted_z = clone.forward_to(frozen_prediction, split_depth)
            loss = functional.mse_loss(predicted_z, target_z)
            loss.backward()
            model_optimizer.step()
    with torch.no_grad():
        feature_mse = float(
            functional.mse_loss(clone.forward_to(prediction, split_depth), target_z)
        )
    return UnSplitResult(prediction.detach(), perf_counter() - started, feature_mse)


def minmax_normalize(images: Tensor) -> Tensor:
    minimum = images.amin(dim=tuple(range(1, images.ndim)), keepdim=True)
    maximum = images.amax(dim=tuple(range(1, images.ndim)), keepdim=True)
    return (images - minimum) / (maximum - minimum).clamp_min(1e-12)


__all__ = [
    "BenchmarkReconstructionDecoder",
    "BenchmarkSpec",
    "LayerwiseBenchmarkNet",
    "SPECS",
    "UnSplitResult",
    "benchmark_spec",
    "first_example_per_class_indices",
    "load_benchmark_datasets",
    "minmax_normalize",
    "observe_transcript",
    "run_unsplit_attack",
    "total_variation",
]
