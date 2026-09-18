from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .pipeline.run_online_transcript_label_attack import (
    OnlineLabelClassifier,
    OnlineLabelClassifierConfig,
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outputs),
            )
            if stride != 1 or inputs != outputs
            else nn.Identity()
        )

    def forward(self, value: Tensor) -> Tensor:
        identity = self.downsample(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + identity)


def _resnet20_stages() -> list[nn.Module]:
    """Return the CIFAR ResNet-20 stages used by the SDAR implementation."""

    return [
        nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
        ),
        BasicBlock(16, 32, stride=2),
        BasicBlock(32, 32),
        BasicBlock(32, 32),
        BasicBlock(32, 64, stride=2),
        BasicBlock(64, 64),
        BasicBlock(64, 64),
    ]


def _cut_index(split_level: int) -> int:
    # The official SDAR levels place level 4 after layer2.0 and level 7
    # after layer3.0. Stage zero contains stem + all three layer1 blocks.
    if split_level not in (4, 5, 6, 7):
        raise ValueError("split level must be one of 4, 5, 6, 7")
    return split_level - 2


class ResNet20Front(nn.Module):
    def __init__(self, split_level: int) -> None:
        super().__init__()
        stages = _resnet20_stages()
        self.network = nn.Sequential(*stages[: _cut_index(split_level)])

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class ResNet20Middle(nn.Module):
    def __init__(self, split_level: int) -> None:
        super().__init__()
        stages = _resnet20_stages()
        self.network = nn.Sequential(*stages[_cut_index(split_level) :])

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class ResNet20Tail(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.pool(value).flatten(1))


def initialize_resnet20(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)


@dataclass
class ClientParts:
    front: ResNet20Front
    tail: ResNet20Tail

    def parameters(self) -> Iterable[nn.Parameter]:
        yield from self.front.parameters()
        yield from self.tail.parameters()


@dataclass
class Exchange:
    z: Tensor
    u: Tensor
    grad_u: Tensor
    grad_z: Tensor
    labels: Tensor
    loss: float
    accuracy: float


def make_client(split_level: int, num_classes: int, device: torch.device) -> ClientParts:
    front = ResNet20Front(split_level).to(device)
    tail = ResNet20Tail(num_classes).to(device)
    initialize_resnet20(front)
    initialize_resnet20(tail)
    return ClientParts(front, tail)


def make_middle(split_level: int, device: torch.device) -> ResNet20Middle:
    middle = ResNet20Middle(split_level).to(device)
    initialize_resnet20(middle)
    return middle


def exchange_step(
    client: ClientParts,
    middle: ResNet20Middle,
    images: Tensor,
    labels: Tensor,
    client_optimizer: torch.optim.Optimizer,
    server_optimizer: torch.optim.Optimizer,
) -> Exchange:
    client.front.train()
    client.tail.train()
    middle.train()
    client_optimizer.zero_grad(set_to_none=True)
    server_optimizer.zero_grad(set_to_none=True)
    z = client.front(images)
    z.retain_grad()
    u = middle(z)
    u.retain_grad()
    logits = client.tail(u)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    if z.grad is None or u.grad is None:
        raise RuntimeError("split gradients were not retained")
    result = Exchange(
        z.detach(),
        u.detach(),
        u.grad.detach() * len(labels),
        z.grad.detach() * len(labels),
        labels.detach(),
        float(loss.detach()),
        float((logits.argmax(1) == labels).float().mean().detach()),
    )
    client_optimizer.step()
    server_optimizer.step()
    return result


@contextlib.contextmanager
def frozen_middle(middle: nn.Module) -> Iterator[None]:
    training = middle.training
    requires_grad = [parameter.requires_grad for parameter in middle.parameters()]
    middle.eval()
    for parameter in middle.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, required in zip(middle.parameters(), requires_grad):
            parameter.requires_grad_(required)
        middle.train(training)


class RepresentationDiscriminator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(32, min(128, channels * 2))
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).flatten()


@dataclass
class SimulatorAttack:
    front: ResNet20Front
    tail: ResNet20Tail
    optimizer: torch.optim.Optimizer
    discriminator: RepresentationDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None


def make_simulator_attack(
    split_level: int,
    num_classes: int,
    device: torch.device,
    *,
    learning_rate: float,
    adversarial: bool,
) -> SimulatorAttack:
    client = make_client(split_level, num_classes, device)
    optimizer = torch.optim.Adam(
        [*client.front.parameters(), *client.tail.parameters()], lr=learning_rate
    )
    if not adversarial:
        return SimulatorAttack(client.front, client.tail, optimizer)
    with torch.no_grad():
        channels = int(client.front(torch.zeros(1, 3, 32, 32, device=device)).shape[1])
    discriminator = RepresentationDiscriminator(channels).to(device)
    return SimulatorAttack(
        client.front,
        client.tail,
        optimizer,
        discriminator,
        torch.optim.Adam(discriminator.parameters(), lr=learning_rate * 0.02),
    )


def _flipped_labels(labels: Tensor, num_classes: int, probability: float) -> Tensor:
    if probability <= 0:
        return labels
    mask = torch.rand(labels.shape, device=labels.device) < probability
    random_labels = torch.randint(0, num_classes, labels.shape, device=labels.device)
    return torch.where(mask, random_labels, labels)


def simulator_step(
    attack: SimulatorAttack,
    middle: ResNet20Middle,
    auxiliary_images: Tensor,
    auxiliary_labels: Tensor,
    victim_z: Tensor,
    *,
    num_classes: int,
    adversarial_weight: float,
    label_flip_probability: float,
) -> dict[str, float]:
    attack.front.train()
    attack.tail.train()
    if attack.discriminator is not None:
        attack.discriminator.train()
    discriminator_loss = 0.0
    if attack.discriminator is not None and attack.discriminator_optimizer is not None:
        with torch.no_grad():
            fake_z = attack.front(auxiliary_images)
        attack.discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits = attack.discriminator(victim_z.detach())
        fake_logits = attack.discriminator(fake_z.detach())
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        discriminator_total = real_loss + fake_loss
        discriminator_total.backward()
        attack.discriminator_optimizer.step()
        discriminator_loss = float(discriminator_total.detach())

    attack.optimizer.zero_grad(set_to_none=True)
    with frozen_middle(middle):
        fake_z = attack.front(auxiliary_images)
        logits = attack.tail(middle(fake_z))
        training_labels = _flipped_labels(
            auxiliary_labels, num_classes, label_flip_probability
        )
        classification_loss = nn.functional.cross_entropy(logits, training_labels)
        loss = classification_loss
        generator_loss = torch.zeros((), device=auxiliary_images.device)
        if attack.discriminator is not None:
            generator_logits = attack.discriminator(fake_z)
            generator_loss = nn.functional.binary_cross_entropy_with_logits(
                generator_logits, torch.ones_like(generator_logits)
            )
            loss = loss + adversarial_weight * generator_loss
        loss.backward()
    attack.optimizer.step()
    return {
        "classification_loss": float(classification_loss.detach()),
        "generator_loss": float(generator_loss.detach()),
        "discriminator_loss": discriminator_loss,
    }


class RunningPrototypes:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.sums: Tensor | None = None
        self.counts = torch.zeros(num_classes, dtype=torch.long)

    @staticmethod
    def features(u: Tensor, grad_z: Tensor, signal_mode: str) -> Tensor:
        values: list[Tensor] = []
        if signal_mode == "u_gradient":
            u_feature = nn.functional.adaptive_avg_pool2d(u.float(), (4, 4)).flatten(1)
            values.append(nn.functional.normalize(u_feature, dim=1))
        grad = grad_z.float()
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        direction = grad / norm.view(-1, 1, 1, 1)
        direction = nn.functional.adaptive_avg_pool2d(direction, (4, 4)).flatten(1)
        grad_feature = torch.cat(
            (
                nn.functional.normalize(direction, dim=1),
                (torch.log10(norm) / 10.0).unsqueeze(1),
            ),
            dim=1,
        )
        values.append(nn.functional.normalize(grad_feature, dim=1))
        return nn.functional.normalize(torch.cat(values, dim=1), dim=1)

    def update(self, u: Tensor, grad_z: Tensor, labels: Tensor, signal_mode: str) -> None:
        features = self.features(u.detach().cpu(), grad_z.detach().cpu(), signal_mode)
        if self.sums is None:
            self.sums = torch.zeros(self.num_classes, features.shape[1])
        for label in range(self.num_classes):
            selected = features[labels.detach().cpu() == label]
            if len(selected):
                self.sums[label] += selected.sum(0)
                self.counts[label] += len(selected)

    def predict(self, u: Tensor, grad_z: Tensor, signal_mode: str) -> Tensor:
        if self.sums is None or bool((self.counts == 0).any()):
            raise RuntimeError("every class needs at least one prototype sample")
        prototypes = nn.functional.normalize(
            self.sums / self.counts.float().unsqueeze(1), dim=1
        )
        features = self.features(u.detach().cpu(), grad_z.detach().cpu(), signal_mode)
        return (features @ prototypes.T).argmax(1)


def make_online_classifier(
    u_channels: int,
    grad_channels: int,
    num_classes: int,
    signal_mode: str,
    device: torch.device,
) -> OnlineLabelClassifier:
    return OnlineLabelClassifier(
        OnlineLabelClassifierConfig(
            u_channels=u_channels,
            grad_z_channels=grad_channels,
            num_classes=num_classes,
            signal_mode=signal_mode,
            signal_spatial_size=4,
            signal_channels=32,
            hidden_channels=64,
            norm_channels=8,
            dropout=0.2,
        )
    ).to(device)


def stratified_subset_indices(
    targets: Sequence[int], fraction: float, seed: int
) -> tuple[list[int], float]:
    if not 0 < fraction <= 1:
        raise ValueError("auxiliary fraction must be in (0, 1]")
    labels = sorted(set(int(value) for value in targets))
    selected: list[int] = []
    for label in labels:
        indices = [i for i, value in enumerate(targets) if int(value) == label]
        random.Random(f"{seed}:aux:{label}").shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        selected.extend(indices[:count])
    selected.sort()
    return selected, len(selected) / len(targets)


def _cifar_partition_indices(targets: Sequence[int], seed: int) -> tuple[list[int], list[int]]:
    victim: list[int] = []
    auxiliary: list[int] = []
    for label in sorted(set(int(value) for value in targets)):
        indices = [i for i, value in enumerate(targets) if int(value) == label]
        random.Random(f"{seed}:partition:{label}").shuffle(indices)
        midpoint = len(indices) // 2
        victim.extend(indices[:midpoint])
        auxiliary.extend(indices[midpoint:])
    return sorted(victim), sorted(auxiliary)


@dataclass(frozen=True)
class DatasetBundle:
    victim_train: Dataset
    auxiliary_train: Dataset
    holdout: Dataset
    class_names: tuple[str, ...]
    requested_aux_fraction: float
    effective_aux_fraction: float
    victim_train_size: int
    auxiliary_pool_size: int


def make_datasets(
    dataset_name: str,
    data_root: str | Path,
    auxiliary_fraction: float,
    seed: int,
    *,
    download: bool,
) -> DatasetBundle:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.ToTensor()
    root = Path(data_root)
    if dataset_name == "cifar10":
        index_source = datasets.CIFAR10(root, train=True, download=download)
        victim_indices, auxiliary_pool_indices = _cifar_partition_indices(
            index_source.targets, seed=2026
        )
        victim_base = datasets.CIFAR10(
            root, train=True, transform=train_transform, download=download
        )
        auxiliary_base = datasets.CIFAR10(
            root, train=True, transform=train_transform, download=download
        )
        pool_targets = [index_source.targets[index] for index in auxiliary_pool_indices]
        relative, effective = stratified_subset_indices(
            pool_targets, auxiliary_fraction, seed
        )
        auxiliary_indices = [auxiliary_pool_indices[index] for index in relative]
        holdout = datasets.CIFAR10(
            root, train=False, transform=eval_transform, download=download
        )
        return DatasetBundle(
            Subset(victim_base, victim_indices),
            Subset(auxiliary_base, auxiliary_indices),
            holdout,
            tuple(index_source.classes),
            auxiliary_fraction,
            effective,
            len(victim_indices),
            len(auxiliary_pool_indices),
        )
    if dataset_name != "animal5":
        raise ValueError("dataset must be cifar10 or animal5")
    animal_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    animal_eval = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    victim = datasets.ImageFolder(root / "victim" / "train", transform=animal_transform)
    auxiliary_pool = datasets.ImageFolder(
        root / "attacker" / "train", transform=animal_transform
    )
    if victim.classes != auxiliary_pool.classes:
        raise ValueError("Animal5 victim and attacker class mappings differ")
    relative, effective = stratified_subset_indices(
        auxiliary_pool.targets, auxiliary_fraction, seed
    )
    holdout = datasets.ImageFolder(
        root / "victim" / "new_holdout", transform=animal_eval
    )
    if holdout.classes != victim.classes:
        raise ValueError("Animal5 holdout class mapping differs")
    return DatasetBundle(
        victim,
        Subset(auxiliary_pool, relative),
        holdout,
        tuple(victim.classes),
        auxiliary_fraction,
        effective,
        len(victim),
        len(auxiliary_pool),
    )


def infinite_batches(loader: DataLoader) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        for images, labels in loader:
            yield images, labels


def make_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    *,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last and len(dataset) >= batch_size,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = [
    "ClientParts",
    "DatasetBundle",
    "Exchange",
    "RepresentationDiscriminator",
    "ResNet20Front",
    "ResNet20Middle",
    "ResNet20Tail",
    "RunningPrototypes",
    "SimulatorAttack",
    "exchange_step",
    "frozen_middle",
    "infinite_batches",
    "initialize_resnet20",
    "make_client",
    "make_datasets",
    "make_loader",
    "make_middle",
    "make_online_classifier",
    "make_simulator_attack",
    "resolve_device",
    "seed_all",
    "simulator_step",
    "stratified_subset_indices",
]
