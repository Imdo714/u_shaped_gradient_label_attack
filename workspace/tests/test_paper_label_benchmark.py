from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.shared_pretrained_encoder_drift_attack.paper_label_benchmark import (
    ResNet20Front,
    ResNet20Middle,
    RunningPrototypes,
    stratified_subset_indices,
)
from src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_condition import (
    _resolve_budgets,
)
from src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_sweep import (
    _conditions,
    build_parser,
)


@pytest.mark.parametrize(
    ("level", "channels", "spatial"),
    [(4, 32, 16), (5, 32, 16), (6, 32, 16), (7, 64, 8)],
)
def test_resnet20_split_shapes(level: int, channels: int, spatial: int) -> None:
    front = ResNet20Front(level).eval()
    middle = ResNet20Middle(level).eval()
    with torch.no_grad():
        z = front(torch.randn(2, 3, 32, 32))
        u = middle(z)
    assert z.shape == (2, channels, spatial, spatial)
    assert u.shape == (2, 64, 8, 8)


def test_stratified_fraction_keeps_every_class() -> None:
    indices, effective = stratified_subset_indices([0] * 40 + [1] * 40, 0.01, 42)
    assert len(indices) == 2
    assert {0 if index < 40 else 1 for index in indices} == {0, 1}
    assert effective == pytest.approx(0.025)


def test_observation_all_reuses_max_step() -> None:
    assert _resolve_budgets(["200", "1000", "all", "5000"], 20_000) == (
        200,
        1000,
        5000,
        20_000,
    )


def test_default_sweep_has_120_conditions(tmp_path: Path) -> None:
    args = build_parser().parse_args([])
    conditions = _conditions(args, tmp_path)
    assert len(conditions) == 2 * 4 * 3 * 5
    assert {condition.dataset for condition in conditions} == {"cifar10", "animal5"}


def test_running_prototype_predicts_nearest_class() -> None:
    prototype = RunningPrototypes(2)
    u = torch.zeros(2, 2, 4, 4)
    grad = torch.stack((torch.ones(2, 4, 4), -torch.ones(2, 4, 4)))
    labels = torch.tensor([0, 1])
    prototype.update(u, grad, labels, "gradient_only")
    assert prototype.predict(u, grad, "gradient_only").tolist() == [0, 1]
