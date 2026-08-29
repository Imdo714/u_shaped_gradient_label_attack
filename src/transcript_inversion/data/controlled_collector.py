from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from ...decoder.data.gradient_label_predictor import GradientLabelPredictor
from ...decoder.data.image_scaling import denormalize_image
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.gradient_flow.gradient_exchange import observe_frozen_gradient_exchange
from .transcript_writer import EvaluatorTargetWriter, ServerTranscriptWriter


def _one_hot(label: int, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes).float()


def collect_controlled_transcripts(
    model: SplitLearningModel,
    loader,
    output_dir: str | Path,
    device: torch.device,
    num_classes: int,
    predictor: GradientLabelPredictor | None = None,
    soft_condition: bool = True,
    max_samples: int | None = None,
) -> Path:
    """Collect four server signals while keeping evaluator targets separate.

    ``predictor=None`` is the oracle-labelled auxiliary control. Victim/test
    collection should provide a gradient label predictor.
    """
    root = Path(output_dir)
    attacker_writer = ServerTranscriptWriter(root)
    evaluator_writer = EvaluatorTargetWriter(root)
    criterion = nn.CrossEntropyLoss()
    collected = 0
    for images, labels, sample_ids in loader:
        for index, sample_id in enumerate(sample_ids):
            if max_samples is not None and collected >= max_samples:
                break
            image = images[index : index + 1].to(device)
            label = labels[index : index + 1].to(device)
            exchange = observe_frozen_gradient_exchange(model, image, label, criterion)
            true_label = int(label.item())
            if predictor is None:
                predicted_label = true_label
                condition = _one_hot(true_label, num_classes)
                confidence = 1.0
                cluster_id = -1
            else:
                prediction = predictor.predict(exchange.grad_h_to_g[0], soft=soft_condition)
                predicted_label = prediction.label
                condition = prediction.probabilities
                confidence = prediction.confidence
                cluster_id = prediction.cluster_id
            attacker_writer.write(
                str(sample_id), exchange.smashed_z[0], exchange.server_output_u[0],
                exchange.grad_h_to_g[0], exchange.grad_g_to_f[0], condition,
                predicted_label, confidence, cluster_id,
            )
            evaluator_writer.write(
                str(sample_id), denormalize_image(image[0]), true_label
            )
            collected += 1
        if max_samples is not None and collected >= max_samples:
            break
    if not attacker_writer.rows:
        raise ValueError("nothing was collected")
    manifest = root / "manifest.csv"
    rows = [
        {
            "sample_id": attacker["sample_id"],
            "attacker_record": attacker["attacker_record"],
            "evaluator_target": evaluator["evaluator_target"],
        }
        for attacker, evaluator in zip(attacker_writer.rows, evaluator_writer.rows)
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "attacker_record", "evaluator_target"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


__all__ = ["collect_controlled_transcripts"]
