from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import Tensor, nn

from ..paper_label_benchmark import (
    ClientParts,
    RunningPrototypes,
    exchange_step,
    infinite_batches,
    make_client,
    make_datasets,
    make_loader,
    make_middle,
    make_online_classifier,
    make_simulator_attack,
    resolve_device,
    seed_all,
    simulator_step,
)


SIGNAL_MODES = ("gradient_only", "u_gradient")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one paired U-shaped ResNet-20 label-inference condition. "
            "Learned C_psi, cosine prototypes, PCAT-label, and SDAR-label share "
            "one live victim/attacker/server trajectory."
        )
    )
    parser.add_argument("--dataset", choices=("cifar10", "animal5"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-level", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--aux-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--observation-budgets", nargs="+", default=["200", "1000", "5000", "all"]
    )
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--attack-start-step", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--target-learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sdar-lambda", type=float, default=0.02)
    parser.add_argument("--sdar-label-flip", type=float, default=0.2)
    parser.add_argument(
        "--signal-modes", nargs="+", choices=SIGNAL_MODES, default=list(SIGNAL_MODES)
    )
    parser.add_argument("--max-holdout-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    return parser


def _resolve_budgets(values: Sequence[str], max_steps: int) -> tuple[int, ...]:
    if max_steps < 1:
        raise ValueError("--max-steps must be positive")
    budgets: list[int] = []
    for value in values:
        budget = max_steps if value.lower() == "all" else int(value)
        if budget < 1 or budget > max_steps:
            raise ValueError("observation budgets must be within [1, --max-steps]")
        budgets.append(budget)
    return tuple(sorted(set(budgets)))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _holdout_loader(bundle, args: argparse.Namespace):
    dataset = bundle.holdout
    if args.max_holdout_samples > 0:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_holdout_samples, len(dataset)))
        )
    return make_loader(
        dataset,
        args.eval_batch_size,
        args.seed + 900_000,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def _holdout_exchange(
    client: ClientParts,
    middle: nn.Module,
    images: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    client.front.eval()
    client.tail.eval()
    middle.eval()
    with torch.enable_grad():
        z = client.front(images)
        u = middle(z)
        logits = client.tail(u)
        loss = nn.functional.cross_entropy(logits, labels)
        grad_z = torch.autograd.grad(loss, z, only_inputs=True)[0]
    # Cross-entropy uses batch mean. Remove the public batch-size scale so a
    # smaller final holdout batch has the same gradient magnitude convention.
    return z.detach(), u.detach(), (grad_z.detach() * len(labels)), logits.detach()


def _metrics(
    labels: list[int], predictions: list[int], num_classes: int
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "samples": len(labels),
    }


def _evaluate(
    budget: int,
    bundle,
    victim: ClientParts,
    middle: nn.Module,
    classifiers: dict[str, nn.Module],
    prototypes: dict[str, RunningPrototypes],
    pcat,
    sdar,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> list[dict[str, object]]:
    labels: list[int] = []
    target_predictions: list[int] = []
    predictions: dict[str, list[int]] = {
        "pcat_label": [],
        "sdar_label": [],
    }
    for mode in classifiers:
        predictions[f"learned_cpsi_{mode}"] = []
        predictions[f"cosine_prototype_{mode}"] = []
    for model in classifiers.values():
        model.eval()
    pcat.front.eval()
    pcat.tail.eval()
    sdar.front.eval()
    sdar.tail.eval()
    for images, target in _holdout_loader(bundle, args):
        images, target = images.to(device), target.to(device)
        z, u, grad_z, target_logits = _holdout_exchange(
            victim, middle, images, target
        )
        labels.extend(target.cpu().tolist())
        target_predictions.extend(target_logits.argmax(1).cpu().tolist())
        with torch.no_grad():
            for mode, classifier in classifiers.items():
                logits = classifier(u, grad_z)
                predictions[f"learned_cpsi_{mode}"].extend(
                    logits.argmax(1).cpu().tolist()
                )
                prototype_prediction = prototypes[mode].predict(u, grad_z, mode)
                predictions[f"cosine_prototype_{mode}"].extend(
                    prototype_prediction.tolist()
                )
            server_u = middle(z)
            predictions["pcat_label"].extend(
                pcat.tail(server_u).argmax(1).cpu().tolist()
            )
            predictions["sdar_label"].extend(
                sdar.tail(server_u).argmax(1).cpu().tolist()
            )
    common: dict[str, object] = {
        "dataset": args.dataset,
        "split_level": args.split_level,
        "requested_aux_fraction": args.aux_fraction,
        "effective_aux_fraction": bundle.effective_aux_fraction,
        "seed": args.seed,
        "observation_steps": budget,
        "sample_exposures": budget * args.batch_size,
        "attack_start_step": args.attack_start_step,
        "effective_attack_updates": max(0, budget - args.attack_start_step + 1),
    }
    num_classes = len(bundle.class_names)
    rows = [
        {
            **common,
            "method": "target_task",
            **_metrics(labels, target_predictions, num_classes),
        }
    ]
    confusion_root = output / "confusions" / f"step_{budget}"
    confusion_root.mkdir(parents=True, exist_ok=True)
    for method, values in predictions.items():
        rows.append(
            {**common, "method": method, **_metrics(labels, values, num_classes)}
        )
        matrix = confusion_matrix(labels, values, labels=list(range(num_classes)))
        np.savetxt(confusion_root / f"{method}.csv", matrix, delimiter=",", fmt="%d")
    return rows


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    budgets = _resolve_budgets(args.observation_budgets, args.max_steps)
    if not 0 < args.aux_fraction <= 1:
        raise ValueError("--aux-fraction must be in (0, 1]")
    if args.attack_start_step < 1 or args.attack_start_step > budgets[0]:
        raise ValueError("--attack-start-step must be between 1 and the first budget")
    if args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive and training batch must be >= 2")
    if not 0 <= args.sdar_label_flip < 1:
        raise ValueError("--sdar-label-flip must be in [0, 1)")
    if args.sdar_lambda < 0:
        raise ValueError("--sdar-lambda cannot be negative")
    return budgets


def run(args: argparse.Namespace) -> None:
    budgets = _validate_args(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_all(args.seed)
    bundle = make_datasets(
        args.dataset,
        args.data_root,
        args.aux_fraction,
        args.seed,
        download=args.download,
    )
    num_classes = len(bundle.class_names)
    victim = make_client(args.split_level, num_classes, device)
    attacker = make_client(args.split_level, num_classes, device)
    attacker.front.load_state_dict(copy.deepcopy(victim.front.state_dict()))
    attacker.tail.load_state_dict(copy.deepcopy(victim.tail.state_dict()))
    middle = make_middle(args.split_level, device)
    victim_optimizer = torch.optim.Adam(victim.parameters(), lr=args.target_learning_rate)
    attacker_optimizer = torch.optim.Adam(attacker.parameters(), lr=args.target_learning_rate)
    server_optimizer = torch.optim.Adam(middle.parameters(), lr=args.target_learning_rate)

    victim_loader = make_loader(
        bundle.victim_train,
        args.batch_size,
        args.seed + 10,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    auxiliary_loader = make_loader(
        bundle.auxiliary_train,
        args.batch_size,
        args.seed + 20,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    victim_batches = infinite_batches(victim_loader)
    auxiliary_batches = infinite_batches(auxiliary_loader)

    with torch.no_grad():
        probe = torch.zeros(2, 3, 32, 32, device=device)
        probe_z = attacker.front(probe)
        probe_u = middle(probe_z)
    classifiers = {
        mode: make_online_classifier(
            int(probe_u.shape[1]),
            int(probe_z.shape[1]),
            num_classes,
            mode,
            device,
        )
        for mode in dict.fromkeys(args.signal_modes)
    }
    classifier_optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
        for mode, model in classifiers.items()
    }
    prototypes = {mode: RunningPrototypes(num_classes) for mode in classifiers}
    pcat = make_simulator_attack(
        args.split_level,
        num_classes,
        device,
        learning_rate=args.attack_learning_rate,
        adversarial=False,
    )
    sdar = make_simulator_attack(
        args.split_level,
        num_classes,
        device,
        learning_rate=args.attack_learning_rate,
        adversarial=True,
    )

    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    start = time.monotonic()
    for step in range(1, max(budgets) + 1):
        victim_images, victim_labels = next(victim_batches)
        auxiliary_images, auxiliary_labels = next(auxiliary_batches)
        victim_images = victim_images.to(device, non_blocking=True)
        victim_labels = victim_labels.to(device, non_blocking=True)
        auxiliary_images = auxiliary_images.to(device, non_blocking=True)
        auxiliary_labels = auxiliary_labels.to(device, non_blocking=True)
        victim_exchange = exchange_step(
            victim,
            middle,
            victim_images,
            victim_labels,
            victim_optimizer,
            server_optimizer,
        )
        attacker_exchange = exchange_step(
            attacker,
            middle,
            auxiliary_images,
            auxiliary_labels,
            attacker_optimizer,
            server_optimizer,
        )
        pcat_losses = sdar_losses = None
        if step >= args.attack_start_step:
            for mode, model in classifiers.items():
                model.train()
                optimizer = classifier_optimizers[mode]
                optimizer.zero_grad(set_to_none=True)
                logits = model(attacker_exchange.u, attacker_exchange.grad_z)
                loss = nn.functional.cross_entropy(logits, auxiliary_labels)
                loss.backward()
                optimizer.step()
                prototypes[mode].update(
                    attacker_exchange.u,
                    attacker_exchange.grad_z,
                    auxiliary_labels,
                    mode,
                )
            pcat_losses = simulator_step(
                pcat,
                middle,
                auxiliary_images,
                auxiliary_labels,
                victim_exchange.z,
                num_classes=num_classes,
                adversarial_weight=0.0,
                label_flip_probability=0.0,
            )
            sdar_losses = simulator_step(
                sdar,
                middle,
                auxiliary_images,
                auxiliary_labels,
                victim_exchange.z,
                num_classes=num_classes,
                adversarial_weight=args.sdar_lambda,
                label_flip_probability=args.sdar_label_flip,
            )
        if step == 1 or step % 100 == 0 or step in budgets:
            record: dict[str, object] = {
                "step": step,
                "victim_loss": victim_exchange.loss,
                "victim_batch_accuracy": victim_exchange.accuracy,
                "attacker_client_loss": attacker_exchange.loss,
                "attacker_client_batch_accuracy": attacker_exchange.accuracy,
                "elapsed_seconds": time.monotonic() - start,
            }
            if pcat_losses is not None and sdar_losses is not None:
                record.update(
                    {f"pcat_{key}": value for key, value in pcat_losses.items()}
                )
                record.update(
                    {f"sdar_{key}": value for key, value in sdar_losses.items()}
                )
            training_rows.append(record)
        if step in budgets:
            rows.extend(
                _evaluate(
                    step,
                    bundle,
                    victim,
                    middle,
                    classifiers,
                    prototypes,
                    pcat,
                    sdar,
                    args,
                    device,
                    output,
                )
            )
            _write_csv(output / "metrics.csv", rows)
            _write_csv(output / "training_history.csv", training_rows)
            if args.save_models:
                torch.save(
                    {
                        "step": step,
                        "victim_front": victim.front.state_dict(),
                        "server_middle": middle.state_dict(),
                        "victim_tail": victim.tail.state_dict(),
                        "classifiers": {
                            key: value.state_dict() for key, value in classifiers.items()
                        },
                        "pcat_front": pcat.front.state_dict(),
                        "pcat_tail": pcat.tail.state_dict(),
                        "sdar_front": sdar.front.state_dict(),
                        "sdar_tail": sdar.tail.state_dict(),
                    },
                    output / f"models_step_{step}.pt",
                )
            print(
                f"dataset={args.dataset} level={args.split_level} "
                f"aux={args.aux_fraction:g} seed={args.seed} step={step} evaluated",
                flush=True,
            )

    config = {
        **vars(args),
        "resolved_output": str(output),
        "resolved_device": str(device),
        "resolved_observation_budgets": list(budgets),
        "observation_budget_unit": "communication_step_batch",
        "sample_exposure_formula": "observation_steps * batch_size",
        "gradient_scale": "dL/dz multiplied by public batch size (per-sample loss convention)",
        "class_names": list(bundle.class_names),
        "victim_train_size": bundle.victim_train_size,
        "auxiliary_pool_size": bundle.auxiliary_pool_size,
        "effective_aux_fraction": bundle.effective_aux_fraction,
        "comparison_scope": "U-shaped label inference only; no image decoder",
        "pcat_variant": "pseudo-front/pseudo-tail with common 100-step delay",
        "sdar_variant": (
            "pseudo-front/pseudo-tail plus representation discriminator and label flip"
        ),
        "paired_world": (
            "all methods share the same victim client, auxiliary attacker client, "
            "server middle, batches, and checkpoints within this condition"
        ),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "threat_model_audit.json").write_text(
        json.dumps(
            {
                "shared_world": True,
                "passive_attack_updates_do_not_modify_target_models": True,
                "victim_labels": "evaluator_only",
                "methods": {
                    "learned_cpsi": {
                        "training_access": [
                            "auxiliary_client_u",
                            "auxiliary_client_dL_dz",
                            "auxiliary_client_label",
                        ],
                        "victim_inference_access": ["victim_u", "victim_dL_dz"],
                        "server_weight_access": False,
                    },
                    "cosine_prototype": {
                        "training_access": [
                            "auxiliary_client_u",
                            "auxiliary_client_dL_dz",
                            "auxiliary_client_label",
                        ],
                        "victim_inference_access": ["victim_u", "victim_dL_dz"],
                        "server_weight_access": False,
                    },
                    "pcat_label": {
                        "training_access": [
                            "auxiliary_x",
                            "auxiliary_y",
                            "victim_z",
                            "server_g",
                        ],
                        "victim_inference_access": ["victim_z", "server_g"],
                        "client_weight_access": False,
                    },
                    "sdar_label": {
                        "training_access": [
                            "auxiliary_x",
                            "auxiliary_y",
                            "victim_z",
                            "server_g",
                        ],
                        "victim_inference_access": ["victim_z", "server_g"],
                        "client_weight_access": False,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "final_step": max(budgets),
                "elapsed_seconds": time.monotonic() - start,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(build_parser().parse_args())


__all__ = ["SIGNAL_MODES", "_resolve_budgets", "build_parser", "run"]
