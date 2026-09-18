from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from .run_joint_transcript_attack import _device, _warmup_classifier
from .run_online_transcript_label_attack import (
    TranscriptBundle,
    _aggregate,
    _attack_validation_ids,
    _evaluate_attack,
    _evaluate_prototype_attack,
    _partition_victim_dataset,
    _train_attack_model,
    _write_csv,
)
from .run_online_z_attack import (
    ZTranscriptBundle,
    _evaluate_z_classifier,
    _evaluate_z_prototype,
    _train_z_classifier,
)
from .run_online_z_gradient_attack import (
    ZGradientTranscriptBundle,
    _evaluate_classifier as _evaluate_z_gradient_classifier,
    _evaluate_prototype as _evaluate_z_gradient_prototype,
    _run_world,
    _temporal_bundles,
    _train_classifier as _train_z_gradient_classifier,
    build_parser as _build_z_gradient_parser,
)


SIGNAL_MODES = ("gradient_only", "u_gradient", "z_only", "z_gradient")
SIGNAL_SEED_OFFSETS = {
    "gradient_only": 1,
    "u_gradient": 2,
    "z_only": 3,
    "z_gradient": 4,
}
SIGNAL_LABELS = {
    "gradient_only": "`dL/dz`-only",
    "u_gradient": "`u + dL/dz`",
    "z_only": "`z`-only",
    "z_gradient": "`z + dL/dz`",
}


def build_parser():
    parser = _build_z_gradient_parser()
    parser.description = (
        "Run a controlled label-inference comparison in which gradient-only, "
        "u+dL/dz, z-only, and z+dL/dz attacks reuse the exact same live Split Learning world."
    )
    parser.set_defaults(
        output=(
            "workspace/results/shared_pretrained_encoder_drift_attack/"
            "animal5_online_fair_z_comparison"
        )
    )
    parser.add_argument(
        "--signal-modes",
        nargs="+",
        choices=SIGNAL_MODES,
        default=list(SIGNAL_MODES),
    )
    return parser


def _as_gradient_bundle(bundle: ZGradientTranscriptBundle) -> TranscriptBundle:
    if bundle.u is None:
        raise ValueError("the controlled comparison requires observed server output u")
    return TranscriptBundle(
        bundle.u,
        bundle.grad_z,
        bundle.labels,
        bundle.sample_ids,
        bundle.rounds,
        bundle.server_steps,
    )


def _as_z_bundle(bundle: ZGradientTranscriptBundle) -> ZTranscriptBundle:
    return ZTranscriptBundle(
        bundle.z,
        bundle.labels,
        bundle.sample_ids,
        bundle.rounds,
        bundle.server_steps,
    )


def _train_signal_model(
    signal_mode: str,
    train_bundle: ZGradientTranscriptBundle,
    validation_bundle: ZGradientTranscriptBundle,
    catalog: ClassCatalog,
    args,
    device: torch.device,
    seed: int,
):
    if signal_mode in ("gradient_only", "u_gradient"):
        return _train_attack_model(
            _as_gradient_bundle(train_bundle),
            _as_gradient_bundle(validation_bundle),
            signal_mode,
            catalog.num_classes,
            args,
            device,
            seed,
        )
    if signal_mode == "z_only":
        return _train_z_classifier(
            _as_z_bundle(train_bundle),
            _as_z_bundle(validation_bundle),
            catalog.num_classes,
            args,
            device,
            seed,
        )
    return _train_z_gradient_classifier(
        train_bundle,
        validation_bundle,
        catalog.num_classes,
        args,
        device,
        seed,
    )


def _evaluate_learned(
    signal_mode: str,
    model,
    bundle: ZGradientTranscriptBundle,
    catalog: ClassCatalog,
    device: torch.device,
):
    if signal_mode in ("gradient_only", "u_gradient"):
        return _evaluate_attack(model, _as_gradient_bundle(bundle), catalog, device)
    if signal_mode == "z_only":
        return _evaluate_z_classifier(model, _as_z_bundle(bundle), catalog, device)
    return _evaluate_z_gradient_classifier(model, bundle, catalog, device)


def _evaluate_cosine(
    signal_mode: str,
    train_bundle: ZGradientTranscriptBundle,
    victim_bundle: ZGradientTranscriptBundle,
    catalog: ClassCatalog,
):
    if signal_mode in ("gradient_only", "u_gradient"):
        return _evaluate_prototype_attack(
            _as_gradient_bundle(train_bundle),
            _as_gradient_bundle(victim_bundle),
            signal_mode,
            catalog,
        )
    if signal_mode == "z_only":
        return _evaluate_z_prototype(
            _as_z_bundle(train_bundle), _as_z_bundle(victim_bundle), catalog
        )
    return _evaluate_z_gradient_prototype(train_bundle, victim_bundle, catalog)


def _table(
    rows: Sequence[dict[str, object]], signal_mode: str, attack_type: str
) -> list[str]:
    lookup = {
        (str(row["augmentation_mode"]), str(row["temporal_mode"])): row
        for row in rows
        if row["signal_mode"] == signal_mode and row["attack_type"] == attack_type
    }
    temporal_modes = ("single_latest", "multi_matched", "multi_natural")
    maxima = {
        temporal: max(
            float(lookup[(mode, temporal)]["mean_accuracy"])
            for mode in ("none", "fixed", "dynamic")
        )
        for temporal in temporal_modes
    }
    lines = [
        "| 공격자 증강 | Single-Latest | Multi-Matched | Multi-Natural |",
        "|---|---:|---:|---:|",
    ]
    for mode, display in (
        ("none", "None (무 증강)"),
        ("fixed", "Fixed"),
        ("dynamic", "Dynamic"),
    ):
        values: list[str] = []
        for temporal in temporal_modes:
            value = float(lookup[(mode, temporal)]["mean_accuracy"])
            text = f"{100 * value:.1f}%"
            values.append(f"**{text}**" if np.isclose(value, maxima[temporal]) else text)
        lines.append(f"| {display} | {' | '.join(values)} |")
    return lines


def _write_result_readme(
    output: Path,
    aggregate: Sequence[dict[str, object]],
    signal_modes: Sequence[str],
    seed_count: int,
    victims_per_seed: int,
) -> None:
    lines = [
        "# 공정 통제 `dL/dz` / `u + dL/dz` / `z` / `z + dL/dz` 라벨 추론 비교",
        "",
        "모든 신호 조건은 동일한 Split Learning 실행과 동일한 sample-level transcript를 재사용한다.",
        "공격기에 전달되는 텐서만 조건별로 다르다.",
    ]
    for signal_mode in signal_modes:
        label = SIGNAL_LABELS[signal_mode]
        lines.extend(
            [
                "",
                f"## Learned `C_psi`: {label}",
                "",
                *_table(aggregate, signal_mode, "learned_cpsi"),
                "",
                f"## Cosine Prototype: {label}",
                "",
                *_table(aggregate, signal_mode, "cosine_class_prototype"),
            ]
        )
    lines.extend(
        [
            "",
            f"- 각 표는 {seed_count}개 seed와 seed별 Victim {victims_per_seed}명의 평균 정확도다.",
            "- Victim 이미지와 라벨은 공격 모델 입력으로 사용하지 않는다.",
            "- Victim 라벨은 evaluator가 공격 성능을 계산할 때만 사용한다.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    collection_rounds = tuple(sorted(set(args.collection_rounds)))
    if not collection_rounds or collection_rounds[0] < 1:
        raise ValueError("collection rounds must start at round 1 or later")
    if collection_rounds[-1] > args.global_rounds:
        raise ValueError("collection rounds must not exceed global rounds")
    signal_modes = tuple(dict.fromkeys(args.signal_modes))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    catalog = ClassCatalog.discover(args.pretrain_data)
    if not (
        catalog.names == ClassCatalog.discover(args.attacker_data).names
        == ClassCatalog.discover(args.victim_data).names
    ):
        raise ValueError("pretrain, attacker, and victim class mappings must match")
    checkpoint = torch.load(
        args.pretrained_autoencoder, map_location=device, weights_only=False
    )
    cut_config = str(checkpoint["cut_config"])
    all_label_rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        seed_everything(seed)
        encoder = ClientFrontFModel(cut_config).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
        base_model = SplitLearningModel(
            encoder,
            ServerMiddleGModel(cut_config).to(device),
            ClientTailHModel(catalog.num_classes).to(device),
            cut_config,
        )
        pretrain_loader = make_loader(
            args.pretrain_data,
            "train",
            args.image_size,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            augment=True,
            class_names=catalog.names,
        )
        for row in _warmup_classifier(
            base_model,
            pretrain_loader,
            args.warmup_epochs,
            args.client_learning_rate,
            device,
        ):
            all_training_rows.append({"seed": seed, "stage": "warmup", **row})
        initial_f = copy.deepcopy(base_model.f_model.state_dict())
        initial_g = copy.deepcopy(base_model.g_model.state_dict())
        initial_h = copy.deepcopy(base_model.h_model.state_dict())
        victim_train, _ = _partition_victim_dataset(
            args.victim_data,
            "train",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=True,
        )
        victim_validation, _ = _partition_victim_dataset(
            args.victim_data,
            "val",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        victim_holdout, _ = _partition_victim_dataset(
            args.victim_data,
            "new_holdout",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        for augmentation_mode in args.augmentation_modes:
            attacker_bundle, victim_bundles, training_rows = _run_world(
                augmentation_mode,
                seed,
                initial_f,
                initial_g,
                initial_h,
                cut_config,
                catalog,
                victim_train,
                victim_validation,
                victim_holdout,
                collection_rounds,
                args,
                device,
            )
            all_training_rows.extend(training_rows)
            validation_ids = _attack_validation_ids(
                attacker_bundle,
                catalog.num_classes,
                args.attack_validation_fraction,
                seed,
            )
            temporal = _temporal_bundles(
                attacker_bundle,
                validation_ids,
                collection_rounds,
                args.matched_budget,
                seed,
            )
            for temporal_index, (temporal_mode, values) in enumerate(temporal.items()):
                train_bundle, validation_bundle, budget = values
                for signal_mode in signal_modes:
                    attack_seed = (
                        seed
                        + 100_000
                        + temporal_index * 100
                        + SIGNAL_SEED_OFFSETS[signal_mode]
                    )
                    model, history = _train_signal_model(
                        signal_mode,
                        train_bundle,
                        validation_bundle,
                        catalog,
                        args,
                        device,
                        attack_seed,
                    )
                    condition_root = (
                        output
                        / f"seed_{seed}"
                        / augmentation_mode
                        / temporal_mode
                        / signal_mode
                    )
                    condition_root.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": model.config.to_dict(),
                            "signal_mode": signal_mode,
                        },
                        condition_root / "learned_cpsi.pt",
                    )
                    _write_csv(
                        condition_root / "classifier_training_history.csv", history
                    )
                    for victim_index, victim_bundle in enumerate(
                        victim_bundles, start=1
                    ):
                        common = {
                            "seed": seed,
                            "augmentation_mode": augmentation_mode,
                            "temporal_mode": temporal_mode,
                            "signal_mode": signal_mode,
                            "budget": budget,
                            "training_records": len(train_bundle),
                            "victim_index": victim_index,
                            "victim_round": args.global_rounds,
                        }
                        for attack_type in (
                            "learned_cpsi",
                            "cosine_class_prototype",
                        ):
                            if attack_type == "learned_cpsi":
                                summary, class_rows, matrix = _evaluate_learned(
                                    signal_mode,
                                    model,
                                    victim_bundle,
                                    catalog,
                                    device,
                                )
                            else:
                                summary, class_rows, matrix = _evaluate_cosine(
                                    signal_mode,
                                    train_bundle,
                                    victim_bundle,
                                    catalog,
                                )
                            all_label_rows.append(
                                {**common, "attack_type": attack_type, **summary}
                            )
                            all_class_rows.extend(
                                {**common, "attack_type": attack_type, **row}
                                for row in class_rows
                            )
                            np.savetxt(
                                condition_root
                                / f"victim_{victim_index}_{attack_type}_confusion.csv",
                                matrix,
                                delimiter=",",
                                fmt="%d",
                            )
                    print(
                        f"seed={seed} augmentation={augmentation_mode} "
                        f"temporal={temporal_mode} signal={signal_mode} complete",
                        flush=True,
                    )

    aggregate = _aggregate(all_label_rows)
    _write_csv(output / "label_inference_summary.csv", all_label_rows)
    _write_csv(output / "aggregate_label_summary.csv", aggregate)
    _write_csv(output / "per_class_metrics.csv", all_class_rows)
    _write_csv(output / "split_training_history.csv", all_training_rows)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_output": str(output),
                "resolved_device": str(device),
                "class_names": list(catalog.names),
                "signal_modes": list(signal_modes),
                "collection_rounds": list(collection_rounds),
                "controlled_comparison": (
                    "all signal modes reuse the same live Split Learning world and "
                    "the same sample-level transcripts"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "threat_model_audit.json").write_text(
        json.dumps(
            {
                "attack_training_targets": ["attacker_y"],
                "victim_private_evaluator_only": ["victim_x", "victim_y"],
                "forbidden_access": [
                    "server_weights",
                    "full_model_snapshot",
                    "victim_local_weights",
                ],
                "signal_views": {
                    "gradient_only": ["dL_dz"],
                    "u_gradient": ["u", "dL_dz"],
                    "z_only": ["z"],
                    "z_gradient": ["z", "dL_dz"],
                },
                "relay_logging_exposure_required": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_result_readme(
        output,
        aggregate,
        signal_modes,
        len(args.seeds),
        args.num_victims,
    )


if __name__ == "__main__":
    run(build_parser().parse_args())


__all__ = [
    "SIGNAL_MODES",
    "_as_gradient_bundle",
    "_as_z_bundle",
    "build_parser",
    "run",
]
