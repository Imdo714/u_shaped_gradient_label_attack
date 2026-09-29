from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scipy.stats import t as student_t


CONTROLLED_MODES = ("none", "dynamic_flip")


@dataclass(frozen=True)
class Condition:
    victim_augmentation: str
    auxiliary_augmentation: str
    seed: int
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled 2x2 Animal5 augmentation ablation for the "
            "L7/10-image/1,000-step label-inference condition."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "workspace/results/shared_pretrained_encoder_drift_attack/conditiond"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="workspace/data/shared_pretrained_encoder_joint_attack/animal5",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--split-level", type=int, choices=(4, 5, 6, 7), default=7)
    parser.add_argument("--aux-fraction", type=float, default=0.05)
    parser.add_argument(
        "--observation-budgets", nargs="+", default=["200", "1000"]
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--attack-start-step", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--target-learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sdar-lambda", type=float, default=0.02)
    parser.add_argument("--sdar-label-flip", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if not 0 < args.aux_fraction <= 1:
        raise ValueError("--aux-fraction must be in (0, 1]")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    resolved = [
        args.max_steps if str(value).lower() == "all" else int(value)
        for value in args.observation_budgets
    ]
    if any(value < 1 or value > args.max_steps for value in resolved):
        raise ValueError("observation budgets must be within [1, --max-steps]")
    if args.attack_start_step < 1 or args.attack_start_step > min(resolved):
        raise ValueError("attack start must not exceed the first observation budget")


def build_conditions(args: argparse.Namespace, output: Path) -> tuple[Condition, ...]:
    conditions: list[Condition] = []
    for victim_mode in CONTROLLED_MODES:
        for auxiliary_mode in CONTROLLED_MODES:
            condition_root = output / (
                f"victim_{victim_mode}__attacker_{auxiliary_mode}"
            )
            for seed in args.seeds:
                conditions.append(
                    Condition(
                        victim_mode,
                        auxiliary_mode,
                        seed,
                        condition_root / f"seed_{seed}",
                    )
                )
    return tuple(conditions)


def _command(args: argparse.Namespace, condition: Condition) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_condition",
        "--dataset",
        "animal5",
        "--data-root",
        args.data_root,
        "--output",
        str(condition.output),
        "--split-level",
        str(args.split_level),
        "--aux-fraction",
        str(args.aux_fraction),
        "--seed",
        str(condition.seed),
        "--observation-budgets",
        *map(str, args.observation_budgets),
        "--max-steps",
        str(args.max_steps),
        "--attack-start-step",
        str(args.attack_start_step),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--target-learning-rate",
        str(args.target_learning_rate),
        "--attack-learning-rate",
        str(args.attack_learning_rate),
        "--sdar-lambda",
        str(args.sdar_lambda),
        "--sdar-label-flip",
        str(args.sdar_label_flip),
        "--victim-augmentation",
        condition.victim_augmentation,
        "--auxiliary-augmentation",
        condition.auxiliary_augmentation,
        "--num-workers",
        "0",
        "--device",
        args.device,
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate(
    conditions: Sequence[Condition],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw: list[dict[str, object]] = []
    for condition in conditions:
        metrics = condition.output / "metrics.csv"
        if not metrics.exists():
            continue
        for source in _read_csv(metrics):
            row: dict[str, object] = dict(source)
            row["condition_output"] = str(condition.output)
            raw.append(row)

    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        key = (
            str(row["victim_augmentation"]),
            str(row["auxiliary_augmentation"]),
            int(row["observation_steps"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    aggregate: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        accuracies = [float(value["accuracy"]) for value in values]
        macro_f1s = [float(value["macro_f1"]) for value in values]
        count = len(accuracies)
        accuracy_sd = statistics.stdev(accuracies) if count > 1 else 0.0
        macro_f1_sd = statistics.stdev(macro_f1s) if count > 1 else 0.0
        multiplier = float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
        aggregate.append(
            {
                "victim_augmentation": key[0],
                "auxiliary_augmentation": key[1],
                "observation_steps": key[2],
                "method": key[3],
                "runs": count,
                "mean_accuracy": statistics.mean(accuracies),
                "accuracy_sd": accuracy_sd,
                "accuracy_95ci_half_width": (
                    multiplier * accuracy_sd / math.sqrt(count)
                ),
                "mean_macro_f1": statistics.mean(macro_f1s),
                "macro_f1_sd": macro_f1_sd,
                "macro_f1_95ci_half_width": (
                    multiplier * macro_f1_sd / math.sqrt(count)
                ),
            }
        )
    return raw, aggregate


def _factorial_effects(raw: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    indexed: dict[tuple[str, str, int, str, int], float] = {}
    for row in raw:
        indexed[
            (
                str(row["victim_augmentation"]),
                str(row["auxiliary_augmentation"]),
                int(row["observation_steps"]),
                str(row["method"]),
                int(row["seed"]),
            )
        ] = float(row["accuracy"])

    observation_steps = sorted({int(row["observation_steps"]) for row in raw})
    methods = sorted({str(row["method"]) for row in raw})
    seeds = sorted({int(row["seed"]) for row in raw})
    rows: list[dict[str, object]] = []
    for step in observation_steps:
        for method in methods:
            effects: dict[str, list[float]] = {
                "victim_flip_when_attacker_none": [],
                "attacker_flip_when_victim_none": [],
                "joint_flip_vs_none": [],
                "victim_attacker_interaction": [],
            }
            for seed in seeds:
                keys = {
                    "a": ("none", "none", step, method, seed),
                    "b": ("dynamic_flip", "none", step, method, seed),
                    "c": ("none", "dynamic_flip", step, method, seed),
                    "d": ("dynamic_flip", "dynamic_flip", step, method, seed),
                }
                if not all(value in indexed for value in keys.values()):
                    continue
                a, b, c, d = (indexed[keys[name]] for name in ("a", "b", "c", "d"))
                effects["victim_flip_when_attacker_none"].append(b - a)
                effects["attacker_flip_when_victim_none"].append(c - a)
                effects["joint_flip_vs_none"].append(d - a)
                effects["victim_attacker_interaction"].append(d - b - c + a)
            for effect, values in effects.items():
                if not values:
                    continue
                count = len(values)
                sd = statistics.stdev(values) if count > 1 else 0.0
                multiplier = (
                    float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
                )
                rows.append(
                    {
                        "observation_steps": step,
                        "method": method,
                        "effect": effect,
                        "runs": count,
                        "mean_accuracy_delta": statistics.mean(values),
                        "accuracy_delta_sd": sd,
                        "accuracy_delta_95ci_half_width": (
                            multiplier * sd / math.sqrt(count)
                        ),
                    }
                )
    return rows


def _plan_rows(
    args: argparse.Namespace, conditions: Sequence[Condition]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        complete = (condition.output / "COMPLETED.json").exists()
        rows.append(
            {
                "index": index,
                "victim_augmentation": condition.victim_augmentation,
                "auxiliary_augmentation": condition.auxiliary_augmentation,
                "seed": condition.seed,
                "status": "complete" if complete else "pending",
                "output": str(condition.output),
                "command": subprocess.list2cmdline(_command(args, condition)),
            }
        )
    return rows


def _write_readme(output: Path, completed: int, total: int) -> None:
    output.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Animal5 L7 Horizontal-Flip 2x2 Ablation",
                "",
                "기존 100% 조건에서 Victim과 공격자 Horizontal Flip의 영향을 분리한다.",
                "",
                f"- 완료: {completed}/{total}",
                "- Dataset: Animal5",
                "- Split: L7",
                "- 공격자 라벨 이미지: 10장 (클래스당 2장)",
                "- Observation: 200, 1,000 communication steps",
                "- Attack start: step 101",
                "- 요청 batch size: 128; 실제 공격자 batch size: 10",
                "- Step 1,000의 공격기 학습 표본 노출: 9,000 (900 updates x 10장)",
                "- Seeds: 42, 43, 44, 45, 46",
                "- Factors: Victim flip none/dynamic x Attacker flip none/dynamic",
                "- dynamic_flip은 p=0.5이며 model RNG와 분리된 전용 RNG 사용",
                "- 기존 결과와 같은 분포의 재현 실험이며 bitwise 동일 view 재생은 아님",
                "- Holdout에는 증강을 적용하지 않음",
                "",
                "## 결과 파일",
                "",
                "- `experiment_plan.csv`: 실행 조건과 상태",
                "- `all_metrics.csv`: seed별 원시 결과",
                "- `aggregate_metrics.csv`: 조건별 평균과 95% CI",
                "- `factorial_effects.csv`: paired 주효과와 상호작용",
                "- 각 seed 디렉터리의 `metrics.csv`와 confusion matrices",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args, output)
    output.joinpath("sweep_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_output": str(output),
                "design": "2x2 victim/attacker controlled horizontal-flip ablation",
                "condition_count": len(conditions),
                "controlled_modes": list(CONTROLLED_MODES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(output / "experiment_plan.csv", _plan_rows(args, conditions))
    if args.plan_only:
        _write_readme(output, 0, len(conditions))
        return

    failures: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        complete = (condition.output / "COMPLETED.json").exists()
        if complete and args.resume:
            print(f"[{index}/{len(conditions)}] complete; skipping {condition.output}")
            continue
        print(
            f"[{index}/{len(conditions)}] victim={condition.victim_augmentation} "
            f"attacker={condition.auxiliary_augmentation} seed={condition.seed}",
            flush=True,
        )
        condition.output.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(_command(args, condition), check=False)
        if result.returncode != 0:
            failures.append(
                {
                    "victim_augmentation": condition.victim_augmentation,
                    "auxiliary_augmentation": condition.auxiliary_augmentation,
                    "seed": condition.seed,
                    "returncode": result.returncode,
                    "output": str(condition.output),
                }
            )
            _write_csv(output / "failures.csv", failures)
            if not args.continue_on_error:
                raise RuntimeError(f"condition failed with code {result.returncode}")
        _write_csv(output / "experiment_plan.csv", _plan_rows(args, conditions))

    raw, aggregate = _aggregate(conditions)
    _write_csv(output / "all_metrics.csv", raw)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    _write_csv(output / "factorial_effects.csv", _factorial_effects(raw))
    _write_csv(output / "failures.csv", failures)
    completed = sum(
        (condition.output / "COMPLETED.json").exists() for condition in conditions
    )
    _write_readme(output, completed, len(conditions))
    output.joinpath("SWEEP_COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "complete" if completed == len(conditions) else "partial",
                "completed_conditions": completed,
                "total_conditions": len(conditions),
                "failures": len(failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(build_parser().parse_args())


__all__ = ["CONTROLLED_MODES", "Condition", "build_conditions", "build_parser", "run"]
