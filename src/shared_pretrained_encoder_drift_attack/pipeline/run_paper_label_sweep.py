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


@dataclass(frozen=True)
class Condition:
    dataset: str
    split_level: int
    aux_fraction: float
    seed: int
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete paired PCAT/SDAR/current-method label-inference sweep. "
            "Each condition is an isolated subprocess and completed conditions are resumable."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "workspace/results/shared_pretrained_encoder_drift_attack/"
            "pcat_sdar_fair_label_sweep"
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=("cifar10", "animal5"), default=["cifar10", "animal5"]
    )
    parser.add_argument("--split-levels", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--aux-fractions", nargs="+", type=float, default=[0.01, 0.05, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
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
        "--signal-modes",
        nargs="+",
        choices=("gradient_only", "u_gradient"),
        default=["gradient_only", "u_gradient"],
    )
    parser.add_argument(
        "--cifar-data-root", default="workspace/data/paper_label_benchmark/cifar10"
    )
    parser.add_argument(
        "--animal5-data-root",
        default="workspace/data/shared_pretrained_encoder_joint_attack/animal5",
    )
    parser.add_argument("--max-holdout-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate(args: argparse.Namespace) -> None:
    if any(level not in (4, 5, 6, 7) for level in args.split_levels):
        raise ValueError("--split-levels accepts only 4, 5, 6, 7")
    if any(not 0 < fraction <= 1 for fraction in args.aux_fractions):
        raise ValueError("--aux-fractions values must be in (0, 1]")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")


def _conditions(args: argparse.Namespace, output: Path) -> list[Condition]:
    conditions: list[Condition] = []
    for dataset in dict.fromkeys(args.datasets):
        for level in sorted(set(args.split_levels)):
            for fraction in sorted(set(args.aux_fractions)):
                for seed in dict.fromkeys(args.seeds):
                    condition_output = (
                        output
                        / dataset
                        / f"level_{level}"
                        / f"aux_{_float_token(fraction)}"
                        / f"seed_{seed}"
                    )
                    conditions.append(
                        Condition(dataset, level, fraction, seed, condition_output)
                    )
    return conditions


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


def _command(args: argparse.Namespace, condition: Condition) -> list[str]:
    data_root = (
        args.cifar_data_root
        if condition.dataset == "cifar10"
        else args.animal5_data_root
    )
    command = [
        sys.executable,
        "-m",
        "src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_condition",
        "--dataset",
        condition.dataset,
        "--data-root",
        str(data_root),
        "--output",
        str(condition.output),
        "--split-level",
        str(condition.split_level),
        "--aux-fraction",
        str(condition.aux_fraction),
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
        "--signal-modes",
        *map(str, args.signal_modes),
        "--max-holdout-samples",
        str(args.max_holdout_samples),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--download" if args.download else "--no-download",
    ]
    if args.save_models:
        command.append("--save-models")
    return command


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate(conditions: Sequence[Condition]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw: list[dict[str, object]] = []
    for condition in conditions:
        for row in _read_csv(condition.output / "metrics.csv"):
            raw.append(dict(row))
    grouped: dict[tuple[str, int, float, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        key = (
            str(row["dataset"]),
            int(row["split_level"]),
            float(row["requested_aux_fraction"]),
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
        f1_sd = statistics.stdev(macro_f1s) if count > 1 else 0.0
        multiplier = float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
        aggregate.append(
            {
                "dataset": key[0],
                "split_level": key[1],
                "requested_aux_fraction": key[2],
                "observation_steps": key[3],
                "method": key[4],
                "runs": count,
                "mean_accuracy": statistics.mean(accuracies),
                "accuracy_sd": accuracy_sd,
                "accuracy_95ci_half_width": multiplier * accuracy_sd / math.sqrt(count),
                "mean_macro_f1": statistics.mean(macro_f1s),
                "macro_f1_sd": f1_sd,
                "macro_f1_95ci_half_width": multiplier * f1_sd / math.sqrt(count),
            }
        )
    return raw, aggregate


def _write_readme(
    output: Path,
    args: argparse.Namespace,
    conditions: Sequence[Condition],
    aggregate: Sequence[dict[str, object]],
) -> None:
    completed = sum((condition.output / "COMPLETED.json").exists() for condition in conditions)
    lines = [
        "# PCAT·SDAR·통신 Gradient 라벨 추론 공정 비교",
        "",
        "이 디렉터리는 한 조건 안에서 동일한 U-Shaped ResNet-20 학습 trajectory를 공유한",
        "Learned C_psi, Cosine Prototype, PCAT-label, SDAR-label 결과를 저장한다.",
        "",
        "## 진행 상태",
        "",
        f"- 완료 조건: {completed}/{len(conditions)}",
        f"- Dataset: {', '.join(dict.fromkeys(args.datasets))}",
        f"- Split level: {', '.join(map(str, sorted(set(args.split_levels))))}",
        f"- Auxiliary fraction: {', '.join(f'{100*x:g}%' for x in sorted(set(args.aux_fractions)))}",
        f"- Observation budget: {', '.join(map(str, args.observation_budgets))} Communication Steps",
        "- `all`은 `--max-steps` 값이다.",
        "- Animal5의 1%는 5개 클래스가 모두 포함되도록 클래스당 최소 1장을 선택하므로 실제 비율이 2.5%다.",
        "",
        "## 비교 해석",
        "",
        "- 모든 공격은 같은 victim/attacker client, shared server g, batch 순서와 평가 Holdout을 공유한다.",
        "- Learned C_psi와 Cosine은 공격자 client의 라벨 있는 `u,dL/dz`로 학습한다.",
        "- PCAT-label은 `z`와 shared `g`를 이용한 pseudo-front/pseudo-tail을 학습한다.",
        "- SDAR-label은 PCAT 구성에 representation discriminator와 label flipping을 추가한다.",
        "- 본 스위프는 라벨 추론 비교이며 PCAT·SDAR의 이미지 decoder는 실행하지 않는다.",
        "- `observation_steps × batch_size`가 sample exposure다. 서로 다른 배치에서 같은 이미지가 반복될 수 있다.",
        "",
        "## 결과 파일",
        "",
        "- `experiment_plan.csv`: 전체 조건과 실행 상태",
        "- `all_metrics.csv`: seed별 원시 결과",
        "- `aggregate_metrics.csv`: 평균, 표준편차, 95% CI",
        "- 각 condition의 `run_config.json`: 실제 적용된 보조 데이터 비율과 위협 모델 세부사항",
    ]
    if aggregate:
        lines.extend(
            [
                "",
                "## 현재 집계",
                "",
                "집계 결과는 `aggregate_metrics.csv`에서 확인한다. 스위프 재실행 시 완료된 조건은 건너뛰고 집계를 갱신한다.",
            ]
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plan_rows(
    args: argparse.Namespace, conditions: Sequence[Condition]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        completed = (condition.output / "COMPLETED.json").exists()
        rows.append(
            {
                "condition_index": index,
                "dataset": condition.dataset,
                "split_level": condition.split_level,
                "aux_fraction": condition.aux_fraction,
                "seed": condition.seed,
                "status": "complete" if completed else "pending",
                "output": str(condition.output),
                "command": subprocess.list2cmdline(_command(args, condition)),
            }
        )
    return rows


def _refresh_outputs(
    output: Path, args: argparse.Namespace, conditions: Sequence[Condition]
) -> None:
    _write_csv(output / "experiment_plan.csv", _plan_rows(args, conditions))
    raw, aggregate = _aggregate(conditions)
    _write_csv(output / "all_metrics.csv", raw)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    _write_readme(output, args, conditions, aggregate)


def run(args: argparse.Namespace) -> None:
    _validate(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions = _conditions(args, output)
    (output / "sweep_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_output": str(output),
                "condition_count": len(conditions),
                "observation_budget_unit": "communication_step_batch",
                "methods": [
                    *[f"learned_cpsi_{mode}" for mode in args.signal_modes],
                    *[f"cosine_prototype_{mode}" for mode in args.signal_modes],
                    "pcat_label",
                    "sdar_label",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _refresh_outputs(output, args, conditions)
    if args.plan_only:
        print(f"Wrote {len(conditions)} conditions to {output / 'experiment_plan.csv'}")
        return
    failures: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        completed = condition.output / "COMPLETED.json"
        if args.resume and completed.exists():
            print(f"[{index}/{len(conditions)}] skip completed {condition.output}")
            continue
        condition.output.mkdir(parents=True, exist_ok=True)
        command = _command(args, condition)
        print(f"[{index}/{len(conditions)}] {subprocess.list2cmdline(command)}", flush=True)
        with (condition.output / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            failure = {
                "dataset": condition.dataset,
                "split_level": condition.split_level,
                "aux_fraction": condition.aux_fraction,
                "seed": condition.seed,
                "return_code": return_code,
                "output": str(condition.output),
            }
            failures.append(failure)
            _write_csv(output / "failures.csv", failures)
            _refresh_outputs(output, args, conditions)
            if not args.continue_on_error:
                raise RuntimeError(f"condition failed; inspect {condition.output / 'run.log'}")
        else:
            _refresh_outputs(output, args, conditions)
    _write_csv(output / "failures.csv", failures)
    _refresh_outputs(output, args, conditions)


if __name__ == "__main__":
    run(build_parser().parse_args())


__all__ = ["Condition", "_conditions", "_float_token", "build_parser", "run"]
