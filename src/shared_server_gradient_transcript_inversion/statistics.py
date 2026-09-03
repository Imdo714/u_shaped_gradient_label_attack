from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("mse", "mae", "psnr", "ssim")
HIGHER_IS_BETTER = {"mse": False, "mae": False, "psnr": True, "ssim": True}


def _metric_rows(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["transcript_id"]: {metric: float(row[metric]) for metric in METRICS}
        for row in rows
    }


def paired_bootstrap_comparison(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    *,
    baseline_name: str,
    candidate_name: str,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    baseline = _metric_rows(baseline_csv)
    candidate = _metric_rows(candidate_csv)
    common_ids = sorted(set(baseline) & set(candidate))
    if not common_ids:
        raise ValueError("paired evaluations have no common transcript IDs")
    if set(baseline) != set(candidate):
        raise ValueError("paired evaluations must contain identical transcript IDs")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(common_ids), size=(samples, len(common_ids)))
    result: dict[str, object] = {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "paired_samples": len(common_ids),
        "bootstrap_samples": samples,
        "metrics": {},
    }
    for metric in METRICS:
        base = np.asarray([baseline[key][metric] for key in common_ids])
        cand = np.asarray([candidate[key][metric] for key in common_ids])
        difference = cand - base
        boot_means = difference[indices].mean(axis=1)
        if HIGHER_IS_BETTER[metric]:
            wins = cand > base
        else:
            wins = cand < base
        result["metrics"][metric] = {
            "baseline_mean": float(base.mean()),
            "candidate_mean": float(cand.mean()),
            "mean_difference_candidate_minus_baseline": float(difference.mean()),
            "ci95": [
                float(np.quantile(boot_means, 0.025)),
                float(np.quantile(boot_means, 0.975)),
            ],
            "candidate_win_rate": float(wins.mean()),
            "higher_is_better": HIGHER_IS_BETTER[metric],
        }
    return result


def write_comparisons(
    condition_csvs: dict[str, Path],
    output_dir: str | Path,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> list[dict[str, object]]:
    requested_pairs = (("A", "C"), ("B", "C"), ("C", "D"))
    comparisons = [
        paired_bootstrap_comparison(
            condition_csvs[baseline],
            condition_csvs[candidate],
            baseline_name=baseline,
            candidate_name=candidate,
            samples=samples,
            seed=seed,
        )
        for baseline, candidate in requested_pairs
        if baseline in condition_csvs and candidate in condition_csvs
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_bootstrap_comparisons.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(comparisons, handle, indent=2)
    return comparisons


__all__ = ["paired_bootstrap_comparison", "write_comparisons"]
