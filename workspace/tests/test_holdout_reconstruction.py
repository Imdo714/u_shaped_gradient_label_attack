from __future__ import annotations

import csv

from PIL import Image

from src.decoder.data.holdout_selection import select_holdout_records
from src.experiments.reconstruction import reconstruct_holdout_suite
from src.shared.data.image_dataset import make_loader


def _write_dataset(root):
    for split in ("train", "val", "test"):
        for label, color in (("cat", "red"), ("dog", "blue")):
            directory = root / split / label
            directory.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (8, 8), color).save(directory / f"{label}_{index}.png")


def _sample_ids(loader) -> set[str]:
    return {
        sample_id
        for _, _, sample_ids in loader
        for sample_id in sample_ids
    }


def test_selected_holdout_is_excluded_from_auxiliary_train_and_validation(tmp_path):
    _write_dataset(tmp_path)
    record = select_holdout_records(
        tmp_path,
        "train",
        ("cat", "dog"),
        holdouts_per_class=1,
        labels=("dog",),
        start_index=0,
    )[0]
    excluded = {record.sample_id}
    train = make_loader(
        tmp_path,
        "train",
        8,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        class_names=("cat", "dog"),
        exclude_sample_ids=excluded,
    )
    validation = make_loader(
        tmp_path,
        "val",
        8,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        class_names=("cat", "dog"),
        exclude_sample_ids=excluded,
    )
    holdout = make_loader(
        tmp_path,
        "train",
        8,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        class_names=("cat", "dog"),
        include_sample_ids=excluded,
    )

    assert record.sample_id not in _sample_ids(train)
    assert record.sample_id not in _sample_ids(validation)
    assert _sample_ids(holdout) == excluded


def test_suite_runs_all_conditions_and_writes_root_summaries(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dataset(data)
    output = tmp_path / "results"
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args.output, args.decoder_preset, args.signals, kwargs))
        return {"samples": 2, "psnr": 10.0}

    monkeypatch.setattr(reconstruct_holdout_suite, "run_reconstruction", fake_run)
    args = reconstruct_holdout_suite.build_parser().parse_args(
        [
            "--data",
            str(data),
            "--output",
            str(output),
            "--holdouts-per-class",
            "1",
        ]
    )
    summaries = reconstruct_holdout_suite.run(args)

    assert len(calls) == len(reconstruct_holdout_suite.CONDITIONS) == 9
    assert calls[0][3]["reuse_observations"] is False
    assert all(call[3]["reuse_observations"] for call in calls[1:])
    assert len(summaries) == 9
    with (output / "condition_summary.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 9
    with (output / "holdout_records.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    for condition in reconstruct_holdout_suite.CONDITIONS:
        assert calls[list(reconstruct_holdout_suite.CONDITIONS).index(condition)][0] == str(
            output / "conditions" / condition
        )


def test_suite_can_run_only_u_grad_z_condition(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dataset(data)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args.signals, kwargs["reuse_observations"]))
        return {"samples": 2, "psnr": 10.0}

    monkeypatch.setattr(reconstruct_holdout_suite, "run_reconstruction", fake_run)
    args = reconstruct_holdout_suite.build_parser().parse_args(
        [
            "--data",
            str(data),
            "--output",
            str(tmp_path / "results"),
            "--holdouts-per-class",
            "1",
            "--conditions",
            "strong_u_grad_z",
        ]
    )
    summaries = reconstruct_holdout_suite.run(args)

    assert calls == [("u,grad_z", False)]
    assert summaries[0]["condition"] == "strong_u_grad_z"


def test_suite_can_select_an_exact_total_holdout_count(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dataset(data)
    output = tmp_path / "results"

    monkeypatch.setattr(
        reconstruct_holdout_suite,
        "run_reconstruction",
        lambda args, **kwargs: {"samples": 3, "psnr": 10.0},
    )
    args = reconstruct_holdout_suite.build_parser().parse_args(
        [
            "--data",
            str(data),
            "--output",
            str(output),
            "--holdout-count",
            "3",
            "--conditions",
            "strong_u_grad_z",
        ]
    )
    reconstruct_holdout_suite.run(args)

    with (output / "holdout_records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["label"] for row in rows] == ["cat", "cat", "dog"]


def test_suite_can_restrict_total_holdouts_to_selected_labels(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dataset(data)
    output = tmp_path / "results"
    monkeypatch.setattr(
        reconstruct_holdout_suite,
        "run_reconstruction",
        lambda args, **kwargs: {"samples": 2, "psnr": 10.0},
    )
    args = reconstruct_holdout_suite.build_parser().parse_args(
        [
            "--data",
            str(data),
            "--output",
            str(output),
            "--holdout-count",
            "2",
            "--holdout-labels",
            "dog",
            "--conditions",
            "strong_u_grad_z",
        ]
    )
    reconstruct_holdout_suite.run(args)

    with (output / "holdout_records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["label"] for row in rows} == {"dog"}
