from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.shared_server_gradient_transcript_inversion.conditions import (
    condition_from_name,
)
from src.shared_server_gradient_transcript_inversion.data import (
    ConditionedTranscriptDataset,
)
from src.shared_server_gradient_transcript_inversion.data_generation.prepare_letter_dataset import (
    prepare_letter_dataset,
)
from src.shared_server_gradient_transcript_inversion.data_generation.prepare_document_dataset import (
    prepare_document_dataset,
)
from src.shared_server_gradient_transcript_inversion.pipeline.collect_shared_server import (
    _client_specs,
)
from src.shared_server_gradient_transcript_inversion.statistics import (
    paired_bootstrap_comparison,
)


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        return {
            "transcript_id": f"sample-{index}",
            "server_output_u": torch.ones(4, 2, 2),
            "grad_g_to_f": torch.full((3, 4, 4), 2.0),
            "smashed_z": torch.full((3, 4, 4), 3.0),
            "target_image": torch.rand(3, 8, 8),
            "true_label": torch.tensor(index),
        }


def test_conditions_mask_omitted_signals_and_remove_z():
    condition_a = ConditionedTranscriptDataset(_Dataset(), condition_from_name("A"))[0]
    assert torch.count_nonzero(condition_a["server_output_u"])
    assert not torch.count_nonzero(condition_a["grad_g_to_f"])
    assert "smashed_z" not in condition_a

    condition_b = ConditionedTranscriptDataset(
        _Dataset(), condition_from_name("grad_z_only")
    )[0]
    assert not torch.count_nonzero(condition_b["server_output_u"])
    assert torch.count_nonzero(condition_b["grad_g_to_f"])

    condition_d = ConditionedTranscriptDataset(_Dataset(), condition_from_name("D"))[0]
    assert torch.count_nonzero(condition_d["smashed_z"])


def _write_metrics(path: Path, psnr_values: tuple[float, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["transcript_id", "mse", "mae", "psnr", "ssim"]
        )
        writer.writeheader()
        for index, psnr in enumerate(psnr_values):
            writer.writerow(
                {
                    "transcript_id": f"sample-{index}",
                    "mse": 1.0 / psnr,
                    "mae": 2.0 / psnr,
                    "psnr": psnr,
                    "ssim": psnr / 100.0,
                }
            )


def test_paired_bootstrap_reports_direction_and_win_rate(tmp_path):
    baseline = tmp_path / "a.csv"
    candidate = tmp_path / "c.csv"
    _write_metrics(baseline, (10.0, 11.0, 12.0))
    _write_metrics(candidate, (11.0, 12.0, 13.0))
    report = paired_bootstrap_comparison(
        baseline,
        candidate,
        baseline_name="A",
        candidate_name="C",
        samples=100,
        seed=7,
    )
    assert report["paired_samples"] == 3
    assert report["metrics"]["psnr"]["candidate_win_rate"] == 1.0
    assert report["metrics"]["mse"]["candidate_win_rate"] == 1.0


def test_victim_client_specs_require_unique_safe_names():
    specs = _client_specs(["one=client-a.pt", "two=client-b.pt"])
    assert specs == [("one", Path("client-a.pt")), ("two", Path("client-b.pt"))]


def test_letter_dataset_has_disjoint_balanced_domains(tmp_path):
    font = Path("C:/Windows/Fonts/arialbd.ttf")
    if not font.is_file():
        return
    output = prepare_letter_dataset(
        tmp_path / "letters",
        image_size=32,
        victim_train=4,
        victim_validation=2,
        victim_test=2,
        victim_holdout=2,
        auxiliary_train=4,
        auxiliary_validation=2,
        workers=1,
        font_paths=[font],
    )
    for domain, split, expected in (
        ("victim", "train", 4),
        ("victim", "new_holdout", 2),
        ("auxiliary_10k", "train", 4),
    ):
        images = list((output / domain / split).glob("*/*.jpg"))
        assert len(images) == expected
        with Image.open(images[0]) as image:
            assert image.size == (32, 32)


def test_document_dataset_is_synthetic_balanced_and_renderable(tmp_path):
    regular = Path("C:/Windows/Fonts/malgun.ttf")
    bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if not regular.is_file() or not bold.is_file():
        return
    output = prepare_document_dataset(
        tmp_path / "documents",
        image_size=128,
        victim_train=4,
        victim_validation=2,
        victim_test=2,
        victim_holdout=2,
        auxiliary_train=4,
        auxiliary_validation=2,
        workers=1,
        regular_font=regular,
        bold_font=bold,
    )
    for label in ("bankbook_copy", "interview_form"):
        images = list((output / "victim" / "train" / label).glob("*.jpg"))
        assert len(images) == 2
        with Image.open(images[0]) as image:
            assert image.size == (128, 128)
