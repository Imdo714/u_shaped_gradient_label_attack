from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


CLASS_NAMES = ("bankbook_copy", "interview_form")
DEFAULT_REGULAR_FONT = Path("C:/Windows/Fonts/malgun.ttf")
DEFAULT_BOLD_FONT = Path("C:/Windows/Fonts/malgunbd.ttf")


@dataclass(frozen=True)
class DocumentPlan:
    domain: str
    split: str
    label: str
    index: int
    seed: int


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = ":".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _balanced_counts(total: int) -> dict[str, int]:
    if total < len(CLASS_NAMES):
        raise ValueError("each split must contain at least one sample per class")
    base, remainder = divmod(total, len(CLASS_NAMES))
    return {
        label: base + int(index < remainder)
        for index, label in enumerate(CLASS_NAMES)
    }


def _plans(
    split_counts: dict[tuple[str, str], int], seed: int
) -> list[DocumentPlan]:
    plans: list[DocumentPlan] = []
    for (domain, split), total in split_counts.items():
        for label, count in _balanced_counts(total).items():
            for index in range(count):
                plans.append(
                    DocumentPlan(
                        domain,
                        split,
                        label,
                        index,
                        _stable_seed(seed, domain, split, label, index),
                    )
                )
    return plans


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _line(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill, width=1) -> None:
    draw.line(xy, fill=fill, width=width)


def _watermark(
    image: Image.Image, text: str, font_path: Path, rng: random.Random
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(font_path, max(14, image.width // 13))
    draw.text(
        (image.width * 0.12, image.height * 0.47),
        text,
        font=font,
        fill=(180, 30, 30, rng.randint(30, 55)),
    )
    overlay = overlay.rotate(-22, resample=getattr(Image, "Resampling", Image).BICUBIC)
    image.alpha_composite(overlay)


def _bankbook(
    size: int,
    rng: random.Random,
    regular_font: Path,
    bold_font: Path,
    serial: int,
) -> tuple[Image.Image, dict[str, str]]:
    paper = tuple(rng.randint(238, 255) for _ in range(3))
    image = Image.new("RGBA", (size, size), (*paper, 255))
    draw = ImageDraw.Draw(image)
    margin = round(size * 0.07)
    accent = rng.choice(((30, 83, 140), (28, 112, 91), (108, 62, 142)))
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=max(4, size // 40),
        outline=accent,
        width=max(2, size // 100),
    )
    title_font = _font(bold_font, max(16, size // 14))
    body_font = _font(regular_font, max(10, size // 24))
    small_font = _font(regular_font, max(8, size // 31))
    draw.ellipse(
        (margin + 7, margin + 8, margin + size // 8, margin + size // 8 + 1),
        fill=accent,
    )
    draw.text(
        (margin + size // 18, margin + size // 20),
        "가",
        font=body_font,
        fill="white",
        anchor="mm",
    )
    draw.text(
        (margin + size // 6, margin + size // 22),
        "가상은행 통장 사본",
        font=title_font,
        fill=accent,
    )
    draw.text(
        (margin + size // 6, margin + size // 7),
        "SYNTHETIC BANK DOCUMENT",
        font=small_font,
        fill=(80, 80, 80),
    )
    account = f"000-{serial % 100:02d}-{(serial * 7919) % 1000000:06d}"
    customer = f"가상고객{serial % 1000:03d}"
    y = margin + size // 4
    fields = (
        ("예금주", customer),
        ("계좌번호", account),
        ("개설일", f"202{serial % 5}.{(serial % 12) + 1:02d}.{(serial % 27) + 1:02d}"),
        ("상품명", rng.choice(("가상 보통예금", "연구용 저축예금"))),
    )
    label_x = margin + size // 22
    value_x = margin + size // 3
    row_height = size // 11
    for field, value in fields:
        draw.rectangle(
            (margin + 4, y - 3, size - margin - 4, y + row_height - 5),
            outline=(150, 160, 170),
            width=1,
        )
        draw.text((label_x, y), field, font=body_font, fill=(55, 55, 55))
        draw.text((value_x, y), value, font=body_font, fill=(20, 20, 20))
        y += row_height
    table_top = y + size // 35
    table_bottom = size - margin - size // 15
    _line(draw, (margin + 4, table_top, size - margin - 4, table_top), accent, 2)
    _line(draw, (margin + 4, table_bottom, size - margin - 4, table_bottom), accent, 1)
    for ratio in (0.27, 0.55, 0.76):
        x = round(margin + (size - 2 * margin) * ratio)
        _line(draw, (x, table_top, x, table_bottom), (150, 160, 170), 1)
    draw.text((margin + 10, table_top + 4), "거래일", font=small_font, fill=(40, 40, 40))
    draw.text((margin + size // 4, table_top + 4), "내용", font=small_font, fill=(40, 40, 40))
    draw.text((margin + size // 2, table_top + 4), "금액", font=small_font, fill=(40, 40, 40))
    draw.text((margin + size * 3 // 4, table_top + 4), "잔액", font=small_font, fill=(40, 40, 40))
    _watermark(image, "연구용 가상문서", bold_font, rng)
    return image, {"synthetic_account": account, "synthetic_name": customer}


def _interview_form(
    size: int,
    rng: random.Random,
    regular_font: Path,
    bold_font: Path,
    serial: int,
) -> tuple[Image.Image, dict[str, str]]:
    paper = tuple(rng.randint(242, 255) for _ in range(3))
    image = Image.new("RGBA", (size, size), (*paper, 255))
    draw = ImageDraw.Draw(image)
    margin = round(size * 0.065)
    accent = rng.choice(((48, 74, 121), (75, 75, 75), (73, 99, 78)))
    title_font = _font(bold_font, max(20, size // 10))
    body_font = _font(regular_font, max(10, size // 24))
    small_font = _font(regular_font, max(8, size // 31))
    draw.text(
        (size // 2, margin),
        "면접 지원서",
        font=title_font,
        fill=accent,
        anchor="ma",
    )
    draw.text(
        (size // 2, margin + size // 10),
        "SYNTHETIC INTERVIEW FORM",
        font=small_font,
        fill=(90, 90, 90),
        anchor="ma",
    )
    y = margin + size // 6
    form_left, form_right = margin, size - margin
    row_height = size // 11
    applicant = f"가상지원자{serial % 1000:03d}"
    application_id = f"FAKE-{serial % 10000:04d}"
    fields = (
        ("지원번호", application_id),
        ("성명", applicant),
        ("연락처", f"000-0000-{serial % 10000:04d}"),
        ("이메일", f"sample{serial % 1000:03d}@example.invalid"),
    )
    split_x = margin + size // 3
    for field, value in fields:
        draw.rectangle(
            (form_left, y, form_right, y + row_height),
            outline=(135, 140, 145),
            width=1,
        )
        _line(draw, (split_x, y, split_x, y + row_height), (135, 140, 145), 1)
        draw.text((form_left + 6, y + 5), field, font=body_font, fill=(55, 55, 55))
        value_font = small_font if field == "이메일" else body_font
        draw.text((split_x + 6, y + 5), value, font=value_font, fill=(20, 20, 20))
        y += row_height
    y += size // 35
    draw.text((form_left, y), "지원 분야", font=body_font, fill=accent)
    y += size // 18
    for index, text in enumerate(("개발", "연구", "디자인")):
        x = form_left + index * size // 4
        draw.rectangle((x, y, x + 9, y + 9), outline=(70, 70, 70), width=1)
        if index == serial % 3:
            draw.line((x + 1, y + 5, x + 4, y + 8, x + 9, y + 1), fill=accent, width=2)
        draw.text((x + 13, y - 2), text, font=small_font, fill=(45, 45, 45))
    y += size // 11
    draw.text((form_left, y), "경험 및 자기소개", font=body_font, fill=accent)
    y += size // 17
    for line_index in range(4):
        line_y = y + line_index * size // 20
        _line(draw, (form_left, line_y, form_right, line_y), (165, 170, 175), 1)
    draw.rectangle(
        (form_left, margin, form_right, size - margin), outline=accent, width=2
    )
    _watermark(image, "연구용 가상문서", bold_font, rng)
    return image, {"synthetic_application_id": application_id, "synthetic_name": applicant}


def _render(
    plan: DocumentPlan,
    destination: Path,
    image_size: int,
    regular_font: Path,
    bold_font: Path,
) -> dict[str, str]:
    rng = random.Random(plan.seed)
    np_rng = np.random.default_rng(plan.seed)
    if plan.label == "bankbook_copy":
        image, synthetic_fields = _bankbook(
            image_size, rng, regular_font, bold_font, plan.index
        )
    else:
        image, synthetic_fields = _interview_form(
            image_size, rng, regular_font, bold_font, plan.index
        )
    angle = rng.uniform(-3.5, 3.5)
    background = image.getpixel((0, 0))
    image = image.rotate(
        angle,
        resample=getattr(Image, "Resampling", Image).BICUBIC,
        fillcolor=background,
    )
    contrast = rng.uniform(0.88, 1.12)
    image = ImageEnhance.Contrast(image.convert("RGB")).enhance(contrast)
    blur_radius = rng.uniform(0.0, 0.55)
    if blur_radius > 0.12:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    noise_sigma = rng.uniform(0.0, 4.0)
    pixels = np.asarray(image, dtype=np.float32)
    pixels += np_rng.normal(0.0, noise_sigma, size=pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=rng.randint(86, 95), optimize=True)
    return {
        "domain": plan.domain,
        "split": plan.split,
        "label": plan.label,
        "index": str(plan.index),
        "seed": str(plan.seed),
        "relative_path": destination.as_posix(),
        "angle": f"{angle:.6f}",
        "contrast": f"{contrast:.6f}",
        "blur_radius": f"{blur_radius:.6f}",
        "noise_sigma": f"{noise_sigma:.6f}",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        **synthetic_fields,
    }


def prepare_document_dataset(
    output_root: Path,
    *,
    image_size: int = 256,
    victim_train: int = 600,
    victim_validation: int = 100,
    victim_test: int = 100,
    victim_holdout: int = 100,
    auxiliary_train: int = 10_000,
    auxiliary_validation: int = 100,
    seed: int = 42,
    workers: int = 8,
    regular_font: Path = DEFAULT_REGULAR_FONT,
    bold_font: Path = DEFAULT_BOLD_FONT,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if image_size < 128:
        raise ValueError("document image_size must be at least 128")
    if workers < 1:
        raise ValueError("workers must be positive")
    regular_font, bold_font = regular_font.resolve(), bold_font.resolve()
    if not regular_font.is_file() or not bold_font.is_file():
        raise FileNotFoundError("Korean TrueType font files were not found")
    split_counts = {
        ("victim", "train"): victim_train,
        ("victim", "val"): victim_validation,
        ("victim", "test"): victim_test,
        ("victim", "new_holdout"): victim_holdout,
        ("auxiliary_10k", "train"): auxiliary_train,
        ("auxiliary_10k", "val"): auxiliary_validation,
    }
    plans = _plans(split_counts, seed)
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for plan in plans:
                relative = (
                    Path(plan.domain)
                    / plan.split
                    / plan.label
                    / f"{plan.label}_{plan.index:05d}.jpg"
                )
                futures.append(
                    executor.submit(
                        _render,
                        plan,
                        staging / relative,
                        image_size,
                        regular_font,
                        bold_font,
                    )
                )
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["relative_path"] = Path(row["relative_path"]).relative_to(
                    staging
                ).as_posix()
                rows.append(row)
                if completed % 1000 == 0 or completed == len(futures):
                    print(
                        f"Generated synthetic documents: {completed}/{len(futures)}",
                        flush=True,
                    )
        rows.sort(key=lambda row: row["relative_path"])
        auxiliary_hashes = {
            row["sha256"]
            for row in rows
            if row["domain"] == "auxiliary_10k" and row["split"] == "train"
        }
        holdout_hashes = {
            row["sha256"]
            for row in rows
            if row["domain"] == "victim" and row["split"] == "new_holdout"
        }
        overlap = auxiliary_hashes & holdout_hashes
        if overlap:
            raise RuntimeError("auxiliary training and victim holdout contain duplicates")
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (staging / "generation_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "generator": "fully synthetic document renderer",
            "privacy_notice": (
                "No real person, bank, company, account, phone number, or email is used."
            ),
            "classes": list(CLASS_NAMES),
            "image_size": image_size,
            "seed": seed,
            "regular_font": regular_font.name,
            "bold_font": bold_font.name,
            "split_counts": {
                f"{domain}/{split}": count
                for (domain, split), count in split_counts.items()
            },
            "auxiliary_train_victim_holdout_sha256_overlap": len(overlap),
        }
        with (staging / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        (staging / "DATASET_SOURCE.md").write_text(
            "# Fully synthetic document dataset\n\n"
            "This dataset contains generated bankbook-copy and interview-form "
            "layouts for authorized privacy research. It contains no real PII, "
            "bank branding, account, company, applicant, phone number, or email. "
            "Every page is marked as a synthetic research document. Auxiliary "
            "training and victim holdout samples use disjoint seeds, and encoded "
            "file hashes are checked for overlap.\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except Exception:
        print(f"Generation failed; partial output remains at {staging}")
        raise
    print(f"Synthetic document dataset ready: {output_root.resolve()}")
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate privacy-safe synthetic bankbook and interview documents."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/data/synthetic_document_experiment"),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--victim-train", type=int, default=600)
    parser.add_argument("--victim-validation", type=int, default=100)
    parser.add_argument("--victim-test", type=int, default=100)
    parser.add_argument("--victim-holdout", type=int, default=100)
    parser.add_argument("--auxiliary-train", type=int, default=10_000)
    parser.add_argument("--auxiliary-validation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--regular-font", type=Path, default=DEFAULT_REGULAR_FONT)
    parser.add_argument("--bold-font", type=Path, default=DEFAULT_BOLD_FONT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_document_dataset(
        args.output,
        image_size=args.image_size,
        victim_train=args.victim_train,
        victim_validation=args.victim_validation,
        victim_test=args.victim_test,
        victim_holdout=args.victim_holdout,
        auxiliary_train=args.auxiliary_train,
        auxiliary_validation=args.auxiliary_validation,
        seed=args.seed,
        workers=args.workers,
        regular_font=args.regular_font,
        bold_font=args.bold_font,
    )


if __name__ == "__main__":
    main()


__all__ = ["CLASS_NAMES", "DocumentPlan", "build_parser", "prepare_document_dataset"]
