from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class ImageFolderWithID(datasets.ImageFolder):
    def sample_id(self, index: int) -> str:
        path, _ = self.samples[index]
        relative_path = Path(path).relative_to(self.root).as_posix()
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
        return f"sample_{digest}"

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        # ImageFolder paths contain the class directory. Never expose that path
        # as the attacker-visible ID; both loggers instead join on this opaque ID.
        return image, label, self.sample_id(index)


def image_transform(image_size: int, augment: bool = False):
    operations: list[object] = [transforms.Resize((image_size, image_size))]
    if augment:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


def validate_class_mapping(dataset: datasets.ImageFolder, class_names: tuple[str, ...]) -> None:
    expected = {name: index for index, name in enumerate(class_names)}
    if dataset.class_to_idx != expected:
        raise ValueError(
            f"Expected class folders/mapping {expected}, found {dataset.class_to_idx}. "
            "Every split must contain the same alphabetically sorted class directories."
        )


def make_loader(
    data_dir: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool = False,
    class_names: tuple[str, ...] | None = None,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
) -> DataLoader:
    root = Path(data_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Dataset split not found: {root}")
    dataset = ImageFolderWithID(
        root,
        transform=image_transform(image_size, augment),
        allow_empty=True,
    )
    if class_names is not None:
        validate_class_mapping(dataset, class_names)
    selected_indices = [
        index
        for index in range(len(dataset))
        if (include_sample_ids is None or dataset.sample_id(index) in include_sample_ids)
        and (exclude_sample_ids is None or dataset.sample_id(index) not in exclude_sample_ids)
    ]
    loader_dataset = (
        dataset if len(selected_indices) == len(dataset) else Subset(dataset, selected_indices)
    )
    return DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_image(path: str | Path, image_size: int):
    with Image.open(path) as image:
        return image_transform(image_size)(image.convert("RGB")).unsqueeze(0)
