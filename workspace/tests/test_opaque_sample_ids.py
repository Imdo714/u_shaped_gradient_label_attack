import hashlib
from pathlib import Path

from PIL import Image

from src.shared.data.image_dataset import ImageFolderWithID


def test_attacker_sample_id_does_not_contain_class_name(tmp_path):
    # This test uses pytest's isolated temporary storage and does not create a
    # research dataset in the project tree.
    class_dir = tmp_path / "cat"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="red").save(class_dir / "obvious_cat_name.jpg")
    dataset = ImageFolderWithID(tmp_path)
    _, _, sample_id = dataset[0]
    expected = hashlib.sha256("cat/obvious_cat_name.jpg".encode("utf-8")).hexdigest()[:20]
    assert sample_id == f"sample_{expected}"
    assert "cat" not in sample_id
