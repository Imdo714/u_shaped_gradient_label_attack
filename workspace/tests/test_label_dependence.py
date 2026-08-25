import torch

from src.experiments.analysis.same_image_different_label import label_gradient
from src.split_learning.architecture.split_learning_model import (
    build_split_learning_model,
)


def test_same_input_different_label_changes_gradient_not_logits():
    torch.manual_seed(11)
    model = build_split_learning_model("middle")
    model.eval()
    image = torch.randn(1, 3, 32, 32)
    _, logits_cat, grad_cat = label_gradient(model, image, 0)
    _, logits_dog, grad_dog = label_gradient(model, image, 1)
    assert torch.equal(logits_cat, logits_dog)
    assert not torch.allclose(grad_cat, grad_dog, rtol=1e-5, atol=1e-7)
    assert torch.linalg.vector_norm(grad_cat - grad_dog) > 1e-7
