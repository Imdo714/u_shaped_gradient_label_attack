from __future__ import annotations

import numpy as np
import torch

from ...split_learning.architecture.split_learning_model import SplitLearningModel


def extract_inference_smashed_feature(
    model: SplitLearningModel,
    image: torch.Tensor,
) -> np.ndarray:
    """Inference mode uses smashed data; it never fabricates a loss gradient."""
    model.eval()
    with torch.no_grad():
        smashed = model.f_model(image).cpu().numpy().reshape(-1)
    return smashed / (np.linalg.norm(smashed) + 1e-12)
