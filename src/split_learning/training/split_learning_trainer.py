from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..architecture.split_learning_model import SplitLearningModel
from ..gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
    run_gradient_exchange_step,
)
from ..logging.gradient_transcript_logger import (
    EvaluatorGroundTruthLogger,
    ServerGradientTranscriptLogger,
)


class SplitLearningTrainer:
    def __init__(
        self,
        model: SplitLearningModel,
        device: torch.device,
        learning_rate: float,
        checkpoint_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_f = torch.optim.Adam(model.f_model.parameters(), lr=learning_rate)
        self.optimizer_g = torch.optim.Adam(model.g_model.parameters(), lr=learning_rate)
        self.optimizer_h = torch.optim.Adam(model.h_model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        debug_samples: int = 3,
        debug_values: int = 8,
    ) -> dict[str, float]:
        """한 epoch을 학습하고, 요청한 샘플 수만큼 상세 통신값을 출력한다.

        debug_samples=0이면 상세 로그를 끈다. 로그가 너무 길어지는 것을 막기
        위해 기본값은 첫 epoch의 3개 샘플이며 각 텐서의 앞 8개 값만 표시한다.
        """
        self.model.train()
        total_loss = correct = count = 0
        debugged_samples = 0
        for batch_id, (images, labels, sample_ids) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)
            should_debug = epoch == 1 and debugged_samples < debug_samples
            result = run_gradient_exchange_step(
                self.model,
                images,
                labels,
                self.criterion,
                self.optimizer_f,
                self.optimizer_g,
                self.optimizer_h,
                update=True,
                debug=should_debug,
                debug_max_values=debug_values,
                debug_sample_ids=[str(sample_id) for sample_id in sample_ids],
                debug_epoch=epoch,
                debug_batch_id=batch_id,
            )
            if should_debug:
                debugged_samples += images.size(0)
            total_loss += result.loss * images.size(0)
            correct += int((result.logits.argmax(1) == labels).sum())
            count += images.size(0)
        return {"loss": total_loss / max(count, 1), "accuracy": correct / max(count, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = correct = count = 0
        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model.predict(images)
            total_loss += float(self.criterion(logits, labels)) * images.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            count += images.size(0)
        return {"loss": total_loss / max(count, 1), "accuracy": correct / max(count, 1)}

    def collect_frozen_transcript(
        self,
        loader: DataLoader,
        epoch: int,
        server_logger: ServerGradientTranscriptLogger,
        evaluator_logger: EvaluatorGroundTruthLogger,
        max_samples: int | None = None,
    ) -> int:
        """Collect per-sample legitimate loss gradients at a single frozen model state."""
        self.model.eval()
        collected = 0
        for batch_id, (images, labels, sample_ids) in enumerate(loader):
            for offset in range(images.size(0)):
                if max_samples is not None and collected >= max_samples:
                    return collected
                image = images[offset : offset + 1].to(self.device)
                client_label = labels[offset : offset + 1].to(self.device)
                result = observe_frozen_gradient_exchange(
                    self.model,
                    image,
                    client_label,
                    self.criterion,
                )
                sample_id = str(sample_ids[offset])
                server_logger.log(
                    sample_id,
                    epoch,
                    batch_id,
                    result.smashed_z[0],
                    result.server_output_u[0],
                    result.grad_h_to_g[0],
                    result.grad_g_to_f[0],
                )
                # Ground truth is written through a separate evaluator object/file.
                evaluator_logger.log(sample_id, epoch, int(client_label.item()))
                collected += 1
        return collected

    def save_checkpoint(self, epoch: int, **metadata: object) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        self.model.save(path, epoch=epoch, **metadata)
        self.model.save(self.checkpoint_dir / "model.pt", epoch=epoch, **metadata)
        return path
