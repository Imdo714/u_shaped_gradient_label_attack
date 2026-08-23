from torch import Tensor, nn


class ClientTailHModel(nn.Module):
    """Client-owned h model: transforms server output u into class logits.

    The true label is used outside this module by the client's loss function.
    The server-owned g model therefore never receives a label argument.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, num_classes),
        )

    def forward(self, u: Tensor) -> Tensor:
        """Return unnormalized class logits for server output u."""
        return self.network(u)


# Compatibility alias for imports used before the role-based package split.
ClientTail = ClientTailHModel

__all__ = ["ClientTailHModel", "ClientTail"]
