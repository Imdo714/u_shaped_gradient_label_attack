from torch import Tensor, nn

from ..architecture.cnn_blocks import conv_block


class ClientFrontFModel(nn.Module):
    """Client-owned f model: transforms input image x into smashed data z."""

    def __init__(self, cut_config: str = "middle") -> None:
        super().__init__()
        if cut_config not in {"early", "middle", "late"}:
            raise ValueError("cut_config must be early, middle, or late")
        blocks = [conv_block(3, 16)]
        if cut_config in {"middle", "late"}:
            blocks.append(conv_block(16, 32))
        if cut_config == "late":
            blocks.append(conv_block(32, 64))
        self.network = nn.Sequential(*blocks)
        self.cut_config = cut_config

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


# Compatibility alias for imports used before the role-based package split.
ClientFront = ClientFrontFModel

__all__ = ["ClientFrontFModel", "ClientFront"]
