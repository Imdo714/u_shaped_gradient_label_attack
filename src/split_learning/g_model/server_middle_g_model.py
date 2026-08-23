from torch import Tensor, nn

from ..architecture.cnn_blocks import conv_block


class ServerMiddleGModel(nn.Module):
    """Server-owned g model: transforms smashed data z into server output u.

    The forward API intentionally has no label argument.
    """

    def __init__(self, cut_config: str = "middle") -> None:
        super().__init__()
        if cut_config == "early":
            self.network = nn.Sequential(conv_block(16, 32), conv_block(32, 64))
        elif cut_config == "middle":
            self.network = conv_block(32, 64)
        elif cut_config == "late":
            self.network = conv_block(64, 64, pool=False)
        else:
            raise ValueError("cut_config must be early, middle, or late")
        self.cut_config = cut_config

    def forward(self, z: Tensor) -> Tensor:
        return self.network(z)


# Compatibility alias for imports used before the role-based package split.
ServerMiddle = ServerMiddleGModel

__all__ = ["ServerMiddleGModel", "ServerMiddle"]
