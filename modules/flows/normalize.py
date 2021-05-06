
import torch
import torch.nn as nn

from torch import Tensor


# -----------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """Yet, another ActNorm implementation for Point Cloud."""

    def __init__(self, channel: int):
        super(ActNorm, self).__init__()

        self.logs = nn.Parameter(torch.zeros((1, channel, 1)))  # log sigma
        self.bias = nn.Parameter(torch.zeros((1, channel, 1)))
        self.eps = 1e-6
        self.is_inited = False

    def forward(self, x: Tensor):
        """
        x: [B, C, N]
        """
        if not self.is_inited:
            self.__initialize(x)

        z = x * torch.exp(self.logs) + self.bias
        # z = (x - self.bias) * torch.exp(-self.logs)
        logdet = x.shape[2] * torch.sum(self.logs)
        return z, logdet

    def inverse(self, z: Tensor):
        # x = z * torch.exp(self.logs) + self.bias
        x = (z - self.bias) * torch.exp(-self.logs)
        return x

    def __initialize(self, x: Tensor):
        with torch.no_grad():
            bias = -torch.mean(x.detach(), dim=[0, 2], keepdim=True)
            logs = -torch.log(torch.std(x.detach(), dim=[0, 2], keepdim=True) + self.eps)
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_inited = True

