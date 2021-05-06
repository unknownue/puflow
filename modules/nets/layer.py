
import torch
import torch.nn as nn
import math

from torch.nn import functional as F
from torch import Tensor


# ---------------------------------------------------------------------
class ResidualNet1D(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_channel: int, hidden_channel, out_channel: int, num_blocks=2, u_channel=None, dropout=None, bn=False, is_zero_initialization=True):
        super(ResidualNet1D, self).__init__()

        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channel

        if u_channel is None:
            self.initial_layer = nn.Linear(in_channel, hidden_channel)
        else:
            self.initial_layer = nn.Linear(in_channel + u_channel, hidden_channel)
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            bn1      = nn.BatchNorm1d(hidden_channel, eps=1e-3) if bn else None
            bn2      = nn.BatchNorm1d(hidden_channel, eps=1e-3) if bn else None
            ac1      = nn.LeakyReLU(inplace=True)
            ac2      = nn.LeakyReLU(inplace=True)
            linear_1 = nn.Linear(hidden_channel, hidden_channel)
            linear_2 = nn.Linear(hidden_channel, hidden_channel)
            dp       = nn.Dropout(p=dropout) if dropout is not None else None

            layers = filter(lambda l: l is not None, [bn1, ac1, linear_1, bn2, ac2, dp, linear_2])
            self.residual_blocks.append(nn.Sequential(*layers))

        if u_channel is not None:
            self.u_linears = nn.ModuleList([
                nn.Linear(u_channel, hidden_channel)
                for _ in range(num_blocks)
            ])
            # self.mix_linears = nn.ModuleList([nn.Sequential(
            #     nn.Linear(hidden_channel + u_channel, hidden_channel),
            #     nn.Tanh(),
            #     nn.Linear(hidden_channel, hidden_channel)
            # ) for _ in range(num_blocks)])

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channel, out_channel),
            nn.Tanh(),
        )

        if is_zero_initialization:
            for blocks in self.residual_blocks:
                nn.init.uniform_(blocks[-1].weight, -1e-3, 1e-3)
                nn.init.uniform_(blocks[-1].bias,   -1e-3, 1e-3)
                # nn.init.zeros_(blocks[-1].weight)
                # nn.init.zeros_(blocks[-1].bias)
    
    def forward(self, h: Tensor, context: Tensor=None):
        """
        h      : [B, C1, N]
        context: [B, C2, N]
        """

        if context is None:
            h = h.transpose(1, 2)
            h = self.initial_layer(h)
        else:
            h_c = torch.cat([h, context], dim=1)
            h = self.initial_layer(h_c.transpose(1, 2))

        for i in range(self.num_blocks):
            residual = self.residual_blocks[i]
            t_h = residual(h)

            if context is not None:
                c = self.u_linears[i](context.transpose(1, 2))
                t_h = F.glu(torch.cat([t_h, c], dim=-1), dim=-1)

                # t_h = self.mix_linears[i](torch.cat([t_h, context.transpose(1, 2)], dim=-1))

            h = h + t_h

        h = self.final_layer(h)
        h = h.transpose(1, 2)
        return h


# ---------------------------------------------------------------------
class SoftClampling(nn.Module):
    """
    From https://github.com/VLL-HD/FrEIA/blob/a5069018382d3bef25a6f7fa5a51c810b9f66dc5/FrEIA/modules/coupling_layers.py#L88
    """

    def __init__(self, is_enable=True, clamp=1.9):
        super(SoftClampling, self).__init__()

        self.is_enable = is_enable
        if is_enable:
            self.clamp = 2.0 * clamp / math.pi
        else:
            self.clamp = None

    def forward(self, scale: Tensor):
        if self.is_enable:
            return self.clamp * torch.atan(scale)
        else:
            return scale


# ---------------------------------------------------------------------
class Swish(nn.Module):
    """A common activation function not included in PyTorch"""
    def forward(self, x):
        return x * torch.sigmoid(x)

