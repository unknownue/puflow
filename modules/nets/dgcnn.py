

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor


# -----------------------------------------------------------------------------------------
class DynamicGraphCNN(nn.Module):
    """
    Dynamic Graph CNN for Learning on Point Clouds.
    Code modified from https://github.com/AnTao97/dgcnn.pytorch.
    """

    def __init__(self, in_channel: int, k: int, emb_dim: int, output='pointwise'):
        super(DynamicGraphCNN, self).__init__()

        self.k = k
        self.output = output  # 'pointwise' or 'global'

        if output == 'pointwise':
            self.out_dim = emb_dim * 2 + 3 + 64 + 64 + 128 + 256
            self.fix_dim = 1
        elif output == 'miminal-pointwise':
            self.out_dim = emb_dim * 3
            self.fix_dim = 0
        elif output == 'global':
            self.out_dim = emb_dim * 2
            self.fix_dim = 0
        elif output == 'global-pointwise':
            self.out_dim = emb_dim
            self.fix_dim = 0
        else:
            self.out_dim = emb_dim
            self.fix_dim = 0
        self.out_dim += self.fix_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256 + self.fix_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(256 + self.fix_dim),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512 + self.fix_dim, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(negative_slope=0.2))

    def num_out_channel(self):
        return self.out_dim

    def forward(self, xyz: Tensor):
        xyz = xyz.transpose(1, 2).contiguous()
        B, _, N = xyz.shape

        x = DynamicGraphCNN.get_graph_feature(xyz, k=self.k)      # (B, 3, N) -> (B, 3 * 2, N, k)
        x = self.conv1(x)                       # (B, 3 * 2, N, k) -> (B, 64, N, k)
        x1, _ = x.max(dim=-1, keepdim=False)    # (B, 64, N, k) -> (B, 64, N)

        x = DynamicGraphCNN.get_graph_feature(x1, k=self.k)     # (B, 64, N) -> (B, 64 * 2, N, k)
        x = self.conv2(x)                       # (B, 64 * 2, N, k) -> (B, 64, N, k)
        x2, _ = x.max(dim=-1, keepdim=False)    # (B, 64, N, k) -> (B, 64, N)

        x = DynamicGraphCNN.get_graph_feature(x2, k=self.k)     # (B, 64, N) -> (B, 64 * 2, N, k)
        x = self.conv3(x)                       # (B, 64 * 2, N, k) -> (B, 128, N, k)
        x3, _ = x.max(dim=-1, keepdim=False)    # (B, 128, N, k) -> (B, 128, N)

        x = DynamicGraphCNN.get_graph_feature(x3, k=self.k)     # (B, 128, N) -> (B, 128 * 2, N, k)
        x = self.conv4(x)                       # (B, 128 * 2, N, k) -> (B, 256, N, k)
        x4, _ = x.max(dim=-1, keepdim=False)    # (B, 256, N, k) -> (B, 256, N)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 64 + 64 + 128 + 256, N)

        _x = self.conv5(x_f)   # (B, 64 + 64 + 128 + 256, N) -> (B, emb_dim, N)
        x1 = F.adaptive_max_pool1d(_x, 1).view(B, -1)   # (B, emb_dim, N) -> (B, emb_dim)

        if self.output == 'pointwise':
            x2 = F.adaptive_avg_pool1d(_x, 1).view(B, -1)   # (B, emb_dim, N) -> (B, emb_dim)
            _x = torch.cat([x1, x2], dim=1).unsqueeze(-1).repeat(1, 1, N)   # (B, emb_dim * 2, N)

            x = torch.cat([_x, xyz, x_f], dim=1)  # (B, emb_dim * 2 + 3 + 64 + 64 + 128 + 256, N)
            return x
        if self.output == 'global':
            x2 = F.adaptive_avg_pool1d(_x, 1).view(B, -1)   # (B, emb_dim, N) -> (B, emb_dim)
            x = torch.cat([x1, x2], dim=-1)
            x = x.unsqueeze(-1).repeat(1, 1, N)
            return x
        if self.output == 'miminal-pointwise':
            x2 = F.adaptive_avg_pool1d(_x, 1).view(B, -1)   # (B, emb_dim, N) -> (B, emb_dim)
            xt = torch.cat([x1, x2], dim=1).unsqueeze(-1).repeat(1, 1, N)   # (B, emb_dim * 2, N)
            x = torch.cat([xt, _x], dim=1)
            return x
        if self.output == 'global-pointwise':
            return x1, _x  # [B, emb_dim], [B, emb_dim, N]

        return x1

    @staticmethod
    def knn(x: Tensor, k: int):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
        return idx

    @staticmethod
    def get_graph_feature(x, k=20, idx=None, dim9=False):
        B = x.size(0)
        N = x.size(2)
        x = x.view(B, -1, N)
        if idx is None:
            if dim9 == False:
                idx = DynamicGraphCNN.knn(x, k=k)   # (B, N, k)
            else:
                idx = DynamicGraphCNN.knn(x[:, 6:], k=k)

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (B, N, num_dims)  -> (B*N, num_dims) #   B * N * k + range(0, B*N)
        feature = x.view(B * N, -1)[idx, :]
        feature = feature.view(B, N, k, num_dims) 
        x = x.view(B, N, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature  # (B, 2 * num_dims, N, k)

