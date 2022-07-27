
import torch
import math
import warnings
import os

import torch.nn as nn

from typing import List
from torch import Tensor

from kaolin.metrics.pointcloud import chamfer_distance as history_chamfer_distance
from metric.emd.emd_module import emdFunction
from pytorch3d.loss import chamfer_distance


# -----------------------------------------------------------------------------------------
class EarthMoverDistance(nn.Module):

    def __init__(self, eps=0.005, iters=50):
        super().__init__()
        self.eps = eps
        self.iters = iters

    def forward(self, preds, gts, **kwargs):
        loss, _ = emdFunction.apply(preds, gts, self.eps, self.iters)
        if kwargs.get('radius') is not None:
            loss = loss / kwargs.get('radius').view(-1, 1)
        return torch.sum(loss)

# -----------------------------------------------------------------------------------------
class ChamferCUDA2(nn.Module):

    def forward(self, points1, points2):
        cost = history_chamfer_distance(points1, points2)
        return torch.sum(cost)

# # -----------------------------------------------------------------------------------------
class ChamferCUDA(nn.Module):

    def forward(self, xyz1: Tensor, xyz2: Tensor, nxyz1: Tensor = None, nxyz2: Tensor = None):
        return chamfer_distance(xyz1, xyz2, x_normals=nxyz1, y_normals=nxyz2, batch_reduction='mean', point_reduction='mean')


