
# Code modified from https://github.com/UncleMEDM/PUGAN-pytorch

import torch
import math
import warnings
import os

import torch.nn as nn

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from knn_cuda import KNN

from typing import List

from metric.PyTorchEMD.emd import earth_mover_distance

from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample, ball_query, grouping_operation

from contextlib import redirect_stdout
with open(os.devnull, "w") as outer_space, redirect_stdout(outer_space):
    from metric.PyTorchCD.chamfer3D import dist_chamfer_3D


# -----------------------------------------------------------------------------------------
class EMDLoss(nn.Module):

    def __init__(self):
        super(EMDLoss, self).__init__()

        self.metric = earth_mover_distance

    def forward(self, pred, gt, radius=1.0):
        """
        pred: [B, N, 3]
        gt  : [B, N, 3]
        """
        _, N, _ = gt.shape

        emd_loss = self.metric(pred, gt, transpose=False) / radius / float(N)
        return torch.sum(emd_loss)



# -----------------------------------------------------------------------------------------
class UniformLoss(nn.Module):
    """
    See also https://github.com/liruihui/PU-GAN/issues/18
    https://github.com/liruihui/PU-GAN/blob/69cf7d7f956c7c31f4a6d9213d8b11a658c52b9c/Common/loss_utils.py#L110-L139
    """

    def __init__(self):
        super(UniformLoss, self).__init__()

        self.knn_uniform = KNN(k=2, transpose_mode=True)

    def forward(self, pcd, percentage=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
        B, N, C = pcd.shape
        pcd_T = pcd.permute(0, 2, 1).contiguous()

        npoint = int(N * 0.05)
        loss = 0
        further_point_idx = furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = gather_operation(pcd_T, further_point_idx)  # [B, C, N]
        for p in percentage:
            nsample = int(N * p)
            r = math.sqrt(p * radius)  # r = torch.sqrt(p * radius)
            disk_area = math.pi * (radius ** 2) / N

            idx = ball_query(r, nsample, pcd.contiguous(), new_xyz.permute(0, 2, 1).contiguous()) # [b, npoint, nsample]

            # expect_len = math.sqrt(2 * disk_area / 1.732)  # using hexagon
            expect_len = math.sqrt(disk_area)  # using square

            grouped_pcd = grouping_operation(pcd_T, idx)  # [B, C, npoint, nsample]
            grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # [B, npoint, nsample, C]

            grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)  # [B * npoint, nsample, C]

            dist, _ = self.knn_uniform(grouped_pcd, grouped_pcd)  # [B * npoint, nsample, k]
            # dist, _ = self.knn(grouped_pcd, k=2)  # [B * npoint, nsample, k]
            # print(dist.shape)
            uniform_dist = dist[:, :, 1:]  # [B * N, nsample, k - 1]
            uniform_dist = torch.abs(uniform_dist + 1e-8)
            uniform_dist = torch.mean(uniform_dist, dim=1)  # [B * N, k - 1]
            uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + 1e-8)
            mean_loss = torch.mean(uniform_dist)
            mean_loss = mean_loss * math.pow(p * 100, 2)
            loss += mean_loss
        
        loss = loss / len(percentage)
        loss = loss * B  # Multiply B to remove the inference of batch size
        return loss
    

# # -----------------------------------------------------------------------------------------
class ChamferCUDA(nn.Module):
    """From https://github.com/ThibaultGROUEIX/ChamferDistancePytorch"""

    def __init__(self):
        super(ChamferCUDA, self).__init__()
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, points1, points2, avg_scale=0.5):
        dist1, dist2, _, _ = self.chamLoss(points1, points2)
        cost = (torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)) * avg_scale
        return torch.sum(cost)

# -----------------------------------------------------------------------------------------
class HausdorffLoss(nn.Module):

    def __init__(self):
        super(HausdorffLoss, self).__init__()
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, points1, points2):
        dist1, dist2, _, _ = self.chamLoss(points1, points2)
        (m1, _), (m2, _) = torch.max(dist1, dim=-1, keepdim=True), torch.max(dist2, dim=-1, keepdim=True)
        m, _ = torch.max(torch.cat([m1, m2], dim=-1), dim=-1)
        return torch.sum(m)


# -----------------------------------------------------------------------------------------
class UpsampingMetrics(object):

    def __init__(self, metrics: List[str]):
        super(UpsampingMetrics, self).__init__()

        self.cd_loss          = ChamferCUDA()   if 'CD'             in metrics else None
        self.emd_loss         = EMDLoss()       if 'EMD'            in metrics else None
        self.hausdorff_loss   = HausdorffLoss() if 'HD'             in metrics else None
        self.uniform_loss     = UniformLoss()   if 'Uniform'        in metrics else None

    def evaluate(self, pred, gt, pcd_radius=None, nor_pred=None, nor_gt=None):
        if pcd_radius is None:
            pcd_radius = torch.ones([pred.shape[0],], device=pred.device)

        result = {}
        if self.cd_loss is not None:
            result['CD'] = self.cd_loss(pred, gt)
        if self.emd_loss is not None:
            result['EMD'] = self.emd_loss(pred, gt, pcd_radius)
        if self.hausdorff_loss is not None:
            result['HD'] = self.hausdorff_loss(pred, gt)
        if self.uniform_loss is not None:
            # TODO: radius in tensor is not support
            result['Uniform'] = self.uniform_loss(pred, percentage=[0.004, 0.008, 0.012], radius=1.0)

        return result
# -----------------------------------------------------------------------------------------
