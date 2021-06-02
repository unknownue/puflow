
import torch
import torch.nn as nn
import warnings

from torch import Tensor

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from knn_cuda import KNN

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from modules.utils.fps import farthest_point_sampling as torch_fps, index_points as torch_pindex



# -----------------------------------------------------------------------------------------
class PatchHelper(object):

    def __init__(self, npoint_patch: int, patch_expand_ratio: float, extract='knn'):
        """
        npoint_patch: number of point in each patch
        patch_expand_ratio: TODO
        extract: only support knn now
        """
        super(PatchHelper, self).__init__()

        self.__npoint_patch  = npoint_patch
        self.__patch_expand_ratio = patch_expand_ratio

        if extract == 'knn':
            self.__extract_func_name__ = 'extract_knn_patch'
            self.knn = KNN(k=self.__npoint_patch, transpose_mode=False)

    def upsample(self, upsampler: nn.Module, pc: Tensor, npoint: int, upratio=None, jitter=False, **kwargs):
        """
        Upsample given point cloud in patches, and sample given number of point from it.
        params:
            upsampler: the network used to upsampling point patch
            permuter : help to permute point before upsampling
            pc: the point cloud waited to be upsampled, in [B, N, 3]
            npoint: the number of output point of each point cloud
            jitter: whether to apply jitter before upsampling
        """
        B, N, C = pc.shape
        extract_func = getattr(self, self.__extract_func_name__)

        pc, g_centroid, g_furthest_distance = PatchHelper.normalize_pc(pc)
        if jitter:
            pc = PatchHelper.jitter_perturbation_point_cloud(pc)

        patches = []

        patches = extract_func(pc, self.knn, self.__npoint_patch, self.__patch_expand_ratio)  # [B, n_patch, k1, 3]
        patches = patches.flatten(0, 1)  # [B, N, 3]
        patches = patches.reshape(B, -1, self.__npoint_patch, C)  # [B, n_patch, k1, 3]


        # if patches.shape[1] < 300:
        #     predict_patches = PatchHelper.__upsampling_patches(upsampler, patches, upratio, **kwargs)
        # else:
        #     predict_patches = []
        #     start, total = 0, patches.shape[1]
        #     while start < total:
        #         end = min(start + 300, total)
        #         partical_patch = PatchHelper.__upsampling_patches(upsampler, patches[:, start: end], upratio, **kwargs)
        #         predict_patches.append(partical_patch)
        #         start = start + 300
        #     predict_patches = torch.cat(predict_patches, dim=1)
        # [B, n_patch * self.__duplicate_patch, k1 * upratio, 3]
        predict_patches = PatchHelper.__upsampling_patches(upsampler, patches, upratio, **kwargs)

        predict_pc = PatchHelper.merge_patches(predict_patches, npoint)

        predict_pc = predict_pc * g_furthest_distance + g_centroid.transpose(1, 2)
        predict_pc = predict_pc.transpose(1, 2).contiguous()

        # predict_pc = torch.cat([pc, predict_pc], dim=1)
        return predict_pc  # [B, npoint, 3]

    @staticmethod
    def __upsampling_patches(upsampler: nn.Module, patches: Tensor, upratio=None, **kwargs):

        B, n_patch, k1, C = patches.shape
        patches = patches.reshape(B * n_patch, k1, C)
        patches, centroids, furthest_distance = PatchHelper.normalize_pc(patches)

        predict_patches = upsampler.sample(patches, upratio=(upratio or 4), **kwargs)  # [B * n_patch, k2, C]
        predict_patches = torch.cat([predict_patches, patches], dim=1)
        predict_patches = predict_patches * furthest_distance + centroids

        predict_patches = predict_patches.reshape(B, n_patch, -1, C)
        return predict_patches

    @staticmethod
    def __extract_idx_patches(pc: Tensor, knn_searcher: KNN, npoint_patch: int, expand_ratio: float, seed_centroids_idx=None) -> Tensor:
        _, N, _ = pc.shape
        pc_T = pc.transpose(1, 2).contiguous()  # [B, C, N]

        if seed_centroids_idx is None:
            n_patch = int(N / npoint_patch * expand_ratio)
            patch_centroids_idx = furthest_point_sample(pc, n_patch)  # [B, n_patch]
        else:
            _, n_patch = seed_centroids_idx.shape
            patch_centroids_idx = seed_centroids_idx
        patch_centroids = gather_operation(pc_T, patch_centroids_idx)  # [B, C, n_patch]
        _, idx_patches = knn_searcher(pc_T, patch_centroids)  # [B, k, n_patch]

        return idx_patches, n_patch

    @staticmethod
    def extract_knn_patch(pc: Tensor, knn_searcher: KNN, npoint_patch: int, expand_ratio: float, seed_centroids_idx=None) -> Tensor:
        """
        Extract patches from point clouds by KNN.
            (Only work for 3D points)
        pc: Initail Point Cloud, in [B, N, C]
        """
        B, _, C = pc.shape
        idx_b = torch.arange(B).view(-1, 1)

        idx_patches, n_patch = PatchHelper.__extract_idx_patches(pc, knn_searcher, npoint_patch, expand_ratio, seed_centroids_idx)  # [B, k, n_patch]
        idx_patches = idx_patches.transpose(1, 2).flatten(start_dim=1)  # [B, n_patch * k]

        patches = pc[idx_b, idx_patches]  # [B, n_patch * k, C]
        return patches.reshape(B, n_patch, npoint_patch, C)  # [B, n_patch, k, C]

    @staticmethod
    def fps(pc: Tensor, n_point: int, transpose=True):
        """
        pc: [B, N, C]
        n_point: number of output point
        """
        centroids_idx = furthest_point_sample(pc, n_point)  # [B, n_patch]
        centroids = gather_operation(pc.transpose(1, 2).contiguous(), centroids_idx)  # [B, C, n_patch]

        if transpose is True:
            return centroids.transpose(1, 2).contiguous()
        else:
            return centroids

    @staticmethod
    def merge_patches(patches: Tensor, npoint: int, origins: Tensor=None):
        """
        patches: input patches, in [B, n_patch, k, 3]
        npoint: number of final points in each point cloud
        origins: Optional point cloud, in [B, N, 3]
        """
        B, _, _, C = patches.shape
        if origins is None:
            patches = patches.reshape(B, -1, C).contiguous()  # [B, n_patch * k, 3]
        else:
            patches = patches.reshape(B, -1, C)
            patches = torch.cat([patches, origins], dim=1).contiguous()
        patches_T = patches.transpose(1, 2).contiguous()

        idx_predict_pc = furthest_point_sample(patches, npoint)
        final_pc = gather_operation(patches_T, idx_predict_pc)
        return final_pc  # [B, 3, npoint]
        return patches_T
    
    @staticmethod
    def merge_pc(pc1: Tensor, pc2: Tensor, npoint: int):
        tmp = torch.cat([pc1, pc2], dim=1)
        idx_fps_seed = torch_fps(tmp, npoint)
        return torch_pindex(tmp, idx_fps_seed)

    @staticmethod
    def normalize_pc(pc: Tensor):
        """
        Normalized point cloud in range [-1, 1].
        pc: [B, N, 3]
        """
        centroid = torch.mean(pc, dim=1, keepdim=True)  # [B, 1, 3]
        pc = pc - centroid  # [B, N, 3]
        dist_square = torch.sum(pc ** 2, dim=-1, keepdim=True).sqrt()  # [B, N, 1]
        furthest_distance, _ = torch.max(dist_square, dim=1, keepdim=True)  # [B, 1, 1]
        pc = pc / furthest_distance
        return pc, centroid, furthest_distance

    @staticmethod
    def jitter_perturbation_point_cloud(pc, sigma=0.010, clip=0.020):
        """
        Randomly jitter points. jittering is per point.
        Input : Original point clouds, in [B, N, 3]
        Return: Jittered point clouds, in [B, N, 3]
        """
        if sigma > 0:
            B, N, C = pc.shape
            assert(clip > 0)
            jittered_data = torch.clamp(sigma * torch.randn(B, N, C, device=pc.device), -1 * clip, clip)
            jittered_data[:, :, 3:] = 0
            jittered_data += pc
            return jittered_data
        else:
            return pc

    @staticmethod
    def remove_outliers(sr: Tensor, lr: Tensor, num_outliers: int) -> Tensor:
        from metric.PyTorchCD.chamfer3D import dist_chamfer_3D
        chamferLoss = dist_chamfer_3D.chamfer_3DDist()

        (B, N, _), device = sr.shape, sr.device
        dist1, dist2, _, _ = chamferLoss(sr, lr)    # [B, N1], [B, N2]
        idx_dist1 = torch.argsort(dist1, dim=-1, descending=True)  # [B, N1]
        idx_outliers = idx_dist1[:, :num_outliers]

        # Get inverse idx
        idxb = torch.arange(B).view(-1, 1)
        inv = torch.ones((B, N)).int().to(device)
        inv[idxb, idx_outliers] = 0
        inv = torch.nonzero(inv, as_tuple=False)[:, 1]
        idx_inverse = inv.view(B, N - num_outliers)

        return sr[idxb, idx_inverse]
# -----------------------------------------------------------------------------------------
