
import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.getcwd())

from torch import Tensor
from modules.utils.fps import square_distance


# -----------------------------------------------------------------------------------------
def permute_by_grid(pts: Tensor, grid_permute: str='distance', n_grid=None, is_return_idx=False):
    """
    Premute point order by grid. The input point values should be in range [-0.5, 0.5]
    pts: Input point cloud, [B, N, 3], or [B, N, 2]
    grid_permute: 'distance' or 'nearest'
    """

    B, N, C = pts.shape
    device = pts.device

    INIT_AXIS = 0
    if C == 3:
        GRID_SIZE = 32 if n_grid is None else n_grid
        GRID_COUNT = int(GRID_SIZE * GRID_SIZE * GRID_SIZE)
        v_min_pts = torch.min(pts)
        v_max_pts = torch.max(pts)
        assert v_min_pts >= -0.5 and v_max_pts <= 0.5

        pts_floor = ((pts + 0.5) * float(GRID_SIZE)).floor()
        min_p = torch.clamp(pts_floor, 0.0, float(GRID_SIZE)).long()

        # the grid index of each point
        idx_pts_grid = min_p[:, :, 2] * (GRID_SIZE ** 2) + min_p[:, :, 1] * GRID_SIZE + min_p[:, :, 0]  # [B, N]
    else:  # C == 2
        # For MNIST
        GRID_SIZE = 28 if n_grid is None else n_grid
        GRID_COUNT = int(GRID_SIZE * GRID_SIZE)
        pts_floor = ((pts + 1.0) / 2.0 * float(GRID_SIZE)).floor()
        min_p = torch.clamp(pts_floor, 0.0, float(GRID_SIZE)).long()

        # the grid index of each point
        idx_pts_grid = min_p[:, :, 1] * GRID_SIZE + min_p[:, :, 0]  # [B, N]

    # Get the index of cell that contains any points
    _N = torch.zeros((B, GRID_COUNT)).to(device)
    add_ones = torch.ones((B, N), dtype=torch.float32).to(device)
    idxb = (torch.arange(B) * GRID_COUNT).view(-1, 1).to(device)
    _N.put_(idx_pts_grid + idxb, add_ones, accumulate=True)  # [B, GRID_COUNT]
    grid_batch, grid_idx = torch.nonzero(_N, as_tuple=True)  # [N1?,] [N2?]

    # Get the center point of grids
    x = (grid_idx  % GRID_SIZE).float()  # [N2?,]
    y = (grid_idx // GRID_SIZE).float()  # [N2?,]
    z = (grid_idx // (GRID_SIZE ** 2)).float()  # [N2?,]
    centers = torch.stack([x + 0.5, y + 0.5, z + 0.5], dim=-1)  # [N2?, C]
    _, batch_chunks_size = torch.unique(grid_batch, sorted=True, return_counts=True)
    bth_centers  = torch.split(centers,  batch_chunks_size.tolist())  # list of [N?, C]
    bth_grid_idx = torch.split(grid_idx, batch_chunks_size.tolist())  # list of [N?,]

    # Sort the grids by distance
    if grid_permute == 'distance':
        ascending_method = distance_ascending
    elif grid_permute == 'nearest':
        ascending_method = nearest_ascending
    else:
        print('Unknown grid permute algorithm'); exit()

    grids_merge_order = []
    for i in range(B):
        batch = bth_centers[i]
        fill_grid_idx = bth_grid_idx[i]
        _, axis_min_grid = torch.min(batch[:, INIT_AXIS], dim=0)
        axis_min_grid_idx = axis_min_grid.cpu().item()

        grid_idx_sorted = ascending_method(batch, axis_min_grid_idx)
        grids_merge_order.append(fill_grid_idx[grid_idx_sorted])

    # Get index of points by the grid order
    pts_indices = magic_argsort(idx_pts_grid, grids_merge_order)  # [B, N]

    # Reorder the points
    if is_return_idx:
        return pts_indices
    else:
        return pts[torch.arange(B).view(-1, 1), pts_indices]


def distance_ascending(pts: Tensor, compare_idx: int):
    N, _ = pts.shape
    compare_pt = pts[compare_idx].unsqueeze(0).repeat(N, 1)  # [N, C]
    distances = torch.sum((pts - compare_pt) ** 2, dim=-1, keepdim=False)  # [N,]
    return torch.argsort(distances, dim=0)

def nearest_ascending(pts: torch.Tensor, compare_idx: int):

    min_idx = compare_idx
    permute_orders = [compare_idx]
    indices = torch.arange(pts.shape[0])

    for _ in range(len(pts) - 1):
        N, _ = pts.shape
        compare_pt = pts[min_idx].unsqueeze(0).repeat(N - 1, 1)       # [N - 1, C]
        pts = torch.cat([pts[:min_idx], pts[(min_idx + 1):]], dim=0)  # [N - 1, C]
        indices = torch.cat([indices[:min_idx], indices[(min_idx + 1):]], dim=0)

        distances = torch.sum((pts - compare_pt) ** 2, dim=-1, keepdim=False)  # [N - 1,]
        _, min_idx = torch.min(distances, dim=0)
        min_idx = min_idx.item()
        permute_orders.append(indices[min_idx])
    return torch.tensor(permute_orders, dtype=torch.long)


def magic_argsort(input: Tensor, comparation: Tensor):
    """
    See also https://discuss.pytorch.org/t/how-to-implement-a-more-general-argsort/

    input: 2D tensor in long, [B, N]
    comparation: comparation order, array like of 1D tensor
    """
    sorted_indices = []
    for d, c in zip(input, comparation):
        mask = d.unsqueeze(1) == c
        _, inds = torch.nonzero(mask, as_tuple=True)
        sorted_indices.append(torch.argsort(inds))
    return torch.stack(sorted_indices)


# -----------------------------------------------------------------------------------------
def permutebyfolding(pts_source: Tensor, foldingnet: nn.Module) -> Tensor:
    """
    Construct reference point by foldingnet.
    Then permute source points by nearest point of reference points.
    """

    B, N, _ = pts_source.shape  # [B, N1, C]
    device = pts_source.device
    idxb = torch.arange(B).view(-1, 1)
    
    pts_reference = foldingnet(pts_source)  # [B, C, N2]
    pts_reference = pts_reference.transpose(1, 2).contiguous()  # [B, N2, C]
    dist = square_distance(pts_source, pts_reference)  # [B, N1, N2]
    nearest_idx = torch.argmin(dist, dim=-1, keepdim=False)  # [B, N1]
    
    sorted_order = torch.argsort(nearest_idx, dim=1)  # [B, N1]
    sorted_idx = torch.empty_like(sorted_order)       # [B, N1]
    sorted_idx[idxb, sorted_order] = torch.arange(N).unsqueeze(0).repeat(B, 1).to(device)

    return pts_source[idxb, sorted_idx]


# -----------------------------------------------------------------------------------------
def permutebymatching(lr: Tensor, sr: Tensor, k: int, n_grid=3, sorted=False, is_return_idx=False) -> Tensor:
    """
    Permute high resolution point by matching low resolution points.
    Low resolution points are permute by permute_by_grid methods.
    params:
        lr: [B, N1, C]
        sr: [B, N2, C]
    """
    B, N1, _ = lr.shape

    lr = lr * 0.5  # values in range [-0.5, 0.5], this is needed for permute_by_grid
    lr = permute_by_grid(lr, grid_permute='nearest', n_grid=n_grid)
    lr = lr * 2.0  # scale back range to [-1, 1]

    dist = square_distance(lr, sr)  # [B, N1, N2]
    _, nearest_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=sorted)  # [B, N1, k]
    nearest_idx = nearest_idx.reshape(B, N1 * k)

    idx_b = torch.arange(B).view(-1, 1)
    new_sr = sr[idx_b, nearest_idx]  # [B, N1 * k, 3]

    if is_return_idx:
        nearest_idx = torch.argsort(dist, dim=-1, descending=False)

        return lr, new_sr, nearest_idx
    else:
        return lr, new_sr


# -----------------------------------------------------------------------------------------
def permutebymatching2(lr: Tensor, sr: Tensor, k: int, n_grid=3, sorted=False, is_return_idx=False):
    """
    Does almost the same as permutebymatching, but returning the permute indices of lr
    """
    B, N1, _ = lr.shape
    idx_b = torch.arange(B).view(-1, 1)

    lr = lr * 0.5  # values in range [-0.5, 0.5], this is needed for permute_by_grid
    __idx_lr = permute_by_grid(lr, grid_permute='nearest', n_grid=n_grid, is_return_idx=True)
    lr = lr[idx_b, __idx_lr]
    lr = lr * 2.0  # scale back range to [-1, 1]

    dist = square_distance(lr, sr)  # [B, N1, N2]
    _, nearest_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=sorted)  # [B, N1, k]
    nearest_idx = nearest_idx.reshape(B, N1 * k)

    new_sr = sr[idx_b, nearest_idx]  # [B, N1 * k, 3]

    if is_return_idx:
        nearest_idx = torch.argsort(dist, dim=-1, descending=False)

        return lr, __idx_lr, new_sr, nearest_idx
    else:
        return lr, __idx_lr, new_sr

# -----------------------------------------------------------------------------------------
def lr_hr_matching(lr: Tensor, sr: Tensor, k: int):
    dist = square_distance(lr, sr)  # [B, N1, N2]
    _, nearest_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)  # [B, N1, k]
    return nearest_idx


# -----------------------------------------------------------------------------------------
class PermutateHelper:
    """A wrapper for all point permutation methods"""

    def __init__(self):
        self.mode = None

    def permutebygrid(self, methods: str, n_grid: int):
        assert methods in ['distance', 'nearest']
        self.mode = 'grid'
        self.grid_permute = methods
        self.n_grid       = n_grid
    
    def permutebyfolding(self, foldingnet_path: str, device):
        self.mode = 'folding'
        self.foldingnet = torch.load(foldingnet_path, map_location=device)

    def permute(self, pts: Tensor, scale=0.5):
        """
        pts: the point cloud to be permutated in point order
        scale: only need this for grid permutation, scale to fit permute_by_grid method need
        """
        if self.mode is None:
            return pts
        if self.mode == 'grid':
            pts = pts * scale
            pts = permute_by_grid(pts, self.grid_permute, n_grid=self.n_grid)
            return pts * (1 / scale)
        if self.mode == 'folding':
            return permutebyfolding(pts, self.foldingnet)


# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.getcwd())

#     from omegaconf import OmegaConf
#     from plotting.image import plot_pointcloud
#     from dataset.pcn import ShapeNetPCNDataModule
# 
#     cfg = OmegaConf.create({
#         'seed': 1085,
#         'dataset': {
#             'root': './data/ShapeNetCompletion/PCN/',
#             'batch_size': 8,
#             'num_worker': 0,
#             'classes': 'plane',
#             'is_gt_downsample': True,
#         },
#     })
# 
#     pcn = ShapeNetPCNDataModule(hparams=cfg)
#     train_loader = pcn.val_dataloader()
# 
#     for batch in train_loader:
#         partial, gt = batch
# 
#         pts = permute_by_grid(gt)
# 
#         plot_pointcloud(pts[5, 0::4], is_show=False, is_save=True, path='runs/PCN/inputs/permute1.png')
#         plot_pointcloud(pts[5, 1::4], is_show=False, is_save=True, path='runs/PCN/inputs/permute2.png')
#         plot_pointcloud(pts[5, 2::4], is_show=False, is_save=True, path='runs/PCN/inputs/permute3.png')
#         plot_pointcloud(pts[5, 3::4], is_show=False, is_save=True, path='runs/PCN/inputs/permute4.png')
#         plot_pointcloud(pts[5, :], is_show=False, is_save=True, path='runs/PCN/inputs/permute5.png')
#         break


    import matplotlib.pyplot as plt
    import h5py as h5

    with h5.File('./data/mnist2d-order-pointcloud.h5', 'r') as h5f:

        train_data = h5f['train_data']  # [B, SAMPLE_NUM_POINTS, 2]
        first_p = [200, 600, 1000, 1400, 2048]
        count = 5
        start = 100
        fig = plt.figure(figsize=(count * 10, 8 * len(first_p)))
        fig.canvas.set_window_title(f'MNIST 2D Point Cloud')

        for i in range(start, start + count):
            
            plot_idx = i
            pts = train_data[plot_idx]  # [SAMPLE_NUM_POINTS, 2]
            pts = torch.from_numpy(pts).unsqueeze(0)
            # pts = pts[:, torch.randperm(2048), :]
            pts = permute_by_grid(pts).squeeze().numpy()
            x, y = pts.transpose()

            for j in range(len(first_p)):
                x1 = x[:first_p[j]]
                y1 = y[:first_p[j]]

                ax = plt.subplot(len(first_p), count, i - start + 1 + j * count)
                scatter = ax.scatter(x1, y1)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                scatter.axes.invert_yaxis()

        # plt.savefig('runs/MNIST/inputs/unorder-permute.png')
        plt.savefig('runs/MNIST/inputs/order-permute.png')
