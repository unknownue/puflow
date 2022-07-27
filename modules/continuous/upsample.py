
import os
import sys
import torch
import numpy as np

from argparse import ArgumentParser
from typing import List
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from modules.utils.patch import PatchHelper
from modules.continuous.interpflow import PointInterpFlow


# -----------------------------------------------------------------------------------------
@torch.no_grad()
def upsampling(data_paths: List[str], target_path: str, checkpoint_path: str, up_ratio: int, num_outlier: int, num_patch: int, num_upsampling: int=None, seed=None):

    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda:0')

    network = PointInterpFlow(pc_channel=3)

    network.load_state_dict(torch.load(checkpoint_path))
    network.set_to_initialized_state()
    network = network.to(device)
    network.eval()
    patch_helper = PatchHelper(num_patch, patch_expand_ratio=4)

    for path in data_paths:
        # print(f'Upsampling points {path}...')
        _, file_name = os.path.split(path)

        pt_input = np.loadtxt(path, dtype=np.float32)
        pt_input = torch.from_numpy(pt_input).unsqueeze(0).to(device)
        pt_input = pt_input[:, torch.randperm(pt_input.shape[1])].contiguous()
        
        if num_upsampling is None: # the number of points after upsamling
            NUM_UPSAMPLING_POINTS = pt_input.shape[1] * up_ratio + (num_outlier or 0)
        else:
            NUM_UPSAMPLING_POINTS = num_upsampling + (num_outlier or 0)

        pred = patch_helper.upsample(network, pt_input, npoint=NUM_UPSAMPLING_POINTS, upratio=up_ratio, jitter=False)

        if num_outlier is not None and num_outlier > 0:
            pred = PatchHelper.remove_outliers(pred, pt_input, num_outlier)

        pred = pred.squeeze().cpu().numpy()
        np.savetxt(Path(target_path) / file_name, pred, fmt='%.6f')
    # print('Finish...!')



# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, help='Path of input directory')
    parser.add_argument('--target', type=str, help='Path of output directory')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--checkpoint', type=str, help='Path of checkpoint')
    parser.add_argument('--up_ratio', type=int, help='upsampling ratio', default=4)
    parser.add_argument('--num_patch', type=int, help='number of point in each patch', default=256)  # 192
    parser.add_argument('--num_out', type=int, default=None, help='number of point of output point cloud')

    args = parser.parse_args()

    if not os.path.exists(args.target):
        os.mkdir(args.target)

    data_paths = []
    for root, dirs, files in os.walk(args.source):
        data_paths.extend([os.path.join(root, f) for f in files if '.xyz' in f])

    upsampling(data_paths, args.target, args.checkpoint, up_ratio=args.up_ratio,
        num_outlier=24, num_patch=args.num_patch, num_upsampling=args.num_out, seed=args.seed)
