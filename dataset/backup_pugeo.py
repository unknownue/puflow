
import numpy as np
import h5py as h5
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from omegaconf.dictconfig import DictConfig
from dataset.pugan import VisionAirPUGAN


# ---------------------------------------------------------------------------------------------
class SketchfabPUGeo(Dataset):
    """Sketchfab Point Cloud Dataset for PUGeo"""

    # -------------------------------------------------------
    def __init__(self, root, split, is_normalize=True, is_jitter=True, is_rotate=True, is_scale=False, seed=42, use_normal=False, verbose=False):
        super(SketchfabPUGeo, self).__init__()

        assert split in ['full', 'train', 'valid', 'test']

        self.root = Path(root)
        self.split = split
        self.is_jitter = is_jitter
        self.is_rotate = is_rotate
        self.is_scale  = is_scale
        self.is_use_normal = use_normal

        split_ratio = {
            'full' : [0.0, 1.0],
            'train': [0.0, 0.65],
            'valid': [0.65, 0.75],
            'test' : [0.75, 1.0],
        }

        if use_normal:
            self.xyz_sparse, self.xyz_dense, self.norm_sparse, self.norm_dense, self.pt_match_idx = self.read_h5_split(split_ratio, True, verbose)
        else:
            self.xyz_sparse, self.xyz_dense, self.pt_match_idx = self.read_h5_split(split_ratio, False, verbose)

        if is_normalize:
            self.xyz_sparse, self.xyz_dense = VisionAirPUGAN.normalize(self.xyz_sparse, self.xyz_dense, verbose)

    # -------------------------------------------------------
    def read_h5_split(self, split_ratio, is_use_normal, verbose):

        with h5.File(self.root, 'r') as h5f:
            start = int(70176 * split_ratio[self.split][0])
            end   = int(70176 * split_ratio[self.split][1])

            xyz_sparse = h5f['sparse_xyz'][start:end]  # [B, N1, 3]
            xyz_dense  = h5f['dense_xyz'][start:end]   # [B, N2, 3]
            pt_match_idx = h5f['sparse_matching_idx'][start:end]

            if verbose:
                print('Number of %s samples: %d' % (self.split, len(xyz_sparse)))

            if is_use_normal:
                norm_sparse = h5f['sparse_norm'][start:end] # [B, N1, 3]
                norm_dense  = h5f['dense_norm'][start:end]  # [B, N2, 3]
                return xyz_sparse, xyz_dense, norm_sparse, norm_dense, pt_match_idx
            else:
                return xyz_sparse, xyz_dense, pt_match_idx

    # -------------------------------------------------------
    def __getitem__(self, index: int):
        xyz_sparse = self.xyz_sparse[index]
        xyz_dense  = self.xyz_dense[index]
        pt_match_idx = self.pt_match_idx[index]

        if self.is_jitter:
            xyz_sparse = VisionAirPUGAN.jitter(xyz_sparse, sigma=0.01, clip=0.03)
        if self.is_rotate:
            xyz_sparse, xyz_dense = VisionAirPUGAN.rotate(xyz_sparse, xyz_dense, z_rotated=True)
        if self.is_scale:
            xyz_sparse, xyz_dense, scale = VisionAirPUGAN.scale(xyz_sparse, xyz_dense, scale_low=0.8, scale_high=1.2)

        if self.is_use_normal:
            norm_sparse = self.norm_sparse[index]
            norm_dense  = self.norm_dense[index]
            return xyz_sparse, xyz_dense, norm_sparse, norm_dense, pt_match_idx
        else:
            return xyz_sparse, xyz_dense, pt_match_idx

    # -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.xyz_sparse)


# ---------------------------------------------------------------------------------------------
class SketchfabPUGeoDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super(SketchfabPUGeoDataModule, self).__init__()
        self.cfg = hparams.dataset
        self.seed = hparams.seed

        # self.rootdir = Path(hydra.utils.get_original_cwd()) / self.cfg.root
        self.rootdir = Path(self.cfg.root)

    def config_dataset(self, split, shuffle=True, is_jitter=False, is_rotate=False, is_scale=False):

        dataset = SketchfabPUGeo(self.rootdir, split,
            self.cfg.is_normalize, is_jitter, is_rotate, is_scale,
            use_normal=self.cfg.is_use_normal,
            seed=self.seed, verbose=False)

        is_pin_memory = (split == 'train') or (split == 'full')
        dataloader = DataLoader(
            dataset, self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_worker, pin_memory=is_pin_memory, drop_last=False,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))
        return dataloader

    def train_dataloader(self):
        return self.config_dataset(split='train',
            shuffle=True, is_jitter=False, is_rotate=False, is_scale=False)
        # return self.config_dataset(split='full',
        #     shuffle=True, is_jitter=False, is_rotate=False, is_scale=False)

    def val_dataloader(self):
        return self.config_dataset(split='valid',
            shuffle=False, is_jitter=False, is_rotate=False, is_scale=False)

    def test_dataloader(self):
        return self.config_dataset(split='test',
            shuffle=False, is_jitter=False, is_rotate=False, is_scale=False)
# ---------------------------------------------------------------------------------------------



if __name__ == "__main__":
    pass
