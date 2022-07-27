
import numpy as np
import h5py
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from omegaconf.dictconfig import DictConfig


# ---------------------------------------------------------------------------------------------
class PUGANdatasetDataset(Dataset):

    def __init__(self, cfg: DictConfig, split: str):
        super(PUGANdatasetDataset, self).__init__()

        self.cfg = cfg.copy()
        self.pt_input, self.pt_gt, self.pt_radius = PUGANdatasetDataset.load_h5_data(path=cfg.train_file, patch_size=cfg.patch_num_point, is_random_input=cfg.use_non_uniform, up_ratio=cfg.up_ratio, is_normalize=True, verbose=True)
        self.sample_cnt = self.pt_input.shape[0]

        if split == 'valid' or split == 'test':
            np.random.seed(2022)
            self.cfg.augment = False
        self.split = split

    @staticmethod
    def load_h5_data(path, patch_size, is_random_input, up_ratio, is_normalize, verbose):

        num_patch__4x = int(patch_size * 4)
        num_patch_out = int(patch_size * up_ratio)

        with h5py.File(path, 'r') as f:
            if is_random_input:
                if verbose:
                    print('Use random input: ', path)
                pt_input = f['poisson_%d' % num_patch__4x][:].astype(np.float32)
                pt_gt    = f['poisson_%d' % num_patch_out][:].astype(np.float32)
            else:
                if verbose:
                    print('Do not use random input: ', path)
                pt_input = f['poisson_%d' % patch_size][:].astype(np.float32)
                pt_gt    = f['poisson_%d' % num_patch_out][:].astype(np.float32)
            
            assert len(pt_input) == len(pt_gt)
        
        radius = np.ones(shape=(len(pt_input)), dtype=np.float32)
        if is_normalize:
            if verbose:
                print("Normalization the data")
            centroid = np.mean(pt_gt[:, :, 0:3], axis=1, keepdims=True)
            pt_gt[:, :, 0:3] = pt_gt[:, :, 0:3] - centroid
            furthest_distance = np.amax(np.sqrt(np.sum(pt_gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
            pt_gt[:, :, 0:3] = pt_gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
            pt_input[:, :, 0:3] = pt_input[:, :, 0:3] - centroid
            pt_input[:, :, 0:3] = pt_input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        
        if verbose:
            print("Total %d samples" % (len(pt_input)))
        return pt_input, pt_gt, radius

    def __getitem__(self, index):
        pi = self.pt_input[index]
        pg = self.pt_gt[index]
        pr = self.pt_radius[index]

        if self.cfg.use_non_uniform:
            permute_idx = np.random.permutation(pi.shape[0])[:self.cfg.patch_num_point]
            pi = pi[permute_idx]
        if self.cfg.augment:
            pi = PUGANdatasetDataset.jitter(pi, sigma=self.cfg.jitter_sigma, clip=self.cfg.jitter_max)
            pi, pg, scales = PUGANdatasetDataset.scale(pi, pg, scale_low=0.8, scale_high=1.2)
            pr = pr * scales
        pi, pg = PUGANdatasetDataset.rotate(pi, pg)

        return pi, pg, pr

    def __len__(self):
        return self.sample_cnt

    @staticmethod
    def jitter(pt, sigma=0.005, clip=0.02):
        jitter_noise = np.clip(sigma * np.random.randn(*pt.shape), -1.0 * clip, clip).astype(np.float32)
        return pt + jitter_noise

    @staticmethod
    def rotate(pt_input, pt_gt, z_rotated=True):
        """
        Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            Original point cloud, in [N, 3]
        Return:
            Rotated point cloud, in [N, 3]
        """
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]),  np.cos(angles[0])]
        ])
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        Rz = np.array([[
            np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        if z_rotated:
            rotation_matrix = Rz.astype(np.float32)
        else:
            rotation_matrix = np.dot(Rz, np.dot(Ry, Rx)).astype(np.float32)

        pt_input = np.dot(pt_input, rotation_matrix)
        pt_gt    = np.dot(pt_gt, rotation_matrix)

        return pt_input, pt_gt

    @staticmethod
    def scale(pt_input, pt_gt, scale_low=0.5, scale_high=2):
        """
        Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Original point cloud, in [N, 3]
        Return:
            Scaled point cloud, in [N, 3]
        """
        scale = np.random.uniform(scale_low, scale_high, 1).astype(np.float32)
        return pt_input * scale, pt_gt * scale, scale


# ---------------------------------------------------------------------
class PUGANdatasetDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super(PUGANdatasetDataModule, self).__init__()
        self.cfg = hparams
        
    def config_dataset(self, split):
        dataset = PUGANdatasetDataset(self.cfg, split)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, num_workers=8, pin_memory=True)

    def train_dataloader(self):
        return self.config_dataset('train')

    def val_dataloader(self):
        return self.config_dataset('valid')
    
    def test_dataloader(self):
        return self.config_dataset('test')
