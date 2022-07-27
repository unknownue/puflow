
import os
import logging

from omegaconf.omegaconf import OmegaConf
import pytorch_lightning as pl

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_resource_variables()

from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataset import Dataset
from dataset.pu1k.fetcher import Fetcher


# ---------------------------------------------------------------------------------------------
class PU1kDataset(Dataset):

    def __init__(self, cfg, use_random_input = False, num_batches=None):
        super(PU1kDataset, self).__init__()

        opts = OmegaConf.create({
            'random':          use_random_input,
            'train_file':      cfg.data_path,
            'batch_size':      cfg.batch_size,
            'patch_num_point': cfg.num_point_patch,
            'num_point':       cfg.num_point_patch,
            "augment":         cfg.is_augment,
            "jitter_sigma":    cfg.jitter_sigma,
            'up_ratio':        cfg.up_ratio,
            "jitter_max":      cfg.jitter_max,
        })
        self.fetcher = Fetcher(opts, num_batches)
        self.fetcher.start()
    
    def __getitem__(self, index):
        input_sparse_xyz, gt_dense_xyz, batch_radius = self.fetcher.fetch()
        
        feed_dict = {
            'input_sparse_xyz_pl': input_sparse_xyz,
            # 'gt_sparse_normal_pl': None,
            'gt_dense_xyz_pl':     gt_dense_xyz,
            # 'gt_dense_normal_lp':  None,
            'up_ratio_pl':          batch_radius,
        }
        return feed_dict
    
    def __len__(self):
        return self.fetcher.num_batches

# ---------------------------------------------------------------------
class PU1kDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super(PU1kDataModule, self).__init__()
        self.cfg = hparams
        
    def config_dataset(self, split, num_batches):

        if split == 'train':
            dataset = PU1kDataset(self.cfg, self.cfg.is_random_input, num_batches)
        else:
            dataset = PU1kDataset(self.cfg, False, num_batches)
        return DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    
    def train_dataloader(self):
        return self.config_dataset('train', num_batches=None)
        # return self.config_dataset('train', num_batches=300)

    def val_dataloader(self):
        return self.config_dataset('valid', num_batches=400)
    
    def test_dataloader(self):
        return self.config_dataset('test')


# ---------------------------------------------------------------------
class ShotdownDatasetCallback(pl.Callback):
    
    def on_train_end(self, trainer, pl_module):
        trainer.train_dataloader.dataset.datasets.fetcher.shutdown()
        trainer.val_dataloaders[0].dataset.fetcher.shutdown()
# ---------------------------------------------------------------------
