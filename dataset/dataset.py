
import os
import logging

import numpy as np
import pytorch_lightning as pl

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_resource_variables()


from torch.utils.data import DataLoader

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data.dataset import Dataset
from dataset.fetcher import Fetcher



# ---------------------------------------------------------------------------------------------
class SketchfabPUGeo(Dataset):

    def __init__(self, cfg, num_batch, is_jitter):
        super(SketchfabPUGeo, self).__init__()

        if not isinstance(cfg.data_path, ListConfig):
            cfg.data_path = [cfg.data_path]
            cfg.up_ratio  = [cfg.up_ratio]

        self.num_batch     = num_batch
        self.input_channel = cfg.input_channel
        self.up_ratio      = cfg.up_ratio
        self.sample_prob   = cfg.sample_prob
        self.num_ratio = len(self.up_ratio)

        self.fetchers = []
        for i in range(len(cfg.data_path)):
            fetcher = Fetcher(
                records=cfg.data_path[i], batch_size=cfg.batch_size,
                step_ratio=cfg.up_ratio[i], up_ratio=cfg.up_ratio[i],
                num_in_point=cfg.num_point_patch, num_shape_point=cfg.num_shape_point,
                drop_out=cfg.drop_out, input_channel=cfg.input_channel,
                jitter=is_jitter, jitter_max=cfg.jitter_max, jitter_sigma=cfg.jitter_sigma)
            self.fetchers.append(fetcher)

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.allow_soft_placement     = True
        tf_config.log_device_placement     = False
        tf_config.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=tf_config)

    def fetch_data(self, dataloader, sess):
        input_sparse, gt_dense, input_r,_ = dataloader.fetch(sess)

        if self.input_channel == 3:
            feed_dict = {
                'input_sparse_xyz_pl': input_sparse,
                'gt_dense_xyz_pl'    : gt_dense,
                'input_r_pl'         : input_r,
            }
        else:
            input_sparse_xyz = input_sparse[:,:,0:3]
            input_sparse_normal = input_sparse[:,:,3:6]
            sparse_l2 = np.linalg.norm(input_sparse_normal, axis=-1, keepdims=True)
            sparse_l2 = np.tile(sparse_l2, [1,3])
            input_sparse_normal = np.divide(input_sparse_normal, sparse_l2)

            gt_dense_xyz = gt_dense[:,:,0:3]
            gt_dense_normal = gt_dense[:,:,3:6]
            dense_l2 = np.linalg.norm(gt_dense_normal, axis=-1, keepdims=True)
            dense_l2 = np.tile(dense_l2, [1,3])
            gt_dense_normal = np.divide(gt_dense_normal,dense_l2)

            feed_dict = {
                'input_sparse_xyz_pl': input_sparse_xyz,
                'gt_sparse_normal_pl': input_sparse_normal,
                'gt_dense_xyz_pl'    : gt_dense_xyz,
                'gt_dense_normal_pl' : gt_dense_normal,
                'input_r_pl'         : input_r,
            }
        return feed_dict

    def reset(self):
        for i, fetcher in enumerate(self.fetchers):
            fetcher.initialize(self.sess, self.up_ratio[i], False)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.num_batch


# ---------------------------------------------------------------------------------------------
class TrainSketchfabPUGeo(SketchfabPUGeo):

    def __getitem__(self, index):
        # i_th = np.random.choice(np.arange(self.num_ratio), 1, p=self.sample_prob)[0]
        # return self.fetch_data(self.fetchers[i_th], self.sess)
        return self.fetch_data(self.fetchers[0], self.sess)

# ---------------------------------------------------------------------------------------------
class ValidSketchfabPUGeo(SketchfabPUGeo):

    def __init__(self, cfg, num_batch, is_jitter):
        super(ValidSketchfabPUGeo, self).__init__(cfg, num_batch, is_jitter)
        self.ratio_counter = 0

    def __getitem__(self, index):
        i_th = self.ratio_counter
        self.ratio_counter = (self.ratio_counter + 1) % self.num_ratio
        return self.fetch_data(self.fetchers[i_th], self.sess)


# ---------------------------------------------------------------------
class ResetDatasetCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, _pl_module):
        trainer.train_dataloader.dataset.datasets.reset()

    def on_validation_epoch_start(self, trainer: pl.Trainer, _pl_module):
        for i in range(len(trainer.val_dataloaders)):
            trainer.val_dataloaders[i].dataset.reset()

    def on_test_epoch_start(self, trainer: pl.Trainer, _pl_module):
        for i in range(len(trainer.test_dataloaders)):
            trainer.test_dataloaders[i].dataset.reset()


class SketchfabPUGeoDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super(SketchfabPUGeoDataModule, self).__init__()
        self.cfg = hparams

    def config_dataset(self, split, num_batch, is_jitter):

        is_jitter = is_jitter and self.cfg.is_jitter
        if split == 'train':
            dataset = TrainSketchfabPUGeo(self.cfg, num_batch, is_jitter)
        else:
            dataset = ValidSketchfabPUGeo(self.cfg, num_batch, is_jitter)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
        return dataloader

    def train_dataloader(self):
        return self.config_dataset('train', self.cfg.num_batch_train, True)

    def val_dataloader(self):
        return self.config_dataset('valid', self.cfg.num_batch_valid, False)
    
    # def val_dataloader(self):
    #     dataset = PUGeoTestDataset()
    #     dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    #     return dataloader

    def test_dataloader(self):
        return self.config_dataset('test', self.cfg.num_batch_test, False)
# ---------------------------------------------------------------------------------------------
