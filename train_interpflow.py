        
import warnings
warnings.filterwarnings('ignore')
# import logging
# logging.getLogger("lightning").setLevel(logging.INFO)

import torch
import pytorch_lightning as pl

from torch.tensor import Tensor
from omegaconf import OmegaConf

from dataset.dataset import SketchfabPUGeoDataModule, ResetDatasetCallback
from metric.loss import UpsampingMetrics
from modules.nets.interpflow import PointInterpFlow

from utils.callback import TimeTrainingCallback
from utils.lightning import LightningModule
from utils.modules import print_progress_log



# -----------------------------------------------------------------------------------------
class TrainerModule(LightningModule):

    def __init__(self):
        super(TrainerModule, self).__init__()

        self.network = PointInterpFlow(pc_channel=3, num_neighbors=8)  # k=8

        self.train_metrics = UpsampingMetrics(['CD', 'EMD'])
        self.valid_metrics = UpsampingMetrics(['CD', 'EMD', 'HD'])
        self.epoch_counter = 1

        self.min_CD    = 100.0

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-6, 2e-4, 40, cycle_momentum=False)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.2)
        # return { 'optimizer': optimizer, 'scheduler': scheduler }

    def training_step(self, batch, batch_idx):

        xyz_dense  = batch['gt_dense_xyz_pl'].squeeze()
        xyz_sparse = batch['input_sparse_xyz_pl'].squeeze()
        radius     = batch['input_r_pl'].squeeze()
        upratio = int(xyz_dense.shape[1] / xyz_sparse.shape[1])

        xyz_pred, logpx = self(xyz_sparse, upratio=upratio)

        metrics = self.train_metrics.evaluate(xyz_pred, xyz_dense, radius)
        # loss = logpx * 1e-5 + metrics['CD'] * 10.0 + metrics['Uniform'] * 0.5 + metrics['EMD'] * 0.5
        loss = logpx * 1e-5 + metrics['CD'] * 10.0 + metrics['EMD'] * 0.5

        self.log('CD', metrics['CD'] * 10.0, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        # self.log('Un', metrics['Uniform'] * 0.5, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        # self.log('EMD', metrics['EMD'] * 0.5, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-5, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):

        xyz_dense  = batch['gt_dense_xyz_pl'].squeeze()
        xyz_sparse = batch['input_sparse_xyz_pl'].squeeze()
        radius     = batch['input_r_pl'].squeeze()
        upratio = int(xyz_dense.shape[1] / xyz_sparse.shape[1])

        predict_x, logpx = self(xyz_sparse, upratio=upratio)
        metrics = self.valid_metrics.evaluate(predict_x, xyz_dense, radius)

        return {
            'vloss'    : logpx.detach().cpu(),
            'CD'       : metrics['CD'],
            'EMD'      : metrics['EMD'],
            'HD'       : metrics['HD'],
        }

    def validation_epoch_end(self, batch):

        log_dict = {
            'vloss'    : torch.tensor([x['vloss'] * 1e-5 for x in batch]).sum().item(),
            'CD'       : torch.tensor([x['CD']        for x in batch]).sum().item(),
            'EMD'      : torch.tensor([x['EMD']       for x in batch]).sum().item(),
            'HD'       : torch.tensor([x['HD']        for x in batch]).sum().item(),
        }
        
        if log_dict['CD'] < self.min_CD:
            save_path = f'runs/ckpt/puflow-x4-CD.ckpt'
            torch.save(self.network.state_dict(), save_path)
            print(f'Best CD metric -> {save_path}')
            self.min_CD = log_dict['CD']

        print_progress_log(self.epoch_counter, log_dict)
        self.epoch_counter += 1


# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path: str=None, begin_checkpoint: str=None):

    comment = 'Baseline-Final'

    geo_dataset_cfg = OmegaConf.create({
        'data_path' : 'data/tfrecord_x4_normal/*.tfrecord',
        'batch_size': 32,  # 32
        'num_batch_train': 270,  # 270
        'num_batch_valid': 30,   # 30
        'num_batch_test' : 100,
        'input_channel'  : 6,
        'up_ratio': 4,
        'num_point_patch': 256,
        'num_shape_point': 5000,
        'drop_out': 1.0,
        'jitter'      : False,
        'jitter_max'  : 0.03,
        'jitter_sigma': 0.01,
    })
    datamodule = SketchfabPUGeoDataModule(geo_dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : 100,
        'precision'            : 32,   # 16
        # 'amp_level'            : 'O1',
        'weights_summary'      : 'top',  # 'top', 'full' or None
        'gradient_clip_val'    : 1e-2,
        'deterministic'        : False,
        'num_sanity_val_steps' : -1,  # -1 or 0
        'checkpoint_callback'  : False,
        'callbacks'            : [TimeTrainingCallback(), ResetDatasetCallback()],
    }

    module = TrainerModule()
    trainer = pl.Trainer(**trainer_config)
    trainer.is_interrupted = False
 
    if phase == 'Train':
        if comment is not None:
            print(f'\nComment: \033[1m{comment}\033[0m')
        if begin_checkpoint is not None:
            module.network.load_state_dict(torch.load(begin_checkpoint))
            module.network.set_to_initialized_state()

        trainer.fit(model=module, datamodule=datamodule)

        if checkpoint_path is not None and trainer_config['fast_dev_run'] is False and trainer.is_interrupted is False:
            save_path = checkpoint_path + f'-epoch{trainer_config["max_epochs"]}.ckpt'
            torch.save(module.network.state_dict(), save_path)
            print(f'Model has been save to \033[1m{save_path}\033[0m')




# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    checkpoint_path = 'runs/ckpt/puflow-x4-baseline.ckpt'
    previous_path = 'runs/ckpt/puflow-x4-CD.ckpt'

    # train('Train', None, None)                      # Train from begining, and save nothing after finish
    train('Train', checkpoint_path, None)           # Train from begining, save network params after finish
    # train('Train', checkpoint_path, previous_path)  # Train from previous checkpoint, save network params after finish

