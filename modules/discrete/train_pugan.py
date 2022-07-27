
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import pytorch_lightning as pl

from torch import Tensor
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor

sys.path.append(os.getcwd())

from dataset.pugan.dataset2 import PUGANdatasetDataModule
from metric.loss import ChamferCUDA, EarthMoverDistance, ChamferCUDA2
from modules.discrete.interpflow import PointInterpFlow

from utils.callback import TimeTrainingCallback
from utils.lightning import LightningProgressBar
from utils.modules import print_progress_log



# -----------------------------------------------------------------------------------------
class TrainerModule(pl.LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.cfg = cfg
        self.network = PointInterpFlow(pc_channel=3)
        
        self.emd_loss = EarthMoverDistance()
        self.chamfer_loss = ChamferCUDA()
        self.chamfer_loss2 = ChamferCUDA2()

        self.epoch = 0
        self.min_CD = 100.0
        self.min_Nor = 15.0

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.cfg.sched_factor, verbose=1, min_lr=1e-4, patience=self.cfg.sched_patience)
        return { 'optimizer': optimizer, 'scheduler': scheduler }

    def training_step(self, batch, batch_idx):

        xyz_sparse, xyz_dense, radius = batch
        upratio = int(xyz_dense.shape[1] / xyz_sparse.shape[1])

        xyz_pred, logpx = self(xyz_sparse, upratio=upratio)

        emd = self.emd_loss(xyz_pred, xyz_dense, radius=radius)
        cd, _ = self.chamfer_loss(xyz_pred, xyz_dense)
        loss = logpx * 1e-4 + emd * 5e-2 + cd * 1e-1

        self.log('CD', cd * 1e-1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('EMD', emd * 5e-2, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-4, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):

        xyz_sparse, xyz_dense, radius = batch
        upratio = int(xyz_dense.shape[1] / xyz_sparse.shape[1])

        predict_x, logpx = self(xyz_sparse, upratio=upratio)
        cd = self.chamfer_loss2(predict_x, xyz_dense)

        valid_dict = {
            'vloss': logpx.detach().cpu(),
            'CD'   : cd,
        }

        return valid_dict

    def validation_epoch_end(self, batch):

        log_dict = {
            'vloss': torch.tensor([x['vloss'] * 1e-5 for x in batch]).sum().item(),
            'CD'   : torch.tensor([x['CD']           for x in batch]).sum().item(),
        }

        self.log('CD',       log_dict['CD'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('vloss', log_dict['vloss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        print_progress_log(self.epoch, { 'CD': log_dict['CD'] }, extra=[])
        self.epoch += 1


# -----------------------------------------------------------------------------------------
def model_specific_args():
    parser = ArgumentParser()

    # Network
    parser.add_argument('--net', type=str, default='UpsamplingFlow', help='Network name')
    # Optimizer and scheduler
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--sched_patience', default=10, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    # Training
    # parser.add_argument('--max_epoch', default=150, type=int)
    parser.add_argument('--seed', default=2021, type=int)

    return parser

# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path: str=None, begin_checkpoint: str=None):

    cfg = model_specific_args().parse_args()
    pl.seed_everything(cfg.seed)

    comment = 'Baseline-pugan'

    pugan_dataset_cfg = OmegaConf.create({
        "use_non_uniform": True,
        "train_file": 'data/PUGAN_poisson_256_poisson_1024.h5',
        "batch_size": 32,
        "patch_num_point": 256,
        "augment": False,
        "jitter_sigma": 0.01,
        "jitter_max": 0.03,
        "up_ratio": 4,
    })
    datamodule = PUGANdatasetDataModule(pugan_dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : 300,
        'precision'            : 32,   # 32, 16, bf16
        'gradient_clip_val'    : 1e-2,
        'deterministic'        : False,
        'num_sanity_val_steps' : -1,  # -1 or 0
        'enable_checkpointing' : False,
        'callbacks'            : [TimeTrainingCallback(), LightningProgressBar()],
    }

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer_config['callbacks'].append(lr_monitor)

    module = TrainerModule(cfg)
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
            if trainer_config['max_epochs'] > 10:
                save_path = checkpoint_path.replace('.ckpt', f'-epoch{trainer_config["max_epochs"]}.ckpt')
                torch.save(module.network.state_dict(), save_path)
                print(f'Model has been save to \033[1m{save_path}\033[0m')



# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    checkpoint_path = 'runs/ckpt/puflow-x4-pugan.ckpt'
    train('Train', checkpoint_path, None)
    
