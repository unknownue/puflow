
from time import time
import pytorch_lightning as pl

from utils.time import ElapseTimer


# ---------------------------------------------------------------------
class TimeTrainingCallback(pl.Callback):

    # -----------------------------------------------------------------
    def on_train_start(self, trainer, pl_module):
        self.timer = ElapseTimer()
        self.timer.start()

    # -----------------------------------------------------------------
    # def on_train_end(self, trainer, pl_module):
    #     self.timer.update()
    #     print(f'\nTraining time: \033[1m{self.timer}\033[0m')

    def on_keyboard_interrupt(self, trainer, pl_module):
        trainer.is_interrupted = True
    
    def on_fit_end(self, trainer, pl_module):
        self.timer.update()
        print(f'\nTraining time: \033[1m{self.timer}\033[0m')