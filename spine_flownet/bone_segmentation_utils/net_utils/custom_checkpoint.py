from pytorch_lightning.callbacks.model_checkpoint import *

class CustomModelCheckpoint(ModelCheckpoint):
    def save_checkpoint(self, trainer, pl_module):
        if trainer.current_epoch <= trainer.min_epochs:
            return
        else:
            super().save_checkpoint(trainer, pl_module)

    def _update_best_and_save(self, *args, **kwargs):
        super()._update_best_and_save( *args, **kwargs)
        self.best_epoch = args[-2].current_epoch