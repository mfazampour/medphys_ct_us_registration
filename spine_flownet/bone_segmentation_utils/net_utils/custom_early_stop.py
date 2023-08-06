from  pytorch_lightning.callbacks.early_stopping import *

class CustomEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer, pl_module):
            """
            Checks whether the early stopping condition is met
            and if so tells the trainer to stop the training.
            """
            logs = trainer.logger_connector.callback_metrics

            if not self._validate_condition_metric(logs):
                return  # short circuit if metric not present

            current = logs.get(self.monitor)

            # when in dev debugging
            trainer.dev_debugger.track_early_stopping_history(self, current)

            if not isinstance(current, torch.Tensor):
                current = torch.tensor(current, device=pl_module.device)

            if trainer.use_tpu and TPU_AVAILABLE:
                current = current.cpu()
            if  trainer.current_epoch > trainer.min_epochs:
                if self.monitor_op(current - self.min_delta, self.best_score):
                    self.best_score = current
                    self.wait_count = 0
                else:
                    self.wait_count += 1
                    should_stop = self.wait_count >= self.patience

                    if bool(should_stop):
                        self.stopped_epoch = trainer.current_epoch
                        trainer.should_stop = True

            # stop every ddp process if any world process decides to stop
            #should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
            #trainer.should_stop = should_stop