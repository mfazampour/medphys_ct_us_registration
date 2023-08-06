from polyaxon_client.tracking import Experiment
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict, Optional, Union
import argparse
from pathlib import Path
import wandb

class BaseLogger(LightningLoggerBase):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._experiment = None

    @property
    def experiment(self):
        """Return the experiment object associated with this logger"""
        return self._experiment

    @property
    def name(self) -> str:
        """Return the experiment name."""
        raise NotImplementedError

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        raise NotImplementedError

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.

        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        raise NotImplementedError

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
                  Polyaxon currently does not support assigning a specific step.
        """

        raise NotImplementedError

    def log_model(self, model):
        raise NotImplementedError

    def log_image(self, model):
        raise NotImplementedError


class PolyaxonLogger(BaseLogger):
    """Docstring for PolyaxonLogger. """

    def __init__(self, hparams):
        """TODO: to be defined. """
        super().__init__(hparams)
        self._experiment = Experiment()
        self.output_path = Path(self.experiment.get_outputs_path())

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
                  Polyaxon currently does not support assigning a specific step.
        """

        self.experiment.log_metrics(step=step, **metrics)

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.
        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        self.experiment.log_params(**params)

    @property
    def name(self) -> str:
        """Return the experiment name."""
        if self._experiment.get_experiment_info() is not None:
            return self._experiment.get_experiment_info()['project_name']

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        if self._experiment.get_experiment_info() is not None:
            return self._experiment.experiment_id

    def log_model(self, model):
        """ Polyaxon cannot log models, therefore it returns without doing anything """
        return

    def log_image(self, image):
        """ Polyaxon cannot log images, therefore it returns without doing anything """
        return
#
# # TODO: remove hard-coded project name
class WandbLogger(BaseLogger):
    """Docstring for PolyaxonLogger. """

    def __init__(self, hparams):
        super().__init__(hparams)

        if hparams.group_name == "":
            wandb.init(project="UnetTraining", dir=hparams.output_path)
        else:
            wandb.init(project="UnetTraining", group=hparams.group_name, job_type="train", dir=hparams.output_path)

        self._config = wandb.config

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
                  Polyaxon currently does not support assigning a specific step.
        """
        wandb.log(metrics)

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.
        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        wandb.config.update(params)

    def log_model(self, model):
        """ Polyaxon cannot log models, therefore it returns without doing anything """
        wandb.watch(model)

    def log_image(self, image, titles="", main_title=""):
        """ Polyaxon cannot log images, therefore it returns without doing anything """

        if not isinstance(image, list):
            image = [image]

        if not isinstance(titles, list):
            titles = [titles]

        wandb.log({main_title: [wandb.Image(img, caption=title) for img, title in zip(image, titles)]})
        return

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return "" #TODO: change this

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        return None

