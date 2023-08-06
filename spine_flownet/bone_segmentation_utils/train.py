import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from net_utils.plx_logger import PolyaxonLogger, WandbLogger
from net_utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from net_utils.configargparse_arguments import build_configargparser
from net_utils.custom_early_stop import CustomEarlyStopping
from datetime import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train(hparams, ModuleClass, ModelClass, DatasetClass, DataModuleClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # load model
    print('In train.py: Loading model...')
    model = ModelClass(hparams=hparams)
    print('...done.')
    # load dataset
    print('In train.py: Loading dataset...')
    dataset = DataModuleClass(hparams=hparams, dataset=DatasetClass)
    print('...done.')
    # load module
    print('In train.py: Loading module...')
    module = ModuleClass(hparams, model, logger=logger)
    print('...done.')

    # ------------------------
    # 3 INIT TRAINER --> continues training
    # ------------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='min',
        prefix=hparams.name,
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )

    early_stop_callback = CustomEarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=5,
        mode='min')

    trainer = Trainer(
        gpus=hparams.gpus,
        logger=logger,
        # fast_dev_run: if true, runs one training and one validation batch
        fast_dev_run=hparams.fast_dev_run,
        # min_epochs: forces training to a minimum number of epochs
        min_epochs=hparams.min_epochs,
        # max_epochs: limits training to a maximum number of epochs
        max_epochs=hparams.max_epochs,
        # saves the state of the last training epoch (all model parameters)
        checkpoint_callback=True,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[early_stop_callback, checkpoint_callback],  # [early_stop_callback, checkpoint_callback]
        weights_summary='full',
        # runs a certain number of validation steps before training to catch bugs
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        # auto_lr_find: if true, will find a learning rate that optimizes initial learning for faster convergence
        auto_lr_find=True,
        # auto_scale_batch_size: if true, will initially find the largest batch size that fits into memory
        auto_scale_batch_size=True,
        # limit_train_batches=0.04,  # use 0.2 for Polyaxon, use 0.03 to avoid memory error on Anna's computer
        # limit_val_batches=0.06,  # use 0.4 for Polyaxon, use 0.05 to avoid memory error on Anna's computer
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------

    if not hparams.test_only:
        trainer.fit(module, dataset)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    else:
        print(
            f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} "
            f"| path: {checkpoint_callback.best_model_path}"
            f"\nTesting...")

        dataset.prepare_data()
        test_loader = dataset.test_dataloader()

        trainer.test(ckpt_path=checkpoint_callback.best_model_path, model=module, test_dataloaders=test_loader)

    print("test done")


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # each LightningModule defines arguments relevant to it
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)
    # ------------------------
    # LOAD DATAMODULE
    # ------------------------
    datamodule_path = f"datamodules.{hparams.datamodule}"
    DataModuleClass = get_class_by_path(datamodule_path)
    parser = DataModuleClass.add_dataset_specific_args(parser)
    # ------------------------
    #  PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()
    # hparams.data_root = input_folder
    # print(hparams.data_root)
    # hparams.augmentation_prob = augmentation_prob

    # setup logging
    exp_name = (
            hparams.module.split(".")[-1]
            + "_"
            + hparams.dataset.split(".")[-1]
            + "_"
            + hparams.model.replace(".", "_")
    )
    if hparams.on_polyaxon:
        plx_logger = PolyaxonLogger(hparams)
        hparams.output_path = plx_logger.output_path
        hparams = plx_logger.hparams
        hparams.name = plx_logger.experiment.experiment_id + "_" + exp_name
    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        hparams.name = date_str + exp_name
        hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb', log_graph=True)
    wb_logger = WandbLogger(hparams)

    argparse_summary(hparams, parser)
    loggers = [tb_logger, plx_logger, wb_logger] if hparams.on_polyaxon else [tb_logger, wb_logger]

    # ---------------------
    # RUN TRAINING
    # ---------------------

    train(hparams, ModuleClass, ModelClass, DatasetClass, DataModuleClass, loggers)
