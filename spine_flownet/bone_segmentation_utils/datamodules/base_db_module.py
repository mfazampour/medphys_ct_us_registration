from torch.utils.data import DataLoader
from net_utils.utils import get_argparser_group
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
from net_utils.utils import str2bool
import abc
from datasets.frame_paired import FramePaired


class BaseDbModuleMeta(type(pl.LightningDataModule), type(abc.ABC)):
    pass


class BaseDbModule(pl.LightningDataModule, abc.ABC, metaclass=BaseDbModuleMeta):
    def __init__(self, hparams, dataset=None):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.out_channels
        self.input_channels = self.hparams.in_channels
        self.batch_size = self.hparams.batch_size
        self.data = {}

        if dataset is not None:
            self.torch_dataset = dataset
        else:
            self.torch_dataset = FramePaired

    def _set_random_splits(self):

        if isinstance(self.hparams.random_split, str):
            splits = self.hparams.random_split.split("_")
            splits = [int(item) for item in splits]
            self.hparams.random_split = splits

    def __dataloader(self, split=None):
        dataset = self.data[split]
        shuffle = split == 'train'  # shuffle also for
        train_sampler = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split='train')
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split='val')
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split='test')
        return dataloader

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument('--preprocess', default='resize_and_crop', type=str)
        dataset_specific_args.add_argument('--load_size', default=256, type=int)
        dataset_specific_args.add_argument('--crop_size', default=256, type=int)
        dataset_specific_args.add_argument('--max_dataset_size', default=np.inf, type=float)
        dataset_specific_args.add_argument("--no_flip", default=True, type=str2bool)
        dataset_specific_args.add_argument("--serial_batches", default=True, type=str2bool)

        return parser


class BasedModuleChildMeta(type(BaseDbModule), type(abc.ABC)):
    pass

