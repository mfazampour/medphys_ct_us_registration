from datamodules.base_db_module import BaseDbModule, BasedModuleChildMeta
import abc
import os
from torch.utils.data import DataLoader
from utils.utils import get_argparser_group
import numpy as np
from utils.utils import str2bool


class MixedDb(BaseDbModule, abc.ABC, metaclass=BasedModuleChildMeta):
    def __init__(self, hparams, dataset=None):
        super().__init__(hparams, dataset)
        self._reformat_hparams()

        self.data_root = dict()
        for split in ['train', 'val', 'test']:
            self.data_root[split] = [os.path.join(self.hparams.data_root, item)
                                     for item in getattr(self.hparams, split + "_folders")]

    # todo: do this in argparse instead
    def _reformat_hparams(self):
        for item in ['train_folders', 'val_folders', 'test_folders']:
            split_folders = getattr(self.hparams, item)
            split_folders = split_folders.replace(" ", "")
            split_folders = split_folders.split(",")
            setattr(self.hparams, item, split_folders)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = [self.torch_dataset(hparams=self.hparams,
                                                   split=split,
                                                   data_root=item) for item in self.data_root[split]]

    def __dataloader(self, split=None):

        if split != 'train':

            dataloaders = [DataLoader(dataset=item,
                                      batch_size=self.hparams.batch_size,
                                      shuffle=False,
                                      sampler=None,
                                      num_workers=self.hparams.num_workers,
                                      pin_memory=True) for item in self.data[split]]
            return dataloaders

        dataloaders = dict()
        for dataset, folder_name in zip(self.data[split], getattr(self.hparams, split + "_folders")):
            shuffle_db = split == 'train'
            train_sampler = None
            dataloaders[folder_name] = DataLoader(dataset=dataset,
                                                  batch_size=self.hparams.batch_size,
                                                  shuffle=shuffle_db,
                                                  sampler=train_sampler,
                                                  num_workers=self.hparams.num_workers,
                                                  pin_memory=True)

        return dataloaders

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
        dataset_specific_args.add_argument("--train_folders", default="", type=str)
        dataset_specific_args.add_argument("--val_folders", default="", type=str)
        dataset_specific_args.add_argument("--test_folders", default="", type=str)

        return parser
