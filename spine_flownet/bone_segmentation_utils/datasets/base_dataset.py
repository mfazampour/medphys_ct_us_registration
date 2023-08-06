from datasets.dataset_utils import *
from torch.utils.data import Dataset
import os
from random import shuffle
from argparse import Namespace

class BaseDataset(Dataset):

    def __init__(self):
        self.current_epoch = 0

    def __len__(self):
        """Return the total number of images in the dataset."""
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError

    def update_epoch(self):
        self.current_epoch += 1

    @staticmethod
    def copy_conf(default_opt, **kwargs):
        conf = Namespace(**vars(default_opt))
        for key in kwargs:
            setattr(conf, key, kwargs[key])
        return conf


class BasePairedDataset(BaseDataset):
    def __init__(self, hparams, split, **kwargs):
        super().__init__()

        assert split in ['train', 'val', 'test'], "Split must be a string - train, test or val"
        assert (hparams.load_size >= hparams.crop_size)

        self.hparams = hparams
        self.input_nc = hparams.in_channels
        self.output_nc = hparams.out_channels
        self.split = split

        data_structure = kwargs['data_structure'] if 'data_structure' in kwargs else 'none'
        self.data_root = kwargs['data_root'] if "data_root" in kwargs else hparams.data_root

        if 'data_list' in kwargs and isinstance(kwargs['data_list'], list):
            self.AB_paths = kwargs['data_list']
            return

        self.dir_AB = os.path.join(self.data_root, split) if data_structure == 'folder_based' else self.data_root
        self.AB_paths = sorted(make_dataset(self.dir_AB, self.hparams.max_dataset_size))  # get image paths
        self.AB_paths = [item for item in self.AB_paths if "@" not in item and "label" not in os.path.split(item)[-1]]

        if 'subject_list' in kwargs and isinstance(kwargs['subject_list'], list):
            self.AB_paths = get_split_subjects_data(self.AB_paths, kwargs['subject_list'])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, idx):
        raise NotImplementedError


class BaseUnpairedDataset(BaseDataset):
    def __init__(self, hparams, split, **kwargs):
        super().__init__()

        self.hparams = hparams
        self.input_nc = self.hparams.input_nc
        self.output_nc = self.hparams.output_nc
        self.split = split

        if 'data_list' in kwargs and isinstance(kwargs['data_list'], list):
            assert 'subject_list' not in kwargs, "either data_list OR subject list should be given, not both"
            self.A_paths = kwargs['data_list'][0]
            self.B_paths = kwargs['data_list'][1]

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            return

        data_structure = kwargs['data_structure'] if 'data_structure' in kwargs else 'none'
        self.AB_dir = os.path.join(self.hparams.data_root, split) if data_structure == 'folder_based' \
            else self.hparams.data_root

        if os.path.exists(self.AB_dir + "A") and os.path.exists(self.AB_dir + "B"):
            self.A_paths = [os.path.join(self.AB_dir + "A", item) for item in os.listdir(self.AB_dir + "A")]
            self.B_paths = [os.path.join(self.AB_dir + "B", item) for item in os.listdir(self.AB_dir + "B")]
        else:
            self.AB_paths = sorted(make_dataset(self.AB_dir, self.hparams.max_dataset_size))
            self.A_paths = [item for item in self.AB_paths if "label" in os.path.split(item)[-1]]
            self.B_paths = [item for item in self.AB_paths if "label" not in os.path.split(item)[-1]]
            shuffle(self.B_paths)

        if 'subject_list' in kwargs and isinstance(kwargs['subject_list'], list):
            self.A_paths = get_split_subjects_data(self.A_paths, kwargs['subject_list'])
            self.B_paths = get_split_subjects_data(self.B_paths, kwargs['subject_list'])

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        raise NotImplementedError
