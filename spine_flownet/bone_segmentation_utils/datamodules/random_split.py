from datamodules.base_db_module import BaseDbModule, BasedModuleChildMeta
import abc
from utils.utils import get_argparser_group
from datasets.dataset_utils import make_dataset
import random

class RandomSplit(BaseDbModule, abc.ABC, metaclass=BasedModuleChildMeta):
    def __init__(self, hparams, dataset=None):
        super().__init__(hparams, dataset)
        self._set_random_splits()

    @staticmethod
    def _filter_list(data_list, keep):

        if keep == "images":
            return [item for item in data_list if "label" not in item]
        elif keep == "labels":
            return [item for item in data_list if "label" in item]

    def _get_splits_for_data(self, extract="images"):
        data_list = sorted(make_dataset(self.data_root, self.hparams.max_dataset_size))  # get image paths
        data_list = self._filter_list(data_list, keep=extract)
        dataset_size = len(data_list)

        if self.hparams.test_folder != '':

            test_data_list = sorted(make_dataset(self.hparams.test_folder, self.hparams.max_dataset_size))
            test_data_list = self._filter_list(test_data_list, keep=extract)

            n_training_samples = round(self.hparams.random_split[0] * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            val_data_list = [item for item in data_list if item not in train_data_list]

        else:
            n_training_samples = round(self.hparams.random_split[0] / 100 * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            if len(self.hparams.random_split) == 2:
                val_data_list = [item for item in data_list if item not in train_data_list]
                test_data_list = []

            else:
                n_val_samples = round(self.hparams.random_split[1] / 100 * dataset_size)
                val_data_list = random.sample(data_list, n_val_samples)

                test_data_list = [item for item in data_list if
                                  item not in train_data_list and item not in val_data_list]

        return [train_data_list, val_data_list, test_data_list]

    def prepare_data(self):

        if self.hparams.db_kind == 'paired':
            splits_data_lists = self._get_splits_for_data(extract="images")

        else:
            splits_data_lists_images = self._get_splits_for_data(extract="images")
            splits_data_lists_labels = self._get_splits_for_data(extract="labels")

            splits_data_lists = [[splits_data_lists_images[0], splits_data_lists_labels[0]],
                                 [splits_data_lists_images[1], splits_data_lists_labels[1]],
                                 [splits_data_lists_images[2], splits_data_lists_labels[2]]]

        for split, data_list in zip(['train', 'val', 'test'], splits_data_lists):
            self.data[split] = self.torch_dataset(self.hparams, data_list=data_list, split=split)

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        parser = BaseDbModule.add_dataset_specific_args(parser)
        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--test_folder", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)

        return parser
