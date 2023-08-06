from datamodules.base_db_module import BaseDbModule, BasedModuleChildMeta
import abc
from net_utils.utils import get_argparser_group
from datasets.dataset_utils import get_subject_ids_from_data, get_subject_based_random_split
import os
import numpy as np


class SubjectSplit(BaseDbModule, abc.ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams, dataset=None):
        super().__init__(hparams, dataset)

        if hparams.max_dataset_size < 0:
            hparams.max_dataset_size = np.inf

        self._set_split_subjects()
        self._set_random_splits()

    def _set_split_subjects(self):

        for split in ['train', 'val', 'test']:

            split_subjects = getattr(self.hparams, split + '_subjects')
            subject_list = split_subjects.split(",")
            subject_list = [item.replace(" ", "") for item in subject_list]

            subject_list = [item for item in subject_list if item != "" and item != " "]
            setattr(self.hparams, split + "_subjects", subject_list)

    def log_db_info(self):

        print("\n---------------------------------------------------------------------------------")
        if len(self.hparams.random_split) == 3:
            print("Db split: train: {} - val: {} - test: {}".format(self.hparams.random_split[0],
                                                                    self.hparams.random_split[1],
                                                                    self.hparams.random_split[2]))

        if len(self.hparams.random_split) == 2:
            print("Db split: train: {} - val: {} - test: {}".format(self.hparams.random_split[0],
                                                                    self.hparams.random_split[1],
                                                                    0))

        for split in ['train', 'val', 'test']:

            subject_list = getattr(self.hparams, split + "_subjects")

            print("Num subjects in {} split: {} - num data: {}".format(split,
                                                                       len(subject_list),
                                                                       len(self.data[split].AB_paths) ))

            string_to_plot = "{} split ids : ".format(split)

            for i in subject_list:
                string_to_plot += "{}, ".format(i)

            print(string_to_plot[0:-2])

        print("---------------------------------------------------------------------------------\n")

    def prepare_data(self):

        train_given = len(self.hparams.train_subjects) != 0
        val_given = len(self.hparams.val_subjects) != 0
        test_given = len(self.hparams.test_subjects) != 0

        subject_ids = get_subject_ids_from_data(os.listdir(self.data_root))

        # if the test is not given either we have a random split with the percentages in self.hparams.random_split or
        # we assume the test set is empty
        if not test_given:
            if not train_given and not val_given:
                self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects = \
                    get_subject_based_random_split(subject_ids, split_percentages=self.hparams.random_split)

            elif train_given and val_given:
                self.hparams.test_subjects = []

            elif not train_given and val_given:
                self.hparams.train_subjects = [item for item in subject_ids if item not in self.hparams.val_subjects]

            elif train_given and not val_given:
                self.hparams.val_subjects = [item for item in subject_ids if item not in self.hparams.train_subjects]

            else:
                raise ValueError("Unsupported configuration")

        if test_given:
            subject_ids = [item for item in subject_ids if item not in self.hparams.test_subjects]

            if train_given and val_given:
                pass

            elif not train_given and not val_given:
                assert len(self.hparams.random_split) == 2, "If test set is given, random split must contain only" \
                                                            " two values, one for train and one for validation"
                self.hparams.train_subjects, self.hparams.val_subjects, _ = \
                    get_subject_based_random_split(subject_ids, split_percentages=self.hparams.random_split)

            elif not train_given and val_given:
                self.hparams.train_subjects = [item for item in subject_ids if item not in self.hparams.val_subjects]

            elif train_given and not val_given:
                self.hparams.val_subjects = [item for item in subject_ids if item not in self.hparams.train_subjects]

            else:
                raise ValueError("Unsupported configuration")

        for subject_list, split in zip(
                [self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects],
                ['train', 'val', 'test']):

            self.data[split] = self.torch_dataset(hparams=self.hparams,
                                                  split=split,
                                                  subject_list=subject_list,
                                                  data_structure='subject_based')

        self.log_db_info()

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """

        parser = BaseDbModule.add_dataset_specific_args(parser)

        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--train_subjects", default='', type=str)
        dataset_specific_args.add_argument("--val_subjects", default='', type=str)
        dataset_specific_args.add_argument("--test_subjects", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)

        return parser