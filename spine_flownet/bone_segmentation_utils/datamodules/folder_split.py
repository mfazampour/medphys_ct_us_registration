from datamodules.base_db_module import BaseDbModule, BasedModuleChildMeta
import abc
from net_utils.utils import get_argparser_group


class FolderSplit(BaseDbModule, abc.ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams, dataset=None):
        super().__init__(hparams, dataset)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = self.torch_dataset(self.hparams, split, data_structure='folder_based')

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        parser = BaseDbModule.add_dataset_specific_args(parser)

        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--unpaired_percentage", default=30, type=float)

        return parser
