from datasets.base_dataset import BasePairedDataset
from PIL import Image
import os
from datasets.dataset_utils import get_params, get_transform
from net_utils.utils import get_argparser_group

class FramePaired(BasePairedDataset):
    def __init__(self, hparams, split, **kwargs):
        super().__init__(hparams, split, **kwargs)

    @staticmethod
    def get_label_name(image_name):
        image_id = image_name.split("_")[-1]
        label_name = image_name.replace(image_id, "label_" + image_id)
        return label_name

    def __getitem__(self, idx):
        # read a image given a random integer index
        A_path = FramePaired.get_label_name(self.AB_paths[idx])  # label
        B_path = self.AB_paths[idx]  # image

        image_name = os.path.split(B_path)[-1].replace(".png", "")

        A = Image.open(A_path).convert('LA') if os.path.exists(A_path) else None  # condition
        B = Image.open(B_path).convert('LA') if os.path.exists(B_path) else None  # image

        if self.split == "test" and A is None:
            A = B.copy()  # this will simply be an invalid image

            # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1), normalize=False)
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)*255 if A is not None else None
        B = B_transform(B) if B is not None else None

        return {'Image': B,
                'Label': A,
                'ImageName': image_name,
                'PositiveWeights': 1,
                'Augmentations': ""}

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        # module_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        # module_specific_args.add_argument('--load_size', default=256, type=int)
        # module_specific_args.add_argument('--crop_size', default=256, type=int)

        return parser