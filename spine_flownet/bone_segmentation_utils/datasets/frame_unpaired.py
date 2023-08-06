from datasets.base_dataset import BaseUnpairedDataset
from PIL import Image
import random
from datasets.dataset_utils import get_params, get_transform

class FrameUnpaired(BaseUnpairedDataset):
    def __init__(self, hparams, split=None, **kwargs):
        super().__init__(hparams, split, **kwargs)

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[idx % self.A_size]  # make sure index is within then range
        if self.hparams.serial_batches:   # make sure index is within then range
            index_B = idx % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A_img.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        # apply image transformation
        A = A_transform(A_img)
        B = B_transform(B_img)

        return {'Image': B,
                'Label': A}
