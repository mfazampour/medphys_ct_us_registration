from datasets.base_dataset import BaseUnpairedDataset
from random import randint
import SimpleITK as sitk
from datasets.dataset_utils import get_transform
import torchvision.transforms as transforms


class SweepUnpaired(BaseUnpairedDataset):
    def __init__(self, hparams, split, **kwargs):
        super().__init__(hparams, split, **kwargs)
        self.batches_per_sweep = 100
        self.convert_transform = transforms.ToTensor()

        self.A_paths = [item for item in self.A_paths if ".mhd" in item]
        self.B_paths = [item for item in self.B_paths if ".mhd" in item]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __len__(self):
        return self.batches_per_sweep * max(self.A_size, self.B_size)

    def __getitem__(self, idx):

        A_path = self.A_paths[idx % self.A_size]  # make sure index is within then range
        if self.hparams.serial_batches:  # make sure index is within then range
            index_B = idx % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = sitk.GetArrayFromImage(sitk.ReadImage(A_path))
        B_img = sitk.GetArrayFromImage(sitk.ReadImage(B_path))

        A_idx = randint(0, A_img.shape[-1] - self.hparams.input_nc)
        B_idx = randint(0, B_img.shape[-1] - self.hparams.output_nc)

        if A_idx + 5*(self.hparams.input_nc - 1) >= A_img.shape[-1]:
            A_idx = A_img.shape[-1] - 5*(self.hparams.input_nc - 1) - 1

        A_spaced_idx = [A_idx + i*5 for i in range(self.hparams.input_nc)]

        A_img = A_img[..., A_spaced_idx]
        B_img = B_img[..., B_idx:B_idx+self.hparams.output_nc]

        is_finetuning = self.split == 'train' and self.current_epoch > self.hparams.max_epochs
        modified_opt = self.copy_conf(self.hparams,
                                      load_size=self.hparams.crop_size if is_finetuning else self.hparams.load_size)

        A = self.convert_transform(A_img).float()
        B = self.convert_transform(B_img).float()

        transform = get_transform(modified_opt, grayscale=False, convert=False, num_channels=self.hparams.input_nc)
        A = transform(A)
        B = transform(B)

        return {'Image': B,
                'Label': A}



