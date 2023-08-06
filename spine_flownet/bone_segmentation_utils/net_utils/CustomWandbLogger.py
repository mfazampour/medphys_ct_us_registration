from pytorch_lightning.loggers import WandbLogger
from net_utils.utils import tensor2im, save_data
import numpy as np
import wandb
import os


class Visualizer:
    def __init__(self, queue_len):
        self.queue_len = queue_len
        self.image_queue = dict()
        self.epoch = 0

    @staticmethod
    def reshape_rgb_images(image):
        # We want to go from C, H, W and we want to bring it to H, W, C to be compatible with plotly + have it between
        image = np.transpose(image, [1, 2, 0])
        # image = int(image + 1 / 2.0 * 255.0)
        return image

    @staticmethod
    def flat_channel_as_images(image_list):
        modified_image_list = []
        for item in image_list:
            modified_image_list.extend(np.dsplit(item, item.shape[0]))
        return modified_image_list

    def update_image_queue(self, image_dict, epoch):
        assert isinstance(image_dict, dict), "image dict must be a dictionary"

        if epoch != self.epoch:
            self.epoch = epoch
            self.clean_visuals_queue()

        queue_full = False

        for key in image_dict.keys():

            images_list = tensor2im(image_dict[key])
            if len(images_list[0].shape) > 2 and images_list[0].shape[0] == 3:
                images_list = [self.reshape_rgb_images(item) for item in images_list]

            if len(images_list[0].shape) > 2 and images_list[0].shape[0] != 1 and images_list[0].shape[0] != 3:
                images_list = self.flat_channel_as_images(images_list)

            if key not in self.image_queue.keys():
                self.image_queue[key] = images_list
            else:
                self.image_queue[key].extend(images_list)

            if len(self.image_queue[key]) > self.queue_len:
                self.image_queue[key] = self.image_queue[key][0:self.queue_len]
                queue_full = True

        return queue_full

    def clean_visuals_queue(self):
        self.image_queue = dict()

    def direct_plot(self, image_dict, epoch, split_batches = False):
        image_dct = dict()

        image_keys = image_dict.keys()
        batch_size = 2  # todo: fix this hard coded

        for key in image_dict.keys():
            images_list = tensor2im(image_dict[key])

            # todo: check this
            # if len(images_list[0].shape) > 2 and images_list[0].shape[0] == 3:
            #     images_list = [self.reshape_rgb_images(item) for item in images_list]
            #
            # if len(images_list[0].shape) > 2 and images_list[0].shape[0] != 1 and images_list[0].shape[0] != 3:
            #     images_list = self.flat_channel_as_images(images_list)

            image_dct[key] = images_list

        return self.stack_images(image_dct)


    @staticmethod
    def pad_image(image, padding):

        if len(image.shape) == 2:
            return np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=1)
        else:
            return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=1)

    def stack_image_queue(self, images_keys=None):
        return self.stack_images(self.image_queue, images_keys)

    def stack_images(self, image_dict, images_keys=None):

        image_rows = []
        title = ''
        if images_keys is None:
            images_keys = image_dict.keys()

        for key in image_dict.keys():
            if key not in images_keys:
                continue
            padded_images = [self.pad_image(item, padding=5) for item in image_dict[key]]
            image_rows.append(np.concatenate(padded_images, axis=0))

            title += " " + key

        concatenated_image = np.concatenate(image_rows, axis=1)
        return concatenated_image, title


class CustomWandbLogger(WandbLogger):
    def __init__(self, hparams, queue_len=10):
        super().__init__(project=hparams.project_name, group=hparams.group_name, job_type='train')
        self.train_visualizer = Visualizer(queue_len)
        self.val_visualizer = Visualizer(queue_len)

        self.update_image_queue = self._update_image_queue
        self.experiment.config.update(hparams)

    def _update_image_queue(self, image_dict, epoch, phase):

        if phase == 'train':
            return self.train_visualizer.update_image_queue(image_dict, epoch)
        elif phase == 'val':
            return self.val_visualizer.update_image_queue(image_dict, epoch)
        else:
            raise ValueError("Unknown phase")

    def direct_plot(self, image_dict, epoch, phase, title=''):
        if phase == 'train':
            concatenated_image, title_img = self.train_visualizer.direct_plot(image_dict, epoch, split_batches=False)
        elif phase == 'val':
            concatenated_image, title_img = self.val_visualizer.direct_plot(image_dict, epoch, split_batches=False)
        else:
            raise ValueError("Unknown phase")

        self.experiment.log({title + title_img: wandb.Image(concatenated_image)})

    def log_image_queue(self, phase, title='', images_keys=None):

        if phase == 'train':
            concatenated_image, title_img = self.train_visualizer.stack_image_queue(images_keys)
        elif phase == 'val':
            concatenated_image, title_img = self.val_visualizer.stack_image_queue(images_keys)

        self.experiment.log({title + title_img: wandb.Image(concatenated_image)})
