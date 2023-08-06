import pytorch_lightning as pl
from torch import optim
from pytorch_lightning.metrics import Accuracy
from net_utils.utils import get_argparser_group
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from mpl_toolkits.axes_grid1 import make_axes_locatable
from net_utils.utils import str2bool
import pandas as pd
EPS = 1e-5

# TODO: modify probabilities and how they are selected

class BoneSegmentation(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):
        super(BoneSegmentation, self).__init__()
        self.hparams = hparams
        self.model = model
        self.example_input_array = torch.zeros(1, 1, 196, 196)
        self.accuracy = Accuracy()

        self.t_logger = logger[0] if logger is not None else None  # setting the tensorboard logger
        self.counter = 0
        self.dice_loss = DiceLoss()

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def dice_score(gt, predictions, prob=0.5):

        predictions[predictions >= prob] = 1
        predictions[predictions < prob] = 0

        intersection = predictions[torch.where(gt > 0)]

        numerator = 2 * torch.count_nonzero(intersection)
        denominator = torch.count_nonzero(gt) + torch.count_nonzero(predictions)

        dice = numerator / denominator

        return dice

    def compute_loss(self, y_pred, y_true, pos_weights):

        if self.hparams.use_positive_weights:
            avg_pos_weights = torch.mean(pos_weights)
            bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=avg_pos_weights)
        else:
            bce_criterion = torch.nn.BCEWithLogitsLoss()

        if self.hparams.loss_function == 'BCE':
            bce_loss = bce_criterion(y_pred, y_true)

            return {'total_loss': bce_loss}
        else:
            dice_loss = self.dice_loss(y_pred, y_true)
            bce_loss = bce_criterion(y_pred, y_true)
            total_loss = dice_loss + bce_loss
            return {'total_loss': total_loss, 'dice_loss':dice_loss, 'bce_loss': bce_loss}

    def training_step(self, train_batch, batch_idx):

        x = train_batch['Image']
        y_true = train_batch['Label']
        filename = train_batch['ImageName']
        augmentations = train_batch['Augmentations']
        pos_weights = train_batch['PositiveWeights']

        if self.hparams.input_type=='Float':
            x = x.float()

        y_pred = self.forward(x)

        train_losses = self.compute_loss(y_pred, y_true, pos_weights)
        train_loss = train_losses['total_loss']

        if self.hparams.loss_function == 'BCE+Dice':
            self.log('Train bce loss', train_losses['bce_loss'])
            self.log('Train dice loss', train_losses['dice_loss'])

        if batch_idx % 300 == 0:
            sigmoid_pred = torch.sigmoid(y_pred)
            self.log_images(x, y_true, sigmoid_pred.detach(), self.current_epoch, batch_idx, filename,
                            'train', augmentations)

        return {'loss': train_loss}

    def training_step_end(self, train_step_output):
        self.log('Train Total Loss', train_step_output['loss'], on_step=True, on_epoch=True)

        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):

        x = val_batch['Image']
        y_true = val_batch['Label']
        filename = val_batch['ImageName']
        pos_weights = val_batch['PositiveWeights']
        augmentation = val_batch['Augmentations']

        if self.hparams.input_type=='Float':
            x = x.float()

        y_pred = self.forward(x)
        val_losses = self.compute_loss(y_pred, y_true, pos_weights)
        val_loss = val_losses['total_loss']

        if batch_idx % 50 == 0:
            sigmoid_pred = torch.sigmoid(y_pred)
            self.log_images(x, y_true, sigmoid_pred.detach(), self.current_epoch, batch_idx, filename,
                            'val', augmentation)

        return {'val_loss_step': val_loss}

    def validation_step_end(self, val_step_output):
        self.log('Validation DICE Loss', val_step_output['val_loss_step'], on_step=True, on_epoch=True)
        return {'val_loss_step': val_step_output['val_loss_step']}

    def validation_epoch_end(self, val_step_output):
        val_loss_list = [item['val_loss_step'] for item in val_step_output]
        val_loss = torch.mean(torch.stack(val_loss_list))
        return {'val_loss': val_loss}

    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):

        if batch_idx == 166:
            print()

        x = test_batch['Image']
        y_true = test_batch['Label']
        filename = test_batch['ImageName']

        if self.hparams.input_type=='Float':
            x = x.float()

        y_pred = self.forward(x)

        test_acc = self.dice_score(gt=y_true,
                                   predictions=y_pred.clone(),
                                   prob=self.hparams.probability_threshold)

        self.save_test_image(test_batch, y_pred, self.hparams.output_path)

        if batch_idx % 30 == 0:
            sigmoid_pred = torch.sigmoid(y_pred)
            self.log_images(x, y_true, sigmoid_pred.detach(), self.current_epoch, batch_idx, filename,
                            'test')

        return {'test_acc': test_acc, 'filename':filename}

    def test_step_end(self, test_step_output):
        # The training losses are accumulated and averaged to give the final, average, test loss
        self.log('Test DICE Loss', test_step_output['test_acc'], on_step=False)
        return test_step_output

    def test_epoch_end(self, test_step_output):

        dice_scores = [item["test_acc"] for item in test_step_output]
        filenames = [item["filename"] for item in test_step_output]

        pd_data = {"filenames": filenames, "dice_scores": dice_scores}
        pd_frame = pd.DataFrame(data=pd_data)
        pd_frame.to_csv(os.path.join(self.hparams.output_path, "test_results.csv"))

    def log_images(self, x, y_true, y_pred, current_epoch, batch_idx, filename=" ", phase='', augmentations=" "):

        if phase == 'train':
            name = f'Train Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]}' \
                   f'(augmentation: {augmentations[0]})'

        elif phase == 'val':
            name = f'Val Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]} ' \
                   f'(augmentation: {augmentations[0]})'

        else:
            name = f'Unknown Phase Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        # Taking the first image, label prediction from the input batch
        image = np.squeeze(x.to("cpu").numpy(), axis=1)[0, :, :]
        label = np.squeeze(y_true.to("cpu").numpy(), axis=1)[0, :, :]
        prediction = np.squeeze(y_pred.to("cpu").numpy(), axis=1)[0, :, :]

        fig, axs = plt.subplots(1, 4, constrained_layout=True)
        pos0 = axs[0].imshow(image, cmap='gray')
        axs[0].set_axis_off()
        axs[0].set_title('Image')
        divider = make_axes_locatable(axs[0])
        cax0 = divider.append_axes("right", size="5%", pad=0.05)
        tick_list = np.linspace(np.min(image), np.max(image), 5)
        cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
        cbar0.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

        pos1 = axs[1].imshow(label)
        axs[1].set_title('Label')
        axs[1].set_axis_off()
        divider = make_axes_locatable(axs[1])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        tick_list = np.linspace(np.min(label), np.max(label), 5)
        cbar1 = fig.colorbar(pos1, cax=cax1, ticks=tick_list, fraction=0.001, pad=0.05)
        cbar1.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

        pos2 = axs[2].imshow(prediction)
        axs[2].set_title('Prediction')
        axs[2].set_axis_off()
        divider = make_axes_locatable(axs[2])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        tick_list = np.linspace(np.min(prediction), np.max(prediction), 5)
        cbar2 = fig.colorbar(pos2, cax=cax2, ticks=tick_list, fraction=0.001, pad=0.05)
        cbar2.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

        pos3 = axs[3].imshow(np.where(prediction>0.5, 1, 0))
        axs[3].set_title('Prediction')
        axs[3].set_axis_off()
        divider = make_axes_locatable(axs[3])
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        tick_list = np.linspace(np.min(prediction), np.max(prediction), 5)
        cbar3 = fig.colorbar(pos3, cax=cax3, ticks=tick_list, fraction=0.001, pad=0.05)
        cbar3.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar
        fig.tight_layout()

        fig.suptitle(name, fontsize=16)
        self.t_logger.experiment.add_figure(tag=name, figure=fig)

    @staticmethod
    def save_test_image(batch, predictions, savepath):

        predictions = np.squeeze(predictions.to("cpu").numpy(), axis=1)
        images = np.squeeze(batch['Image'].to("cpu").numpy(), axis=1)
        labels = np.squeeze(batch['Label'].to("cpu").numpy(), axis=1)

        batch_size = predictions.shape[0]
        for i in range(batch_size):

            filename = batch['ImageName'][i]
            prediction = predictions[i]
            label = labels[i]
            image = images[i]

            spine_id = filename.split("_")[0]

            # spine10_ts_1_0_.png -> 1_0_.png; spine10_ts_1_0_005.png -> 1_0_005.png
            ts_img_id = filename.split("ts_")[-1]

            # 1_0_.png -> .png;spine10_ts_1_0_005.png -> 005.png
            img_id = ts_img_id.split("_")[-1]

            # 1_0_.png > 1_0; 1_0_005.png -> 1_0
            ts = ts_img_id.replace("_" + img_id, "")

            save_folder = os.path.join(savepath, spine_id, ts)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            pred_filepath = os.path.join(save_folder, filename.split(".")[0] + "_pred.npy")
            np.save(pred_filepath, prediction)

            image_filepath = os.path.join(save_folder, filename.split(".")[0] + "_image.npy")
            np.save(image_filepath, image)

            label_filepath = os.path.join(save_folder, filename.split(".")[0] + "_gt.npy")
            np.save(label_filepath, label)

    @staticmethod
    def add_module_specific_args(parser):

        module_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        module_specific_args.add_argument('--in_channels', default=1, type=int)
        module_specific_args.add_argument('--out_channels', default=1, type=int)
        module_specific_args.add_argument('--probability_threshold', default=0.5, type=float)
        module_specific_args.add_argument('--input_type', default='Double', type=str)
        parser.add_argument("--use_positive_weights", type=str2bool, nargs='?', const=True, default=True,
                            help="Activate nice mode.")
        module_specific_args.add_argument('--loss_function', default='BCE', type=str)
        return parser

