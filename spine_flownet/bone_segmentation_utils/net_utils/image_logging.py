import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from PIL import Image

def image_with_colorbar(fig, ax, image, cmap=None, title="", clim=None):

    if clim is None:
        clim = (np.min(image), np.max(image))

    if cmap is None:
        pos0 = ax.imshow(image, clim=clim)
    else:
        pos0 = ax.imshow(image, cmap=cmap, clim=clim)
    ax.set_axis_off()
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax0 = divider.append_axes("right", size="5%", pad=0.05)

    tick_list = np.linspace(clim[0], clim[1], 5)
    tick_labels = ["{:.2f}".format(item) for item in tick_list]
    cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
    cbar0.ax.set_yticklabels(tick_labels)  # vertically oriented colorbar

def log_images(epoch, batch_idx, image_list, image_name_list, cmap_list='gray', filename='', phase='', clim=None):
    """
    Example:
    image_list = [image_batch, label_batch, prediction_batch]
    label_list = ['input images', 'labels', 'prediction']
    """

    if isinstance(cmap_list, str):
        cmap_list = [cmap_list for _ in image_list]

    if isinstance(filename, list):
        filename = [filename for _ in image_list]

    batch_size = image_list[0].size(0)
    fig_list = []
    title_list = []

    for i in range(batch_size):

        fig, axs = plt.subplots(1, len(image_list))
        plot_title = f'{phase} Epoch: {epoch}, Batch: {batch_idx}, filename: {filename[i]}'

        for image_batch, image_name, camp, ax in zip(image_list, image_name_list, cmap_list, axs):
            np_image = np.squeeze(image_batch.to("cpu").numpy()[i, :, :, :])

            if np_image.shape[0] == 3:
                image_with_colorbar(fig, ax, np.rollaxis(np_image, 0, 3), cmap=None, title=image_name)
            else:
                image_with_colorbar(fig, ax, np_image, cmap=camp, title=image_name, clim=clim)

        fig.suptitle(plot_title, fontsize=16)

        fig.tight_layout()
        fig_list.append(fig)
        title_list.append(plot_title)

    return fig_list, title_list

def save_test_image(fake_images, filenames, savepath):

    images = np.squeeze(fake_images.to("cpu").numpy(), axis=1)

    batch_size = images.shape[0]
    for i in range(batch_size):
        filename = filenames[i] + ".png"
        image = np.squeeze(images[i])

        cropped_image = image[0: -9, 24:-24]

        cropped_image = cropped_image + np.min(cropped_image)
        cropped_image = cropped_image/np.max(cropped_image) * 255
        cropped_image = cropped_image.astype(np.uint8)

        image_filepath = os.path.join(savepath, filename)

        im = Image.fromarray(cropped_image)
        im.save(image_filepath)


