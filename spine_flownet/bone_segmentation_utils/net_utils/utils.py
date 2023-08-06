# from importlib import util
from pydoc import locate
import inspect
import argparse
import numpy as np
from PIL import Image
import torch


def argparse_summary(arg_list, parser):
    arg_dict = vars(arg_list)
    action_groups_dict = {}
    for i in range(len(parser._action_groups)):
        action_groups_dict[parser._action_groups[i].title]=[]
    for j in parser._actions:
        if j.dest == "help":
            continue
        try:
            action_groups_dict[j.container.title].append((j.dest, arg_dict[j.dest]))
        except:
            print(f"not working: {j.dest}")

    value = "########################ArgParseSummaryStart########################"
    len_group_var = 55
    for k in parser._action_groups:
        group = k.title
        length_filler = len_group_var-len(group)
        length_filler1 = length_filler-(length_filler//2)
        length_filler2 = length_filler-length_filler1
        value+= f"\n{''.join(['-']*length_filler1)}{group}{''.join(['-']*length_filler2)}"
        for l in action_groups_dict[group]:
            value += "\n  {0:<25s}: {1:21s}  ".format(l[0], str(l[1]))
    value += "\n########################ArgParseSummaryEnd########################"
    print(value)


def get_argparser_group(title, parser):
    for group in parser._action_groups:
        if title == group.title:
            return group
    return None


def get_class_by_path(dot_path=None):
    if dot_path:
        MyClass = locate(dot_path)
        assert inspect.isclass(MyClass), f"Could not load {dot_path}"
        return MyClass
    else:
        return None


def get_function_by_path(dot_path=None):
    if dot_path:
        myfunction = locate(dot_path)
        assert inspect.isfunction(myfunction), f"Could not load {dot_path}"
        return myfunction
    else:
        return None


def get_model_by_function_path(hparams):
    model_constructor = get_function_by_path("models." + hparams.model)
    model = model_constructor(hparams)
    return model


def get_model_by_class_path(hparams):
    ModelClass = get_class_by_path("models." + hparams.model)
    model = ModelClass(hparams)
    return model


def get_dataset_by_class_path(hparams):
    DatasetClass = get_class_by_path("datasets." + hparams.dataset)
    dataset = DatasetClass(hparams)
    return dataset

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tensor2np_array(input_tensor):
    np_array_batch = np.squeeze(input_tensor.cpu().numpy(), axis=1)

    output_data = []
    for i in range(np_array_batch.shape[0]):
        np_array = np.squeeze(np_array_batch[i, ...])
        output_data.append(np_array)

    return output_data

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if isinstance(input_image, np.ndarray):
        return [input_image.astype(imtype)]

    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
    else:
        return input_image

    image_list = []
    for i in range(image_tensor.size(0)):
        for j in range(image_tensor.size(1)):

            image_numpy = image_tensor[i, j, ...].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
            image_numpy = np.expand_dims(image_numpy, 0)
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

            image_list.append(image_numpy.astype(imtype))

    return image_list





def save_data(data, filename, fmt='npy', is_label=False):

    if "." in filename:
        [filename, fmt] = filename.split(".")

    if fmt == 'npy':
        np.save(filename + "." + fmt, data)

    elif fmt == 'png':

        if len(data.shape) == 3 and data.shape[0] <= 3:
            data = np.transpose(data, [1, 2, 0])
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = np.squeeze(data)

        rescaled_image = data - np.min(data)
        rescaled_image = rescaled_image / np.max(rescaled_image) * 255

        if is_label:
            rescaled_image = np.where(rescaled_image > 1, 255, 0)

        rescaled_image = rescaled_image.astype(np.uint8)

        pil_image = Image.fromarray(rescaled_image)
        pil_image.save(filename + "." + fmt)


