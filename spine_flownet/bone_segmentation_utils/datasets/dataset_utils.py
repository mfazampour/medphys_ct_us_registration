from PIL import Image
import torchvision.transforms as transforms
import os
import random
import numpy as np
import torch


# todo: convert all the __ method to work with tensors too

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def __make_power_2(img, base, method=Image.BICUBIC):

    if isinstance(img, torch.Tensor):
        ow, oh = img.size(-2), img.size(-1)
    else:
        ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img

def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)

def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))

def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, num_channels=3, normalize=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
        num_channels = 1

    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    elif 'scale_shortside' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))

    if 'zoom' in opt.preprocess:
        if params is None:
            transform_list.append(
                transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method)))
        else:
            transform_list.append(transforms.Lambda(
                lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if 'patch' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))

    if 'trim' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))

    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [
            transforms.Normalize([0.5 for _ in range(num_channels)], [0.5 for _ in range(num_channels)])]
    return transforms.Compose(transform_list)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir_path, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def get_id_from_filename(filename):
    """
    Assuming that the filename has the form subjectId_imageId.fmt, it extracts and reutnrs the subjectId
    Args:
        filename:

    Returns:
    """

    # if the input id a file path, only keep the filename
    filename = os.path.split(filename)[-1]
    sub_id = filename.split("_")[0]
    return sub_id


def get_subject_based_random_split(subject_ids, split_percentages=(80, 10, 10)):
    """
    Considering an input data_list where the data are saved as subjectId_imageId.fmt, it generates train and validation
    folders with a subject based split in a random way
    Args:
        full_data_list:
        split_percentage: it is "trainPerc_valPerc_testPerc" or "trainPerc_valPerc"

    Returns:
    """

    assert sum(split_percentages) == 100, "Only full db subjects usage is supported"

    splits = ['train', 'val', 'test']
    ordered_splits = np.argsort(split_percentages)
    dataset_size = len(subject_ids)

    if len(split_percentages) == 2:
        split_percentages = (split_percentages[0], split_percentages[1], 0)

    split_dict = dict()
    for i in ordered_splits:
        split_percentage = float(split_percentages[i]) / 100
        n_split_subjects = min(round(split_percentage * dataset_size), len(subject_ids))

        split_subjects = random.sample(subject_ids, n_split_subjects)
        subject_ids = [item for item in subject_ids if item not in split_subjects]
        split_dict[splits[i]] = split_subjects

    if len(subject_ids) > 0:
        split_dict[splits[i]].extend(subject_ids)

    return split_dict['train'], split_dict['val'], split_dict['test']


def get_subject_ids_from_data(full_data_list):
    full_subject_ids = [get_id_from_filename(item) for item in full_data_list]
    subject_ids = list(set(full_subject_ids))
    return subject_ids


def get_split_subjects_data(data_list, split_subject_ids):
    """
    Given a split (train, test, val) if the config contains the IDs of the subjects in that split then the function
    only keeps the files in the self.dir_AB that contain the specific subject
    Returns:
    """
    split_data_list = [item for item in data_list if get_id_from_filename(item) in split_subject_ids]
    return split_data_list
