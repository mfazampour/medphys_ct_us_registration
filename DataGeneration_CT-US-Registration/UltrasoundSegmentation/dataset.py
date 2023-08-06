import os

import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImage, AddChannel
import torchio as tio


def create_subjects(img_path, label_path, field_num, patient_id):
    return tio.Subject(image=tio.ScalarImage(img_path), label=tio.LabelMap(label_path),
                       field_num=field_num, patient_id=patient_id)


def create_dataset(root_path_spine, txt_file):

    # Create a list of all the image and label paths
    subjects_train = []
    subjects_val = []

    # Read the lines from the text file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Process each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        if np.random.rand(1) > 0.8:
            mode = 'val'
        else:
            mode = 'train'
        patient_dir = os.path.join(root_path_spine, f"sub-{line}/")
        for field_num in range(10):
            field_dir = os.path.join(patient_dir, f"sub-{line}forcefield{field_num}_us_set/")
            label_dir = os.path.join(patient_dir, f"labels_force{field_num}/raycasted/")
            if not os.path.exists(field_dir) or not os.path.exists(label_dir):
                continue
            if os.listdir(label_dir).__len__() == 0:
                continue
            image_path = os.path.join(field_dir, f"Ultrasound.png")
            label_path = os.path.join(label_dir, f"Images_raycasted.png")
            if mode == 'train':
                subjects_train.append(create_subjects(image_path, label_path, field_num, line))
            else:
                subjects_val.append(create_subjects(image_path, label_path, field_num, line))
            for i in range(1, 50):
                subject = create_subjects(os.path.join(field_dir, f"Ultrasound{format(i, '02d')}.png"),
                                                      os.path.join(label_dir, f"Images_raycasted{format(i, '02d')}.png"),
                                                      field_num, line)
                if mode == 'train':
                    subjects_train.append(subject)
                else:
                    subjects_val.append(subject)


    # Create the dataset
    transforms_train = [
        tio.CropOrPad((512, 512, 1)),
        tio.Lambda(lambda x: x != 0, types_to_apply=[tio.LABEL]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.RandomAffine(),
    ]

    transforms_val = [
        tio.CropOrPad((512, 512, 1)),
        tio.Lambda(lambda x: x != 0, types_to_apply=[tio.LABEL]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ]

    transform = tio.Compose(transforms_train)
    dataset_train = tio.SubjectsDataset(subjects_train, transform=transform)

    transform = tio.Compose(transforms_val)
    dataset_val = tio.SubjectsDataset(subjects_val, transform=transform)

    return dataset_train, dataset_val
