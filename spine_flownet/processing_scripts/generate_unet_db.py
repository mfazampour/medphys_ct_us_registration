import os
import numpy as np


def get_subject_split(spine_id, training_splits):
    for split in training_splits.keys():
        if spine_id in training_splits[split]:
            return split


def generate_us_labels_batch_file(src_labels_dir, src_ultrasound_dir, output_dir, batch_file_path, training_splits):
    """
    Generate the script to generate the batch file to be used for launching the imfusion_workspaces/us_label_segmentation_data.iws.

    :param: src_labelmaps_dir: str: The path to the labelmaps
    :param: dst_pcs_dir: str: The directory where the point cloud files will be saved
    :param: batch_file_path: str: The path where the (imfusion) .txt batch file will be generated

    Example:

        .. code-block:: text
            INPUTMHD;OUTPUTPC
            <src_labelmaps_dir>\spine<spineId>\raycasted_ts<timestampId>.mhd;<dst_pcs_dir>\spine<spineId>\raycasted_ts<timestampId>.txt
                                        ...

    """
    #assert "train" in training_splits.keys() and "val" in training_splits.keys() and "test" in training_splits.keys()

    for split in [ "test"]:
        if not os.path.exists(os.path.join(output_dir, split)):
            os.makedirs(os.path.join(output_dir, split))

    spine_ids = os.listdir(src_labels_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fid = open(batch_file_path, "w")
    # fid.write("USNAME;LABELNAME;INPUTLABEL;INPUTUS;OUTPUTPATH")
    fid.write("USNAME;INPUTUS;OUTPUTPATH")

    for spine_id in spine_ids:

        # it appeared in my pc i had to hard code it @Jane
        if "spine" not in spine_id:
            continue

        split = get_subject_split(spine_id, training_splits)
        if split is None:
            print("subject not in any training split")
            continue

        save_path = os.path.join(output_dir, split)

        ts_list = [item.split(".")[0].replace("_labelmap", "") for item in
                   os.listdir(os.path.join(src_labels_dir, spine_id))
                   if ".imf" in item]

        for ts in ts_list:

            input_labels_path = os.path.join(src_labels_dir, spine_id, ts + "_labelmap" + ".imf")
            input_us_path = os.path.join(src_ultrasound_dir, spine_id, ts + ".imf")
            image_name = spine_id + "_" + ts + "_"
            label_name = spine_id + "_" + ts + "_label_"

            fid.write("\n" + image_name + ";" + input_us_path + ";" + save_path)

    fid.close()


train_spines = ["spine" + str(i) for i in range(1, 21)]
val_spines = ["spine21"]
test_spines = ["spine" + str(i) for i in range(1, 23)]
generate_us_labels_batch_file(src_labels_dir="E:/NAS/jane_project/simulated_us_labelmaps",
                              src_ultrasound_dir="E:/NAS/jane_project/simulated_us",
                              output_dir="C:/Repo/thesis/bone_segmentation_utils/data",
                              batch_file_path="../imfusion_workspaces/us_label_data.txt",
                              training_splits = {
                                                 "test": test_spines
                                                 })
print("done")
