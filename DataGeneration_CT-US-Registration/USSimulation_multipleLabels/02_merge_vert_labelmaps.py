import os
import argparse
import glob
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

def get_name_labelmap_deformed_vertebra(spine_id,verLev,deform):
    return spine_id + "_verLev" + str(verLev) + "_forces" + str(deform) + "_deformed_centered_20_0.nii.gz"

def get_vertebrae_labelmaps_deformed_centered_paths(vertebrae_dir, spine_id,deform):
    vertebrae_labelmaps = []

    # get all 5 vertebrae paths for one spine id and one deformation
    for i in range(20,25):
        curr_vert_path = os.path.join(vertebrae_dir,spine_id + "_verLev" + str(i),get_name_labelmap_deformed_vertebra(spine_id, i, deform ))
        if(not Path(curr_vert_path).is_file()):
            print("No deformed vertebra labelmap found for spine %s, vert %s and deform %d" % (spine_id, str(i), deform), file=sys.stderr)
        vertebrae_labelmaps.append(curr_vert_path)
    return vertebrae_labelmaps

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate strings in between vertebrae for spine deformation")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar spines"
    )

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )

    arg_parser.add_argument(
        "--root_path_spines",
        required=True,
        dest="root_path_spines",
        help="Root path to the spine folders."
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    arg_parser.add_argument(
        "--workspace_file_merge_labelmaps",
        required=True,
        dest="workspace_file_merge_labelmaps",
        help="ImFusion workspace file that merges multiple vertebrae labelmaps into one"
    )

    args = arg_parser.parse_args()

    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]
    placeholders = ['PathTo20', 'PathTo21', 'PathTo22', 'PathTo23', 'PathTo24', 'PathToSave']

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):
            vertebrae_labelmaps = get_vertebrae_labelmaps_deformed_centered_paths(vertebrae_dir=args.root_path_vertebrae,spine_id=spine_id, deform=deform )
            arguments = ""
            for vert_lev in range(5):
                arguments += placeholders[vert_lev] + "=" + vertebrae_labelmaps[vert_lev] + " "
            path_combined_labelmap = os.path.join(args.root_path_spines,spine_id,os.path.basename(vertebrae_labelmaps[0]).replace("_verLev20",""))
            arguments += placeholders[5] + "=" + path_combined_labelmap
            print(arguments)
            os.system("ImFusionConsole" + " " + args.workspace_file_merge_labelmaps + " " + arguments)
