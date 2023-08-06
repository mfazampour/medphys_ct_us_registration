import argparse
import os
import glob
import sys
import nibabel as nib
import numpy as np

def transform_pixels_to_mm(coord_in_voxels, volume_size):
    # assume that the label map is situated exactly in the center of the coordinate system
    # as a consequence of having centered the deformed lumbar spine mesh

    coord_in_mm = []
    for i in range(3):
        coord_in_mm.append(coord_in_voxels[i] - 1/2 * volume_size[i])

    return coord_in_mm

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generating transducer splines and direction splines for each "
                                                     "deformed spine. These are further used in the ultrasound "
                                                     "simulation")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--root_path_spines",
        required=True,
        dest="root_path_spines",
        help="Root path to the spine folders."
    )

    arg_parser.add_argument(
        "--path_splinedata",
        required=True,
        dest="splinedata",
        help="Path to csv file that will contain contains the transducer spline and the direction spline for every labelmap."
    )
    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    print("Generating transducer splines and direction splines for each deformed spine. These are further used in the ultrasound simulation")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    # open the csv for writing
    csv_file = open(args.splinedata,'w')
    # add the first row
    csv_file.write("Name;TransdSpline;DirSpline;\n")

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):
            look_for = "**/*" + str(spine_id) + "*forces" + str(deform) + "*deformed*" +'*centered*' '*.nii.gz'
            filenames = sorted(
                glob.glob(os.path.join(args.root_path_spines, look_for), recursive=True))
            if (len(filenames) != 1):
                print(str(len(filenames)) + " spines found for " + str(spine_id) + " and deform " + str(deform),
                      file=sys.stderr)
                continue

            # open with nii gz
            vol = nib.load(filenames[0])
            data = vol.get_fdata()
            shape = data.shape
            rotation = (vol.affine)[:3,:3]
            translation = (vol.affine)[:3,3]

            transd_spline_in_mm = []

            transd_spline_start_point_in_voxels = [shape[0]/2,shape[1]-2,shape[2]-2]
            transd_spline_end_point_in_voxels = [shape[0]/2, shape[1]-2, 2]
            transd_spline_in_mm.extend(transform_pixels_to_mm(transd_spline_start_point_in_voxels,shape))
            transd_spline_in_mm.extend(transform_pixels_to_mm(transd_spline_end_point_in_voxels,shape))

            # dir_spline in voxels is also correct
            dir_spline_in_mm = []
            dir_spline_start_point_in_voxels = [shape[0]/2, 2, shape[2]-2]
            dir_spline_end_point_in_voxels = [shape[0]/2, 2, 2]
            dir_spline_in_mm.extend(transform_pixels_to_mm(dir_spline_start_point_in_voxels,shape))
            dir_spline_in_mm.extend(transform_pixels_to_mm(dir_spline_end_point_in_voxels,shape))

            row = ""
            row += spine_id + "forcefield" + str(deform) + ";"

            for individ_coord in transd_spline_in_mm:
                row += str(individ_coord) + " "
            row+= ";"

            for individ_coord in dir_spline_in_mm:
                row += str(individ_coord) + " "
            row += ";\n"
            print(row)
            # save row in a csv
            csv_file.write(row)





