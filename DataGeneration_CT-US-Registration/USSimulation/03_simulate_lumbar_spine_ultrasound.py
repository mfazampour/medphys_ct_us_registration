import argparse
import os
import glob
import sys

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Simulating ultrasound images from label maps of deformed lumbar spine")

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--workspace_file",
        required=True,
        dest="workspace_file",
        help="ImFusion workspace files that has all of the necessary algo info to transform from object to labelmap"
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
        "--path_splinedata",
        required=True,
        dest="splinedata",
        help="Path to csv file that contains the transducer spline and the direction spline for every labelmap."
    )

    args = arg_parser.parse_args()
    print("Simulating ultrasound images from label maps of deformed lumbar spine")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToFile', 'TransdSpline', 'DirSpline', 'PathToSaveTrackingStream', 'PathToSaveUS', 'PathToSaveLabels']

    # open the csv file
    with open(args.splinedata, "r") as f:
        lines = f.readlines()

    # create 2 lists, one of transducer splines and one of direction splines
    splines = {}
    for l in lines[1:]:
        # a line looks like this: Name;TransdSpline;DirSpline
        splines[l.split(";")[0]] = [[l.split(";")[1]], [l.split(";")[2]]]

    for spine_id in spine_ids:
        for deform in range(int(args.nr_deform_per_spine)):
            print("Simulating ultrasound for: " + str(spine_id) + "and deformation" + str(deform))
            look_for = "**/*" + str(spine_id) + "*forcefield" + str(deform) + "*deformed*" '*centered*'+ '*.nii.gz'
            filenames = sorted(
                glob.glob(os.path.join(args.root_path_spines, look_for), recursive=True))
            if (len(filenames) != 1):
                print("More or less than 1 spine found for " + str(spine_id),
                      file=sys.stderr)
                continue

            # read the transducer spline and the direction spline from the csv file
            list_at_curr_key = splines[spine_id + "forcefield" + str(deform)]

            # create arguments list to call ImFusion with
            arguments = ""
            dir_name = os.path.dirname(filenames[0])
            for p in placeholders:
                if p == 'PathToFile':
                    value = filenames[0]
                elif p == 'TransdSpline':
                    value = list_at_curr_key[0]
                elif p == 'DirSpline':
                    value = list_at_curr_key[1]
                elif p == 'PathToSaveTrackingStream':
                    value = os.path.join(dir_name, "tracking_force" + str(deform) + ".ts")
                elif p == 'PathToSaveUS':
                    save_us_to = os.path.join(dir_name, "ultrasound_force" + str(deform))
                    if not os.path.exists(save_us_to):
                        os.mkdir(save_us_to)
                    value = save_us_to
                elif p == 'PathToSaveLabels':
                    save_labels_to = os.path.join(dir_name, "labels_force" + str(deform))
                    if not os.path.exists(save_labels_to):
                        os.mkdir(save_labels_to)
                    value = save_labels_to

                arguments += p + "=" + str(value) + " "
            print('ARGUMENTS: ', arguments)
            os.system("ImFusionSuite" + " " + args.workspace_file + " " + arguments)
            print('################################################### ')
