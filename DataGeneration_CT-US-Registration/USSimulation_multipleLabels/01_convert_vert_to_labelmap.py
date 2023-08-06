import os
import argparse
import glob
import sys

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate strings in between vertebrae for spine deformation")

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
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )
    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    print("Converting deformed vertebrae from obj format to labelmaps")

    # iterate over the txt file
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToFile', 'PathToSave', 'InsideValue']
    for spine in spine_ids:
        for verLev in range(20,25):
            vert = spine + "_verLev" + str(verLev)
            print("Processing: " + str(vert))
            look_for = "**/*" + str(vert) + "*deformed*" + "*centered*" + '*.obj'
            filenames = sorted(glob.glob(os.path.join(args.root_path_vertebrae, look_for), recursive=True))
            if (len(filenames) != int(args.nr_deform_per_spine)):
                print("No deformed files or more than number of deformations could be found for " + str(vert),
                      file=sys.stderr)
                continue

            for deform in range(int(args.nr_deform_per_spine)):
                arguments = ""
                # call imfusion console with the correct parameters
                arguments += placeholders[0] + "=" + filenames[deform] + " "
                arguments += placeholders[1] + "=" + filenames[deform].replace(".obj", ".nii.gz") + " "
                arguments += placeholders[2] + "=" + str(verLev-7)
                os.system("ImFusionConsole" + " " + args.workspace_file + " " + arguments)
