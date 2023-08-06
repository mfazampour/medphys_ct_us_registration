import os
import argparse
import glob
import sys

def merge_obj_files(look_for, save_to, root_path_vertebrae,placeholders):
    lumbar_vertebrae =["verLev20","verLev21","verLev22","verLev23","verLev24"]

    filenames = sorted(
        glob.glob(os.path.join(root_path_vertebrae, look_for), recursive=True))

    filenames_lumbar = [filename for filename in filenames if any([lumb in os.path.basename(filename) for lumb in lumbar_vertebrae]) and 'scaled' not in os.path.basename(filename)]

    if (len(filenames_lumbar) != 5):
        print("More or less than 5 vertebrae were found for " + str(spine_id),
              file=sys.stderr)
        return
    arguments = ""
    for vert_lev in range(5):
        arguments += placeholders[vert_lev] + "=" + filenames_lumbar[vert_lev] + " "

    # add the output path as parameter
    arguments += placeholders[5] + "=" + save_to
    print(arguments)
    # call imfusion console
    os.system("ImFusionConsole" + " " + args.workspace_file + " " + arguments)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Merge multiple vertebrae meshes into one spine mesh")

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
        help="ImFusion workspace files that has all of the necessary algo to merge multiple object files into one"
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
        help="Root path to the spines folders."
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    args = arg_parser.parse_args()
    # iterate over spine IDS
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    placeholders = ['PathToL20', 'PathToL21', 'PathToL22', 'PathToL23', 'PathToL24', 'PathToSave']
    for spine_id in spine_ids:
        # merge the undeformed vertebrae into an obj file which contains the lumbar spine
        print("Merging original undeformed vertebrae of spine: " + str(spine_id) )
        # find original vertebrae as obj files
        merge_obj_files(
            look_for="**/*" + str(spine_id) + '*_msh.obj',
            save_to= os.path.join(args.root_path_spines,spine_id,spine_id  + "_lumbar_msh.obj"),
            root_path_vertebrae=args.root_path_vertebrae,
            placeholders=placeholders
        )

        for deform in range(int(args.nr_deform_per_spine)):
            # for each deformation merge the deformed vertebrae into one obj file
            print("Merging: " + str(spine_id) + " deformation: " +  str(deform))
            merge_obj_files(
                look_for= "**/*" + str(spine_id) + "*forces*" + str(deform) + "*deformed_20*" + '*.obj',
                save_to=os.path.join(args.root_path_spines,spine_id,spine_id + "forcefield" + str(deform) +"_lumbar_deformed.obj"),
                root_path_vertebrae=args.root_path_vertebrae,
                placeholders=placeholders
            )
