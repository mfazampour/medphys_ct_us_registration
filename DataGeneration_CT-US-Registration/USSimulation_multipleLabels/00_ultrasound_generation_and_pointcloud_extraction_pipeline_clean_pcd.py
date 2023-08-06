import subprocess
import argparse
import os

if __name__ == '__main__':
    """
    1. Convert vertebral meshes to individual labelmaps 
    2. Merge these labelmaps into one labelmap 
    3. Generate the splines for ultrasound simulation
    4. Simulate Ultrasound, save the labels as well as the tracking data  
    5. Raycast the labels 
    6. Extract PCD from compounded labels 
    """

    arg_parser = argparse.ArgumentParser(description="Pipeline of ultrasound simulation and point cloud extraction")

    arg_parser.add_argument(
        "--root_path_spines",
        required=True,
        dest="root_path_spines",
        help="Root path to the spine folders."
    )

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )

    arg_parser.add_argument(
        "--list_file_names_spines",
        required=True,
        dest="txt_file_spines",
        help="Txt file that contains all spines that contain all lumbar spines"
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformations per spine"
    )

    arg_parser.add_argument(
        "--pipeline",
        nargs='+',
        default=['all'],
        help="Specify the steps of the pipeline that will be executed "
    )

    args = arg_parser.parse_args()

    root_path_spines = args.root_path_spines
    root_path_vertebrae = args.root_path_vertebrae

    txt_file_lumbar_spines = args.txt_file_spines
    path_splinedata = os.path.join(root_path_vertebrae, "splines.csv")
    nr_deform_per_spine = args.nr_deform_per_spine

    workspace_file_simulate_us = "imfusion_workspaces/simulate_US_and_save_segmentations_cleanpcd.iws"
    workspace_file_extract_pointcloud = "imfusion_workspaces/extract_pcd_from_US_labelmaps_cleanpcd.iws"
    workspace_file_obj_to_labelmap = "imfusion_workspaces/objToLabelMap_with_centered.iws"
    workspace_file_merge_labelmaps = "imfusion_workspaces/merge_labelmaps.iws"

    pipeline = args.pipeline

    if 'vert_mesh_to_labelmap' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '01_convert_vert_to_labelmap.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', workspace_file_obj_to_labelmap,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'merge_vert_labelmaps' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '02_merge_vert_labelmaps.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine,
                        '--workspace_file_merge_labelmaps', workspace_file_merge_labelmaps])

    if 'generate_splines' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '03_generate_splines.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_spines', root_path_spines,
                        '--path_splinedata', path_splinedata,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'simulate_US' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '04_simulate_lumbar_spine_ultrasound_cleanpcd.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', workspace_file_simulate_us,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine,
                        '--path_splinedata', path_splinedata])

    if 'raycast' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '05_raycast_bmode_data_multiple_labels.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'extract_pcd' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '06_extract_pcd_from_US_labelmaps_cleanpcd.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file',  workspace_file_extract_pointcloud,
                        '--root_path_spines', root_path_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--nr_deform_per_spine', nr_deform_per_spine])