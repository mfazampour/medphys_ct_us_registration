import subprocess
import argparse
import os

if __name__ == '__main__':
    """
    # START HERE 
    Pipeline for ultrasound generation, partial point cloud extraction and dataset creation for shape completion
    From this pipeline we obtain partial point clouds that contain points from the current vertebra as well as points 
    from the neighboring ones. The purpose is to train a robust model against segmentation errors of bone in ultrasound
    which is a common issue.
    1. Create labelmaps from the deformed spines which are in form of mesh files
    2. Automatically generate transducer splines and direction splines for each deformed spine. 
    These are further used in the ultrasound simulation --> 02_generate_splines.py
    3. Use imfusion workspace file to simulate ultrasound and segment the bone -->  simulate_US_and_save_segmentations.py
        Steps followed in the IWS file simulate_US_and_save_segmentations.iws
        1.1 Load labelmap from deformed spine 
        1.2 Simulate ultrasound sweep 
        1.3 Resample sweep to 1x1x1 mm
        1.4 Create US segmentations through volume reslicing (from US and initial labelmap) and export them
    4. Raycast the obtained segmentations (otherwise parts of vert that are not visible in the ultrasound will appear in the 
    segmentation --> 04_raycast_bmode_data.py
    # TODO add: 
    5. Unet call --> segmented US scans 
    6. Extract PCD 
    
    # END HERE 
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
    splinedata = os.path.join(root_path_vertebrae, "splines.csv")
    nr_deform_per_spine = args.nr_deform_per_spine

    workspace_file_obj_to_labelmap = "imfusion_workspaces/objToLabelMap_with_centered.iws"
    workspace_file_simulate_us = "imfusion_workspaces/simulate_US_and_save_segmentations_noisypcd.iws"
    workspace_file_extract_pointcloud = "imfusion_workspaces/extract_pcd_from_US_labelmaps.iws"

    pipeline = args.pipeline

    if 'convert_obj_to_labelmaps' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '01_convert_spines_to_labelmap.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', workspace_file_obj_to_labelmap,
                        '--root_path_spine', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine
                        ])

    if 'generate_splines' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '02_generate_splines.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_spines', root_path_spines,
                        '--path_splinedata', splinedata,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'simulate_US' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '03_simulate_lumbar_spine_ultrasound.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', workspace_file_simulate_us,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine,
                        '--path_splinedata', splinedata])

    if 'raycast' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '04_raycast_bmode_data.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'extract_pcd' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '05_extract_pcd_from_US_labelmaps.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', workspace_file_extract_pointcloud,
                        '--root_path_spines', root_path_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine])

