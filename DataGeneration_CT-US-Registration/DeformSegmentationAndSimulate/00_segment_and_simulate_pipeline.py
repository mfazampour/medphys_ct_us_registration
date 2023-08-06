import subprocess
import argparse
import os

if __name__ == '__main__':
    """
      Prerequisites for this pipeline: 
      1. The folder structure is the following 
      - root_path_spines directory:
          <root_path_spines>/<spine_id>/ folders are already created 

      - root_path_vertebrae:
          <root_path_vertebrae>/<spine_id>/<spine_id>*_msh.obj --> mesh files of individual vertebrae are used for deformation
          --> to separate spine segmentations into vertebrae segmentations and transform segmentation to mesh check 
              - "https://github.com/miruna20/thesis/blob/main/separate_spine_into_vertebrae.py"
              - "https://github.com/miruna20/thesis/blob/main/convert_segmentation_into_mesh.py"
      2. There exists a .txt file containing the verse names of the spines that will be processed
      
      3. The deformation pipeline should already have happened 
      
      4. Folder containing the segmentation output from the Totalsegmentator

      Steps of the pipeline:
      #1. Crop ROI based on the position of Sacrum and T11
      #2. Extrapolate the deformation to the whole image including the soft tissue 
      #3. replace the labels that we have from Totalsegmentator
      #4. simulate ultrasound using imfusion
      #5. make the raycasted images again using imfusion and save the labels for training the u-net for us segmentation 
      #6. extract all of the point clouds for each vertebra -> next is to run "Prepare for Network"
      
      """

    arg_parser = argparse.ArgumentParser(description="Generate full deformed images and simulate ultrasound")

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
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformed spines per initial spine."
    )

    arg_parser.add_argument(
        "--pipeline",
        nargs='+',
        default=['all'],
        help="Specify the steps of the pipeline that will be executed "
    )

    arg_parser.add_argument(
        "--segmentation_folder",
        required=True,
        help="folder containing the output of the total segmentator"
    )

    arg_parser.add_argument(
        "--verse_path",
        required=True,
        help="output folder path for cropped spine folders"
    )

    args = arg_parser.parse_args()

    root_path_spines = args.root_path_spines
    root_path_vertebrae = args.root_path_vertebrae
    txt_file_lumbar_spines = args.txt_file
    nr_deform_per_spine = args.nr_deform_per_spine

    root_folder_json_files = os.path.join(root_path_vertebrae,"spring_files")
    forces_folder = os.path.join(root_path_vertebrae, "forces_folder")

    pipeline = args.pipeline

    if 'crop_roi' in pipeline or 'all' in pipeline:
        subprocess.run(['python', './01_crop_ROI.py',
                        '--root_path_spine', args.verse_path,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--segmentation_folder', args.segmentation_folder,
                        "--out_path_spine", root_path_spines])

    if 'create_deformed_image' in pipeline or 'all' in pipeline:
        subprocess.run(['python', "02_create_deformed_image.py",
                        '--root_path_spine', root_path_spines,
                        '--list_file_names', txt_file_lumbar_spines
                        ])

    if 'ultrasound_labels' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '03_ultrasound_labels.py',
                        '--root_path_spine', root_path_spines,
                        '--list_file_names', txt_file_lumbar_spines
                        ])

    if 'ultrasound_simulation' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '04_ultrasound_simulation.py',
                        '--root_path_spine', root_path_spines,
                        '--list_file_names', txt_file_lumbar_spines
                        ])

    if 'raycast_bmode_data' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '05_1_raycast_bmode_data.py',
                        '--root_path_spines', root_path_spines,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine])

    if 'extract_pcd' in pipeline or 'all' in pipeline:
        subprocess.run(['python', "06_extract_pcd_from_US_labelmaps_cleanpcd.py",
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', "./imfusion_workspaces/extract_pcd_from_US_labelmaps_cleanpcd.iws",
                        '--root_path_spines', root_path_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--nr_deform_per_spine', nr_deform_per_spine
                        ])

    if 'extract_pcd_network' in pipeline or 'all' in pipeline:
        subprocess.run(['python', "07_extract_pcd_from_network_output_cleanpcd.py",
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', "./imfusion_workspaces/extract_pcd_from_US_labelmaps_cleanpcd.iws",
                        '--root_path_spines', root_path_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--nr_deform_per_spine', nr_deform_per_spine
                        ])
