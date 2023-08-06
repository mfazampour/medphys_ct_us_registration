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

      Steps of the pipeline:
      #1. Generate springs in between vertebra body and facets as well as fixed points. These serve for the spine deformation
      #2. Run deformation script (call sofa from python environment) and generate the deformed spines, save them as vtu (one vtu/vertebra) 
      #3. Convert vtu to obj
      #4. Scale mesh back up 
      #5. Merge vertebrae into one spine mesh
      #6. Center the spines and the vertebrae meshes
      
      # TODO add: 
      #4. Convert vtu to txt, txt to GT registration (from Jane's pipeline) ?
      """

    arg_parser = argparse.ArgumentParser(description="Generate strings in between vertebrae for spine deformation")

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

    args = arg_parser.parse_args()

    root_path_spines = args.root_path_spines
    root_path_vertebrae = args.root_path_vertebrae
    txt_file_lumbar_spines = args.txt_file
    nr_deform_per_spine = args.nr_deform_per_spine

    root_folder_json_files = os.path.join(root_path_vertebrae,"spring_files")
    forces_folder = os.path.join(root_path_vertebrae, "forces_folder")

    pipeline = args.pipeline

    if 'scale_mesh_down' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '01_scale_mesh_down.py',
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_scale_mesh', "imfusion_workspaces/scale_down_mesh.iws"])

    if 'generate_springs' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '02_generate_springs_spine_deformation.py',
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--root_folder_json_files', root_folder_json_files,
                        '--list_file_names', txt_file_lumbar_spines
                        ])

    # deforms all spines in the list without GUI
    # works when called from an environment with python version matching the one from the installation of sofa
    # in my case python 3.9
    # QT and therefore the GUI might not work if QT is not installed in the same environment
    # for deforming all vertebrae, the GUI is not needed
    if 'deform_spines' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '03_deform_lumbar_spines.py',
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_folder_json_files', root_folder_json_files,
                        '--nr_deform_per_spine', nr_deform_per_spine,
                        '--root_folder_forces_files', forces_folder,
                        '--deform_all'
                        ])
    if 'convert_vtu_to_obj' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '04_convert_vtu_to_obj.py',
                        '--list_file_names', txt_file_lumbar_spines,
                        '--root_path_vertebrae', root_path_vertebrae
                        ])
    if 'scale_mesh_up' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '05_scale_mesh_up.py',
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_scale_mesh', "imfusion_workspaces/scale_up_mesh.iws"])
    if 'merge_vertebrae_into_spine' in pipeline or 'all' in pipeline:
        subprocess.run(['python', "06_merge_vertebrae_into_spine_mesh.py",
                        '--list_file_names', txt_file_lumbar_spines,
                        '--workspace_file', "imfusion_workspaces/merge_lumbar_vertebrae.iws",
                        '--root_path_spines', root_path_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--nr_deform_per_spine', nr_deform_per_spine
                        ])
    if 'center_spine_and_vertebrae' in pipeline or 'all' in pipeline:
        subprocess.run(['python', '07_center_mesh.py',
                        '--root_path_spines', root_path_spines,
                        '--root_path_vertebrae', root_path_vertebrae,
                        '--list_file_names', txt_file_lumbar_spines,
                        '--nr_deform_per_spine', nr_deform_per_spine])