import os
import argparse
import json
import glob
import sys
import random
import numpy as np

# set these in terminal before running:
os.environ['SOFA_ROOT'] = '/home/farid/SOFA/v22.12.00/'
existing_value = os.environ.get('LD_LIBRARY_PATH', '')
new_value = "/home/farid/SOFA/v22.12.00/lib/:/home/farid/SOFA/v22.12.00/collections/SofaSimulation/lib/:" + existing_value
os.environ['LD_LIBRARY_PATH'] = new_value

print(os.environ.get('LD_LIBRARY_PATH'))

import Sofa
import Sofa.Core
import SofaRuntime
import Sofa.Gui


SofaRuntime.importPlugin("Sofa.Component.StateContainer")
SofaRuntime.importPlugin("SofaOpenglVisual")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
SofaRuntime.importPlugin('SofaComponentAll')

constant_force_fields_on_anterior = {
    'vert1': '0 0.01 0.0',
    'vert2': '0 0.02 0.0',
    'vert3': '0 0.07 0.0',
    'vert4': '0 0.02 0',
    'vert5': '0 0.01 0',
}

constant_force_fields_on_posterior= {
    'vert1': '0 -0.01 0.0',
    'vert2': '-0.01 -0.02 0.0',
    'vert3': '-0.01 -0.05 0.0',
    'vert4': '-0.01 -0.02 0',
    'vert5': '0 -0.01 0',
}

def get_path_vertebrae_mesh(root_path_vertebrae, spine_id, vert_id):
    label = str(vert_id + 20)
    folder_name = os.path.join(root_path_vertebrae, str(spine_id) + "_verLev" + str(label))

    # in the folder look for obj file
    filenames = glob.glob(os.path.join(folder_name, '*_scaled_msh.obj'))
    if (len(filenames) != 1):
        raise "There are multiple obj file for this vertebra: " + str(spine_id)

    return filenames[0], folder_name

def get_force_field(anterior = True):

    # sample for x axis from interval [-0.01,0.009] one value. This will be used for vert2x vert3x and vert4x
    # and accounts for the fact that the patient might be slightly tilted when laying in the CT
    # for y axis:
    # for vert1 and vert5 sample one value in interval [-0.015, -0.01]
    # for vert2 and vert4 sample one value in interval [-0.02, -0.015]
    # for vert3 sample one value in interval [-0.04, -0.05]
    if(anterior):
        x_axis_force = random.randint(-10, 9) * 0.001
        y_axis_force_v1_v5 = random.randint(10, 15) * 0.001
        y_axis_force_v2_v4 = random.randint(15, 20) * 0.001
        y_axis_force_v3 = random.randint(30, 50) * 0.001
    else:
        x_axis_force = random.randint(-10, 9) * 0.001
        y_axis_force_v1_v5 = random.randint(-15,-10) * 0.001
        y_axis_force_v2_v4 = random.randint(-20,-15) * 0.001
        y_axis_force_v3 = random.randint(-50,-30) * 0.001

    force_fields = {
        # vert        x                            y                          z
        'vert1':    '0.0'          + ' ' +  str(y_axis_force_v1_v5) + ' ' + '0.0',
        'vert2': str(x_axis_force) + ' ' +  str(y_axis_force_v2_v4) + ' ' + '0.0',
        'vert3': str(x_axis_force) + ' ' +  str(y_axis_force_v3)    + ' ' + '0.0',
        'vert4': str(x_axis_force) + ' ' +  str(y_axis_force_v2_v4) + ' ' + '0.0',
        'vert5':     '0.0'         + ' ' +  str(y_axis_force_v1_v5) + ' ' + '0.0'
    }

    return force_fields

def add_collision_function(root):
    # Collision function
    root.addObject('DefaultPipeline', verbose='0', name='CollisionPipeline')
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('DefaultContactManager', response='PenalityContactForceField', name='collision response')
    # root.addObject('PenaltyContactForceField', response='default', name='collision response')

    root.addObject('LocalMinDistance', name='Proximity', alarmDistance='0.0005', contactDistance='0.000001',
                   angleCone='0.0')
    root.addObject('DiscreteIntersection')


def add_vertebra_node(parent_node_vertebrae, nr_vertebra, spine_id, filename_vertebra, save_vtu_to,
                      constant_force_field, forcesID):
    curr_vert_id = str(nr_vertebra + 1)
    label = str(20 + nr_vertebra)
    curr_vert = parent_node_vertebrae.addChild('vert' + curr_vert_id)

    curr_vert.addObject('MechanicalObject', name='center_mass' + curr_vert_id, template='Rigid3d')
    curr_vert.addObject('RestShapeSpringsForceField', name='fixedPoints' + curr_vert_id, stiffness='5000', angularStiffness='50')

    # mechanical model node as child from the current vertebra node
    mecha_node = curr_vert.addChild('mecha_node' + curr_vert_id)

    mecha_node.addObject('MeshObjLoader',
                         name='ObjLoader' + curr_vert_id,
                         filename=filename_vertebra,
                         printLog='true',
                         flipNormals='0')

    mecha_node.addObject('MeshTopology', name='v' + curr_vert_id + '_topology', src='@ObjLoader' + curr_vert_id)
    mecha_node.addObject('MechanicalObject', name='points' + curr_vert_id, template='Vec3d',
                         src='@v' + curr_vert_id + '_topology')
    mecha_node.addObject('TriangleCollisionModel')

    # visual model node  as a child of the mecha node
    visual_model = mecha_node.addChild('VisualModel')
    visual_model.addObject('OglModel', name='Visual', src='@../ObjLoader' + curr_vert_id, color='white',
                           translation='0 0 0')
    visual_model.addObject('IdentityMapping')

    # on each vertebra a different force is applied
    mecha_node.addObject('ConstantForceField', force=constant_force_field['vert' + curr_vert_id])

    mecha_node.addObject('RigidMapping', input='@..', output='@.')

    mecha_node.addObject('VTKExporter',
                         filename=os.path.join(save_vtu_to, str(spine_id) + '_verLev' + label + "_forces" + str(forcesID) + 'scaled_deformed_20_'),
                         listening='true', edges='0', triangles='1', quads='0', tetras='0',
                         pointsDataFields='points' + curr_vert_id + '.position', exportEveryNumberOfSteps='20')


def add_springs_between_vertebrae(parent_node_vertebrae, nr_vertebra, springs_data):
    idx_first_vertebra = str(nr_vertebra)
    idx_second_vertebra = str(nr_vertebra + 1)
    curr_vertebrae_pair = 'v' + idx_first_vertebra + 'v' + idx_second_vertebra
    # print('currvertpair' + str(curr_vertebrae_pair))

    object1 = '@vert' + idx_first_vertebra + '/mecha_node' + idx_first_vertebra + '/points' + idx_first_vertebra
    object2 = '@vert' + idx_second_vertebra + '/mecha_node' + idx_second_vertebra + '/points' + idx_second_vertebra

    # add springs between vertebrae bodies
    parent_node_vertebrae.addObject('StiffSpringForceField',
                                    template='Vec3d',
                                    name='box_springs' + idx_first_vertebra,
                                    object1=object1,
                                    object2=object2,
                                    spring=springs_data['springs'][curr_vertebrae_pair]['body'])

    # add springs between facet left
    parent_node_vertebrae.addObject('StiffSpringForceField',
                                    template='Vec3d',
                                    name='facet_left_springs' + idx_first_vertebra,
                                    object1=object1,
                                    object2=object2,
                                    spring=springs_data['springs'][curr_vertebrae_pair]['facet_left'])

    # add springs between facet right
    parent_node_vertebrae.addObject('StiffSpringForceField',
                                    template='Vec3d',
                                    name='facet_right_springs' + idx_first_vertebra,
                                    object1=object1,
                                    object2=object2,
                                    spring=springs_data['springs'][curr_vertebrae_pair]['facet_right'])


def add_fixed_points(parent_node_vertebra, nr_vertebra, springs_data):
    fixed_points = parent_node_vertebra.addChild('fixed_points' + str(nr_vertebra))
    fixed_points.addObject('MechanicalObject',
                           name='Particles' + str(nr_vertebra),
                           template='Vec3d',
                           position=springs_data['fixed_points_positions']['v' + str(nr_vertebra)])
    fixed_points.addObject('MeshTopology',
                           name='Topology',
                           hexas=springs_data['fixed_points_indices']['v' + str(nr_vertebra)])

    fixed_points.addObject('UniformMass', name='Mass', vertexMass='1')
    fixed_points.addObject('FixedConstraint',
                           template='Vec3d',
                           name='fixedConstraint' + str(nr_vertebra),
                           indices=springs_data['fixed_points_indices']['v' + str(nr_vertebra)])


def createScene(root, spine_id, path_json_file, root_path_vertebrae, constant_force_field, forcesID):
    nr_vertebrae = 5
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')

    # Visualization style
    root.addObject('VisualStyle', displayFlags='showVisual showBehaviorModels showInteractionForceFields')

    add_collision_function(root)

    # Parent node of all vertebrae
    together = root.addChild('Together')
    together.addObject('EulerImplicitSolver', name='cg_odesolver', printLog='0', rayleighStiffness='0.09',rayleighMass='0.1')
    #together.addObject('EulerImplicitSolver', name='cg_odesolver', printLog='0', rayleighStiffness='9',rayleighMass='10')
    together.addObject('CGLinearSolver', name='linear solver', iterations='1000', tolerance='1e-09', threshold='1e-15')

    # mech objects of vertebrae
    for i in range(nr_vertebrae):
        filename, dirname = get_path_vertebrae_mesh(root_path_vertebrae, spine_id, i)
        add_vertebra_node(parent_node_vertebrae=together, nr_vertebra=i, spine_id=spine_id,
                          filename_vertebra=filename, save_vtu_to=dirname, constant_force_field=constant_force_field, forcesID=forcesID)

    # read springs file
    file = open(path_json_file)
    json_data = json.load(file)

    # add springs in between vertebrae
    for i in range(1, nr_vertebrae):
        add_springs_between_vertebrae(parent_node_vertebrae=together, nr_vertebra=i, springs_data=json_data)

    # add positions of fixed points for v1
    add_fixed_points(parent_node_vertebra=together, nr_vertebra=1, springs_data=json_data)

    # add positions of fixed points for v5
    add_fixed_points(parent_node_vertebra=together, nr_vertebra=5, springs_data=json_data)

    # add springs between points in v1 and fixed points in v1 and the same for v5
    together.addObject('StiffSpringForceField',
                       name='points_fixed1',
                       object1='@fixed_points1/Particles1',
                       object2='@vert1/mecha_node1/points1',
                       spring=json_data['springs']['v0v1'])

    together.addObject('StiffSpringForceField',
                       name='points_fixed5',
                       object1='@fixed_points5/Particles5',
                       object2='@vert5/mecha_node5/points5',
                       spring=json_data['springs']['v5v6'])

    return root


def deform_one_spine(spine_id, path_json_file, root_path_vertebrae, constant_force_field, forcesID, use_gui=True):
    print("Deforming " + str(spine_id))
    root = Sofa.Core.Node('root')
    createScene(root, spine_id, path_json_file, root_path_vertebrae, constant_force_field, forcesID)
    Sofa.Simulation.init(root)

    if not use_gui:
        for iteration in range(20):
            print("Iteration:" + str(iteration))
            Sofa.Simulation.animate(root, root.dt.value)
            # print(str(root.dt.value))
    else:
        # Find out the supported GUIs
        print("Supported GUIs are: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
        # iterate over all spines in the txt file
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)

        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")
    # Sofa.Simulation.Deinit(root)

    print("Simulation is done.")


def deform_all_spines(txt_file, json_root_folder, vertebrae_root_folder, nr_deformations_per_spine, forces_folder, anterior):
    if (txt_file == None or json_root_folder == None or vertebrae_root_folder == None):
        raise Exception("Please provide all parameters: txt_file, json_root_folder and vertebrae_root_folder when calling the script for deformation of all spines ")
        return
    # algo for processing multiple spines
    # open txt file with list of files
    with open(txt_file) as file:
        spine_ids = [line.strip() for line in file]

    # we can only deform all spines by setting use_gui to False
    # iterate over all, deform and save vtu files
    for spine_id in spine_ids:
        # spine_id = spine_ids[0]
        json_path = os.path.join(json_root_folder, spine_id + ".json")

        for nr_deform in range(nr_deformations_per_spine):
            force_field = get_force_field(anterior=np.random.uniform(size=1) > 0.5)  # randomly switch between anterior and posterior

            # save the current force field:
            force_fields_for_one_deformation = open(os.path.join(forces_folder, spine_id +  "_" + str(nr_deform) + ".txt"), "w")
            force_fields_for_one_deformation.write(str(force_field))
            #force_fields_for_one_deformation.write("testtest")
            #force_fields_for_one_deformation.write("\n")

            # perform simulation and save the results
            try:
                deform_one_spine(spine_id, json_path, vertebrae_root_folder, force_field, nr_deform, use_gui=False)
            except Exception:
                print("There was something wrong with deforming: " + str(spine_id), file=sys.stderr)


if __name__ == '__main__':
    # example setup
    """
    spine_id = 'sub-verse500'
    path_json_file = '/home/miruna20/Documents/Thesis/Code/Preprocessing/master_thesis/samples/subverse500.json'
    """

    arg_parser = argparse.ArgumentParser(description="Create scene for lumbar spine deformation")

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=False,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=False,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--root_folder_json_files",
        required=False,
        dest="root_json_files",
        help="Root folder where the json files will be saved."
    )

    arg_parser.add_argument(
        "--deform_all",
        action="store_true",
        dest="deform_all",
        help="Activate flag to deform all spines without GUI and save the vtu results"
    )

    arg_parser.add_argument(
        "--root_folder_forces_files",
        required=False,
        dest="forces_folder",
        help="Root folder where the txt files with force fields will be saved."
    )

    arg_parser.add_argument(
        "--nr_deform_per_spine",
        required=True,
        dest="nr_deform_per_spine",
        help="Number of deformed spines per initial spine."
    )

    args = arg_parser.parse_args()
    nr_deformations_per_spine = int(args.nr_deform_per_spine)

    # create folder for forces files if it doesn't exist
    if(not os.path.exists(args.forces_folder)):
        os.mkdir(args.forces_folder)

    print("Deforming spines with sofa framework")
    anterior = True
    # automatically deform all spines and save vtu files, in this case use_gui is automatically set to False
    if (args.deform_all):
        deform_all_spines(args.txt_file, args.root_json_files, args.root_path_vertebrae,nr_deformations_per_spine,args.forces_folder, anterior=anterior)
    else:
        # alternatively you can choose to deform only one spine with or without GUI e.g to verify exactly how the deformation works
        spine_id = 'sub-verse807'
        deform_one_spine(
            spine_id=spine_id,
            path_json_file=os.path.join(
                args.root_json_files, spine_id + ".json"),
            root_path_vertebrae=args.root_path_vertebrae,
            constant_force_field=constant_force_fields_on_anterior,
            forcesID=0,
            use_gui=True,
        )