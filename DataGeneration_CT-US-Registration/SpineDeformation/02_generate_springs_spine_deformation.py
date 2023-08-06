import numpy as np
import open3d as o3d
import math
import os
import pathlib
import argparse
import json
import trimesh
import sys

s_body = 500
d_body = 3

s_facet = 8000
d_facet = 500

s_for_fixed_points = 1000
d_for_fixed_points = 10
#length_strings_between_fixed_points = 0.001
length_strings_between_fixed_points = 0.000001

# offset_center_of_mass = 5
# radius_search_for_facet = 1
radius_search_for_facet = 0.001
offset_center_of_mass = 0.005

def get_vertebrae_meshes_from_filenames(root_folder, spine_name):
    vertebrae_meshes = []
    for i in range(0, 5):
        path = os.path.join(root_folder, spine_name + "_verLev" + str(20 + i))
        pathVertebra = \
            list(
                pathlib.Path(path).glob('*scaled_msh.obj'))
        if (len(pathVertebra) != 1):
            print(pathVertebra, path)
            raise Exception("There are multiple obj file for this vertebra: " + str(spine_id) + " " + str(i))
        mesh = o3d.io.read_triangle_mesh(str(pathVertebra[0]))
        # print("Path vertebra: " + str(pathVertebra[0]))
        vertebrae_meshes.append(mesh)
    return vertebrae_meshes

def process_vertebrae_bodies(vert1, vert2, s, d):

    # get bounding boxes
    _, vert1_bb2 = get_bounding_boxes_of_vertebra_body(vert1)
    vert2_bb1, _ = get_bounding_boxes_of_vertebra_body(vert2)

    # get indices of the meshes that are in these bounding boxes
    indices_vert1, indices_vert2 = get_indices_of_vertebrae_bodies(vert1, vert2, vert1_bb2, vert2_bb1)

    # call print_specs_between_vertebrae
    return print_spring_specs_between_vertebrae(vert1, vert2, indices_vert1, indices_vert2, s, d, body=True)

def process_facets(vert1, vert2, s, d):
    indices_facet_v1_left, indices_facet_v2_left, indices_facet_v1_right, indices_facet_v2_right = get_indices_of_facets_of_2_vertebrae(
        vert1, vert2)

    # due to some missegmentations (see L4-L5 in sub-verse519) the 2 vertebrae meshes are also very close at the base, not only between facets which generates wrong springs
    # to fix this issue, we only consider points with y coordinate larger than the y coord of the center of mass point
    indices_facet_v1_left = get_indices_points_with_y_coordinate_higher_than_center_of_mass(vert1,indices_facet_v1_left)
    indices_facet_v1_right = get_indices_points_with_y_coordinate_higher_than_center_of_mass(vert1,indices_facet_v1_right)
    indices_facet_v2_left = get_indices_points_with_y_coordinate_higher_than_center_of_mass(vert2,indices_facet_v2_left)
    indices_facet_v2_right = get_indices_points_with_y_coordinate_higher_than_center_of_mass(vert2,indices_facet_v2_right)

    specs_facet_left = print_spring_specs_between_vertebrae(vert1, vert2, indices_facet_v1_left, indices_facet_v2_left,
                                                            s, d, body=False)
    specs_facet_right = print_spring_specs_between_vertebrae(vert1, vert2, indices_facet_v1_right,
                                                             indices_facet_v2_right, s, d, body=False)

    return specs_facet_left, specs_facet_right

def process_fixed_points(vert1, position_bounding_box_for_fixed_points):

    leftbb, rightbb = get_bounding_boxes_of_vertebra_body(vert1)

    points_within_bb = o3d.geometry.OrientedBoundingBox.get_point_indices_within_bounding_box(
        leftbb if position_bounding_box_for_fixed_points=="prev" else rightbb, vert1.vertices)

    indices_body_t12_v1 = get_indices_points_with_selected_normals_within_bb(vert1, [0,0,1] if position_bounding_box_for_fixed_points=="prev" else [0,0,-1], points_within_bb)

    # find springs for these fixed points
    indices_fixed_points, positions_fixed_points, springs_fixed_points = print_springs_specs_fixed_points(vert1,
                                                                                                          indices_body_t12_v1,
                                                                                                          s_for_fixed_points,
                                                                                                          d_for_fixed_points,
                                                                                                          length_strings_between_fixed_points)
    return indices_fixed_points, positions_fixed_points, springs_fixed_points

def get_bounding_boxes_of_vertebra_body(fullVertebra):
    # find the center of mass of the vertebra
    center_of_mass = fullVertebra.get_center()

    # crop the rest and have only the vertebra body
    minBounds = fullVertebra.get_min_bound()
    maxBounds = fullVertebra.get_max_bound()

    minx = minBounds[0]
    maxx = maxBounds[0]
    miny = minBounds[1]
    maxy = maxBounds[1]
    minz = minBounds[2]
    maxz = maxBounds[2]

    crop_point_along_y = center_of_mass[1] + offset_center_of_mass
    middle_point_along_z = math.floor((minz + maxz) / 2)

    numpyArray = np.array([
        [minx, miny, minz], [minx, miny, maxz], [maxx, miny, minz], [maxx, miny, maxz],  # lower face of the cube
        [minx, crop_point_along_y, minz], [minx, crop_point_along_y, maxz], [maxx, crop_point_along_y, maxz],
        [maxx, crop_point_along_y, minz]  # upper face of the cube
    ])

    vertebrabody_points = o3d.utility.Vector3dVector(numpyArray)
    bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points=vertebrabody_points)
    vertebrabody_mesh = fullVertebra.crop(bounding_box)

    # bb1 = bounding box belonging to Lx and between Lx and L(x-1)
    # bb2 = will be the bounding box belonging to Lx and between Lx and L(x+1)
    bounding_box_face1_points = [[minx, miny, maxz],
                                 [maxx, miny, maxz],
                                 [maxx, crop_point_along_y, center_of_mass[2]],
                                 [minx, crop_point_along_y, center_of_mass[2]],
                                 [minx, miny, center_of_mass[2]],
                                 [minx, crop_point_along_y, maxz],
                                 [maxx, miny, center_of_mass[2]],
                                 [maxx, crop_point_along_y, maxz]]
    bb1 = o3d.geometry.OrientedBoundingBox.create_from_points(
        points=o3d.utility.Vector3dVector(np.asarray(bounding_box_face1_points)))

    bounding_box_face2_points = [[minx, miny, center_of_mass[2]],
                                 [maxx, miny, center_of_mass[2]],
                                 [maxx, crop_point_along_y, minz],
                                 [minx, crop_point_along_y, minz],
                                 [maxx, miny, minz],
                                 [maxx, crop_point_along_y, center_of_mass[2]],
                                 [minx, crop_point_along_y, center_of_mass[2]],
                                 [minx, miny, minz]
                                 ]

    bb2 = o3d.geometry.OrientedBoundingBox.create_from_points(
        points=o3d.utility.Vector3dVector(np.asarray(bounding_box_face2_points)))

    if visualize:
        print("visualizing bb1 which is pointing towards L(x-1). This should be on the side of the facets")
        o3d.visualization.draw([bb1, fullVertebra])
        print("visualizing bb2 which is pointing towards L(x+1). This should be opposite to the facets")
        o3d.visualization.draw([bb2, fullVertebra])

    return bb1, bb2

def get_indices_of_facets_of_2_vertebrae(vert1_mesh, vert2_mesh):
    # get centers of the 2 vertebrae
    center_vert1 = vert1_mesh.get_center()

    # create point cloud from the vertices of the mesh
    vert1_vert2_pc = o3d.geometry.PointCloud()

    # combine points from both vertebrae into one pc
    points_vert1_vert2_stacked = np.concatenate((np.asarray(vert1_mesh.vertices), np.asarray(vert2_mesh.vertices)))
    vert1_vert2_pc.points = o3d.utility.Vector3dVector(points_vert1_vert2_stacked)
    vert1_vert2_pc.paint_uniform_color([0.5, 0.5, 0.5])

    # so that we know that starting with index=nr_points_vert1 the points belong to the second vertebra
    nr_points_vert1 = np.asarray(vert1_mesh.vertices).shape[0]

    # create kd tree
    pcd_tree = o3d.geometry.KDTreeFlann(vert1_vert2_pc)

    idx_of_facet_points_in_vert1_left = []
    idx_of_facet_points_in_vert1_right = []
    idx_of_facet_points_in_vert2_left = []
    idx_of_facet_points_in_vert2_right = []

    #if visualize:
    #o3d.visualization.draw([vert1_vert2_pc])

    #print("In total to search for: " + str(nr_points_vert1))
    for idx_vert1 in range(0, nr_points_vert1):
        #print(str(idx_vert1) + "/" + str(nr_points_vert1))
        # find all points within a radius
        [k, idx_neighbors, dist] = pcd_tree.search_radius_vector_3d(vert1_vert2_pc.points[idx_vert1], 0.001)

        # find first index and therefore closest that is larger or equal to nr_points_vert
        index_closest_point_from_pc2 = next(filter(lambda index: index >= nr_points_vert1, idx_neighbors), None)

        #vert1_vert2_pc.colors[idx_vert1] = [1, 0, 0]
        #np.asarray(vert1_vert2_pc.colors)[idx_neighbors] = [0, 1, 0]

        # if None then no point from the other point-cloud is close, so we exclude them
        if (index_closest_point_from_pc2 != None):

            if (np.asarray(vert1_mesh.vertices)[idx_vert1][0] < center_vert1[0]):
                idx_of_facet_points_in_vert1_left.append(idx_vert1)
                idx_of_facet_points_in_vert2_left.append(index_closest_point_from_pc2 - nr_points_vert1)

            else:
                idx_of_facet_points_in_vert1_right.append(idx_vert1)
                idx_of_facet_points_in_vert2_right.append(index_closest_point_from_pc2 - nr_points_vert1)

            # paint them green
            if(visualize):
                vert1_vert2_pc.colors[idx_vert1] = [1, 0, 0]
                np.asarray(vert1_vert2_pc.colors)[index_closest_point_from_pc2] = [0, 1, 0]
                o3d.visualization.draw_geometries([vert1_vert2_pc])

    return idx_of_facet_points_in_vert1_left, idx_of_facet_points_in_vert2_left, idx_of_facet_points_in_vert1_right, idx_of_facet_points_in_vert2_right

def get_indices_points_with_selected_normals_within_bb(vert, normal_vector, bb_indices):
    # get all indices of points that have a certain normal vector e.g [0,0,1]
    normals = np.asarray(vert.vertex_normals)
    indices = [i for i in range(normals.shape[0]) if (np.array_equal(normals[i], normal_vector) and i in bb_indices)]

    # if the current set of indices does not have any normals that align perfectly with z axis, allow for a small tolerance
    if(len(indices)  == 0):
        indices = []
        for i in range(normals.shape[0]):
            if(i in bb_indices):
                if(abs(normals[i][0] - normal_vector[0]) < 0.1 and abs(normals[i][1] - normal_vector[1]) < 0.1 and abs(normals[i][2] - normal_vector[2]) < 0.1):
                    indices.append(i)
    return indices

def get_indices_points_with_y_coordinate_lower_than_center_of_mass(vert,bb_indices):
    vertices = np.asarray(vert.vertices)
    y_coord_center_of_mass = vert.get_center()[1]
    indices = [i for i in range(vertices.shape[0]) if vertices[i][1] < y_coord_center_of_mass and i in bb_indices]
    return indices

def get_indices_points_with_y_coordinate_higher_than_center_of_mass(vert,bb_indices):
    vertices = np.asarray(vert.vertices)
    y_coord_center_of_mass = vert.get_center()[1]+offset_center_of_mass
    indices = [i for i in range(vertices.shape[0]) if vertices[i][1] >= y_coord_center_of_mass and i in bb_indices]
    return indices

def get_indices_points_with_z_coord_bigger_than_center_of_mass(vert, bb_indices):
    vertices = np.asarray(vert.vertices)
    z_coord_center_of_mass = vert.get_center()[2]
    indices = [i for i in range(vertices.shape[0]) if vertices[i][1] > z_coord_center_of_mass-2*offset_center_of_mass and i in bb_indices]
    return indices

def get_indices_points_with_z_coord_smaller_than_center_of_mass(vert, bb_indices):
    vertices = np.asarray(vert.vertices)
    z_coord_center_of_mass = vert.get_center()[2]
    indices = [i for i in range(vertices.shape[0]) if vertices[i][1] <= z_coord_center_of_mass+2*offset_center_of_mass and i in bb_indices]
    return indices

def visualize_lines_between_indices(ind1,ind2,mesh1,mesh2):
    # TODO eliminate duplicity between this function and the similar code used in print springs
    pairs_of_indices = [list(a) for a in zip(ind1, ind2)]

    points = []
    lines = []
    k = 0
    for i, j in pairs_of_indices:
        points.append(mesh1.vertices[i])
        points.append(mesh2.vertices[j])
        lines.append([k, k + 1])
        k += 2
    print("Visualizing all possible lines between vert")
    # create lines in between vertebrae bodies
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set, mesh1, mesh2])

def print_spring_specs_between_vertebrae(vert1, vert2, indices_vert1, indices_vert2, s, d, body=True):
    np.random.shuffle(indices_vert1)
    np.random.shuffle(indices_vert2)

    points = []
    lines = []
    k = 0

    nr_springs = 0
    if (body):
        nr_springs = 800
    else:
        nr_springs = 250

    pairs_of_indices = [list(a) for a in zip(indices_vert1, indices_vert2)]

    idx = np.round(np.linspace(0, len(pairs_of_indices) - 1, nr_springs, dtype='int'))

    filtered_pairs_of_indices = np.array(pairs_of_indices)[idx.astype(int)]
    accum = ""
    for i, j in filtered_pairs_of_indices:
        # f.write()
        accum += "{0} {1} {2} {3} {4} ".format(i, j, s, d, np.linalg.norm(vert1.vertices[i] - vert2.vertices[j]))
        points.append(vert1.vertices[i])
        points.append(vert2.vertices[j])
        lines.append([k, k + 1])
        k += 2

    if (visualize):
        print("Visualizing springs that will be used for deformation between vert " + ("bodies" if body else "facets"))
        # create lines in between vertebrae bodies
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([line_set, vert1, vert2])
    return accum

def print_springs_specs_fixed_points(vert1, indices_vert1, s, d, dist):
    """
    Get the positions of the fixed points belonging to v1 and v5 and the corresponding springs
    :param vert1: mesh vertebra
    :param indices_vert1: indices from the mesh that belong to the bounding box
    :param s: stiffness
    :param d: damping coeff
    :param dist: length of the spring
    :param visualization: whether or not the springs should be visualized
    :return:
    """
    lines = []
    points = []
    k = 0

    # set a certain number of fixed points to verify if this is the cause of flying spines
    #nr_fixed_points = 2000
    #idx = np.round(np.linspace(0, len(indices_vert1) - 1, nr_fixed_points, dtype='int'))
    #filtered_indices_vert1 = np.array(indices_vert1)[idx.astype(int)]

    filtered_indices_vert1 = indices_vert1
    positions = ""
    # the points whose indices are within the bounding box are the fixed points
    for i in filtered_indices_vert1:
        positions += "{0} {1} {2}  ".format(vert1.vertices[i][0], vert1.vertices[i][1], vert1.vertices[i][2])
        points.append(vert1.vertices[i])
        points.append(vert1.vertices[i])
        lines.append([k, k + 1])
        k += 2

    # we need the indices of these points for the mesh hexagonal shape
    indices = ""
    for i, _ in enumerate(filtered_indices_vert1):
        indices += "{0} ".format(i)

    springs = ""
    for i, j in enumerate(filtered_indices_vert1):
        springs += "{0} {1} {2} {3} {4}  ".format(i, j, s, d, dist)

    if (visualize):
        print("Visualizing fixed_points")
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([line_set, vert1])
    return indices, positions, springs

def get_indices_of_vertebrae_bodies(vert1, vert2, bb_v1_v2, bb_v2_v1):
    # find all points within bounding box
    indices_in_bb_v1_v2 = o3d.geometry.OrientedBoundingBox.get_point_indices_within_bounding_box(bb_v1_v2,
                                                                                                 vert1.vertices)
    indices_in_bb_v2_v1 = o3d.geometry.OrientedBoundingBox.get_point_indices_within_bounding_box(bb_v2_v1,
                                                                                                vert2.vertices)

    if(visualize):
        print("Lines between points with indices within the bounding boxes")
        visualize_lines_between_indices(indices_in_bb_v1_v2,indices_in_bb_v2_v1,vert1,vert2)

    # filter the ones with higher y than the center of mass
    indices_in_bb_v1_v2_vert_body = get_indices_points_with_y_coordinate_lower_than_center_of_mass(vert1,indices_in_bb_v1_v2)
    indices_in_bb_v2_v1_vert_body = get_indices_points_with_y_coordinate_lower_than_center_of_mass(vert2,indices_in_bb_v2_v1)

    if(visualize):
        print("Lines between points with indices within the bounding boxes and y coord lower than center of mass")
        visualize_lines_between_indices(indices_in_bb_v1_v2_vert_body, indices_in_bb_v2_v1_vert_body, vert1, vert2)

    # select only points that have the normals parallel to the z axis from positive to negative
    indices_in_bb_on_surface_v1_v2 = get_indices_points_with_selected_normals_within_bb(vert1, [0, 0, -1], indices_in_bb_v1_v2_vert_body)
    # select only points that have the normals parallel to the z axis from negative to positive
    indices_in_bb_on_surface_v2_v1 = get_indices_points_with_selected_normals_within_bb(vert2, [0, 0, 1], indices_in_bb_v2_v1_vert_body)

    if(visualize):
        print("Lines between points with indices within the bounding boxes and normals aligned with z axis")
        visualize_lines_between_indices(indices_in_bb_on_surface_v1_v2,indices_in_bb_on_surface_v2_v1,vert1,vert2)

    return indices_in_bb_on_surface_v1_v2, indices_in_bb_on_surface_v2_v1

def generate_springs_for_one_spine(root_path_spine, spine_id, json_file):
    # list of vertebrae meshes
    vertebrae_meshes = get_vertebrae_meshes_from_filenames(root_folder=root_path_spine,
                                                           spine_name=spine_id)
    # iterate over all pairs

    json_data = {}
    dict_pairs = {}

    for idx_vert in range(1, len(vertebrae_meshes)):
        curr_pair_of_vertebra = "v" + str(idx_vert) + "v" + str(idx_vert + 1)  # e.g v1v2 or v2v3

        print("Working on springs between vertebra: " + "L" + str(idx_vert) + " and " + str("L" + str(idx_vert + 1)))

        v1 = vertebrae_meshes[idx_vert - 1]
        v2 = vertebrae_meshes[idx_vert]

        dict_curr_pair_strings = {}

        #print("Generating springs between vertebrae bodies")
        dict_curr_pair_strings["body"] = process_vertebrae_bodies(v1, v2, s_body, d_body)

        #print("Generating springs between facet joints on the left")
        dict_curr_pair_strings["facet_left"], dict_curr_pair_strings["facet_right"] = process_facets(
            v1, v2, s_facet, d_facet)

        dict_pairs[curr_pair_of_vertebra] = dict_curr_pair_strings

    # get fixed points
    L1_indices_fixed_points, L1_positions_fixed_points, L1_springs_fixed_positions = process_fixed_points(
        vertebrae_meshes[0], "prev")
    L5_indices_fixed_points, L5_positions_fixed_points, L5_springs_fixed_positions = process_fixed_points(
        vertebrae_meshes[4], "after")

    dict_pairs["v0v1"] = L1_springs_fixed_positions
    dict_pairs["v5v6"] = L5_springs_fixed_positions

    json_data["springs"] = dict_pairs

    json_data["fixed_points_positions"] = {
        "v1": L1_positions_fixed_points,
        "v5": L5_positions_fixed_points
    }
    json_data["fixed_points_indices"] = {
        "v1": L1_indices_fixed_points,
        "v5": L5_indices_fixed_points
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":

    """
    # example setup
    root_vertebrae = "/home/miruna20/Documents/Thesis/sofa/vertebrae/train"
    spine_name = "sub-verse500"
    txt_file = "../samples/test.txt"
    """

    arg_parser = argparse.ArgumentParser(description="Generate strings in between vertebrae for spine deformation")

    arg_parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        dest="root_path_vertebrae",
        help="Root path to the vertebrae folders."
    )

    arg_parser.add_argument(
        "--root_folder_json_files",
        required=False,
        dest="root_json_files",
        help="Root folder where the json files will be saved."
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--visualize",
        action="store_true",
        dest="visualize",
        help="Activate flag for visualization"
    )
    print("Generating springs for spine deformation")
    # iterate over these spine ids and get the corresponding L1-L5 vertebrae
    args = arg_parser.parse_args()
    visualize = args.visualize
    # iterate over the txt file and process all spines
    with open(args.txt_file) as file:
        spine_ids = [line.strip() for line in file]

    # create folder for json files if it doesn't exist
    if(not os.path.exists(args.root_json_files)):
        os.mkdir(args.root_json_files)

    for spine_id in spine_ids:
        print("Processing " + str(spine_id))
        try:
            generate_springs_for_one_spine(root_path_spine=args.root_path_vertebrae,spine_id=spine_id,json_file=os.path.join(args.root_json_files,str(spine_id) + ".json"))
        except Exception as e:
            print("Error occured for:  " + str(spine_id) + str(e), file=sys.stderr)

    """
    spine_id = "sub-verse833"
    test_new_alignmend(path_L20_vertebra="/home/miruna20/Documents/Thesis/SpineDeformation/vertebrae/train/sub-verse833_verLev20/sub-verse833_dir-ax_seg-vert_msk_verLev20_msh.obj")
    """