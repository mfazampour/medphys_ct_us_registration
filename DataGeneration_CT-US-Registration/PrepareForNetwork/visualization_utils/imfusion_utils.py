import xml.etree.ElementTree as ET
import os

imfusion_utils_dir_path = os.path.dirname(os.path.realpath(__file__))


def get_block(parent_block, block_name):
    for child in parent_block:
        if child.attrib["name"] == block_name:
            return child


def get_et_block_from_xml(filename = "load_point_cloud"):
    tree = ET.parse(imfusion_utils_dir_path + "/imfusion_xml_algorithms/" + filename + ".iws")
    root = tree.getroot()
    return root, tree


def create_parent_block(root, block_name):
    parent_block = get_block(root, block_name)

    # if there is no Algorithms in the xml file, create one
    if parent_block is None:
        parent_block = ET.Element("property")
        parent_block.attrib["name"] = block_name
        parent_block.tail = '\n'
        parent_block.text = '\n'
        root.insert(1, parent_block)

    parent_block = get_block(root, block_name)
    return root, parent_block


def set_param_dict_in_et_block(et_block, param_dict):
    for param in et_block:
        if param.attrib["name"] in param_dict.keys():
            param.text = param_dict[param.attrib["name"]]

    return et_block


def add_block_to_xml(root, parent_block_name= "Algorithms", block_name = "load_point_cloud", param_dict={}):
    root, algorithms_block = create_parent_block(root, parent_block_name)
    pc_load_block, _ = get_et_block_from_xml(filename=block_name)

    pc_load_block = set_param_dict_in_et_block(pc_load_block, param_dict)

    pc_load_block.tail = '\n'
    algorithms_block.insert(0, pc_load_block)

    # ET.dump(root)
    return root


def dump_et_root(et_root):
    ET.dump(et_root)


def write_on_file(et_tree, filepath):
    et_tree.write(filepath)


def get_empty_imfusion_ws():
    tree = ET.parse(imfusion_utils_dir_path + "/imfusion_xml_algorithms/empty_imfusion_ws.iws")
    root = tree.getroot()
    return tree, root

#tree = ET.parse("C:\\Repo\\thesis\\tests\\pc_sanity_check.iws")
# tree = ET.parse("imfusion_xml_algorithms/empty_imfusion_ws.iws")
# root = tree.getroot()
# add_block_to_xml(root, "Algorithms", "load_point_cloud",
#                  param_dict = {"location": "location_txt", "outputUids": "outputUids_txt"})
