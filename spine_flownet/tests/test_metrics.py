import unittest
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

sys.path.insert(0, "../utils")
import metrics

file_dir = os.path.dirname(os.path.realpath(__file__))


def string2array(input_str, delimiter=","):
    return np.array([float(item) for item in input_str.split(delimiter)])


def read_transformation(filepath):
    with open(filepath, "r") as fid:
        mat_string = fid.readlines()[0]

    split_mat_string = mat_string.split("],")

    clean_mat_lines = [item.replace("]", "").replace("[", "") for item in split_mat_string]
    mat = np.array([string2array(item) for item in clean_mat_lines])
    return mat


class TestMetrics(unittest.TestCase):

    @staticmethod
    def get_angle_error(estimated_rot, gt_rot):

        estimated_angles = R.from_matrix(estimated_rot).as_euler('zyx', degrees=True)
        gt_angles = R.from_matrix(gt_rot).as_euler('zyx', degrees=True)

        angle_error = estimated_angles - gt_angles

        angle_error_percentage = [angle_error[i] / gt_angles[i] * 100 if gt_angles[i] != 0 else -1 for i in range(3)]

        return angle_error, angle_error_percentage

    @staticmethod
    def get_translation_error(estimated_translation, gt_translation):

        translation_error = estimated_translation - gt_translation

        translation_error_percentage = [translation_error[i] / gt_translation[i] * 100 if gt_translation[i] != 0
                                        else -1 for i in range(3)]

        return translation_error, translation_error_percentage

    @staticmethod
    def list2string(input_list, decimal=4, suffix=""):

        decimal_string = '%.' + str(decimal) + 'f'
        lis_of_strings = [decimal_string % n + suffix for n in input_list]

        return " ".join(lis_of_strings)

    def test_umeyama_absolute_orientation(self):

        root = "test_data/test_umeyama_absolute_orientation/"

        for target, transformation in zip(["target_pc1", "target_pc2"], ["transformation_1", "transformation_2"]):

            source_pc = np.loadtxt(root + "source_pc.txt")
            target_pc = np.loadtxt(root + target + ".txt")
            gt_transformation = read_transformation(root + transformation + ".txt")

            rot, t = metrics.umeyama_absolute_orientation(from_points=source_pc,
                                                          to_points=target_pc)

            estimated_transformation = metrics.rot_transl2homogeneous(rot, t)

            rot_err, rot_err_perc = self.get_angle_error(estimated_rot=rot,
                                                         gt_rot=gt_transformation[0:3, 0:3])

            transl_err, transl_err_perc = self.get_translation_error(estimated_translation=t,
                                                                     gt_translation=gt_transformation[0:3, 3])

            print("Angle error: {} - in Percentage: {}".format(self.list2string(rot_err, decimal=4),
                                                               self.list2string(rot_err_perc, decimal=2, suffix="%")))
            print("Translation error: {} - in Percentage: {}".format(self.list2string(transl_err, decimal=4),
                                                                     self.list2string(transl_err_perc, decimal=2, suffix="%")))

            # print(gt_transformation)
            # print(estimated_transformation)

            np.testing.assert_almost_equal(gt_transformation, estimated_transformation, decimal=4)


if __name__ == '__main__':
    unittest.main()
