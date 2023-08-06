import numpy as np
import pyquaternion
from scipy.spatial.transform import Rotation as R


def rot_transl2homogeneous(rot, t):
    homogeneous_transformation = np.eye(4)
    homogeneous_transformation[0:3, 0:3] = rot
    homogeneous_transformation[0:3, 3] = t

    return homogeneous_transformation


def umeyama_absolute_orientation(from_points, to_points, fix_scaling=True):
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)

    if fix_scaling:
        t = mean_to - R.dot(mean_from)
        return R, t

    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)

    return c * R, t


def pose_distance(p1, p2):

    translation_distance = np.linalg.norm(p1[0:3, -1] - p2[0:3, -1])
    q0 = R.from_matrix(p1[0:3, 0:3]).as_quat()
    q1 = R.from_matrix(p2[0:3, 0:3]).as_quat()

    quaternion_distance = pyquaternion.Quaternion.absolute_distance(pyquaternion.Quaternion(q0), pyquaternion.Quaternion(q1))
    return translation_distance, quaternion_distance


def vertebrae_pose_error(source, gt_flow, predicted_flow):

    translation_distance_list = []
    quaternion_distance_list = []
    for vertebrae_level in range(1, 6):
        vertebra_idxes = np.argwhere(source[:, 3] == vertebrae_level).flatten()

        source_v = source[vertebra_idxes, 0:3]
        gt_deformed_v = source[vertebra_idxes, 0:3] + gt_flow[vertebra_idxes]
        predicted_deformed_v = source[vertebra_idxes, 0:3] + predicted_flow[vertebra_idxes]

        gt_T = np.eye(4)
        predicted_T = np.eye(4)

        gt_T[0:3, 0:3], gt_T[0:3, -1] = umeyama_absolute_orientation(source_v, gt_deformed_v)
        predicted_T[0:3, 0:3], predicted_T[0:3, -1] = umeyama_absolute_orientation(source_v, predicted_deformed_v)

        translation_distance, quaternion_distance = pose_distance(gt_T, predicted_T)
        translation_distance_list.append(translation_distance)
        quaternion_distance_list.append(quaternion_distance)

    return quaternion_distance_list, translation_distance_list

