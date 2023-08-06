import numpy as np
import pyquaternion
from scipy.spatial.transform import Rotation as R
from test_utils import rigid_transform_3D
from sklearn.neighbors import NearestNeighbors


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


def compute_rigid_transform(source_pc, target_pc):

    if source_pc.shape[0] != 3:
        source_pc = np.transpose(source_pc)
        target_pc = np.transpose(target_pc)

    T = np.eye(4)
    R, t = rigid_transform_3D(source_pc, target_pc)
    T[0:3, 0:3], T[0:3, -1] = R, np.squeeze(t)

    return T


def np_chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def vertebrae_pose_error(source, gt_flow, predicted_flow, tre_points=None):

    translation_distance_list = []
    quaternion_distance_list = []
    tre_list = []
    impr_tre_list = []
    for vertebrae_level in range(1, 6):
        vertebra_idxes = np.argwhere(source[:, 3] == vertebrae_level).flatten()

        source_v = source[vertebra_idxes, 0:3]
        gt_deformed_v = source[vertebra_idxes, 0:3] + gt_flow[vertebra_idxes]
        predicted_deformed_v = source[vertebra_idxes, 0:3] + predicted_flow[vertebra_idxes]

        gt_T = compute_rigid_transform(source_v, gt_deformed_v)
        predicted_T = compute_rigid_transform(source_v, predicted_deformed_v)

        translation_distance, quaternion_distance = pose_distance(gt_T, predicted_T)

        # todo: change this as now there is a bug with double
        translation_distance_list.append(translation_distance)
        quaternion_distance_list.append(quaternion_distance)

        # if tre_points is None:
        #     return quaternion_distance_list, translation_distance_list

        # todo: check this
        vertebra_target = tre_points[tre_points[:, -1] == vertebrae_level]
        vertebra_target[:, -1] = 1  # making the point homogeneous

        vertebra_target = np.transpose(vertebra_target)

        gt_registered_target = np.matmul(gt_T, vertebra_target)  # Nx4
        predicted_registered_target = np.matmul(predicted_T, vertebra_target)  # Nx4
        tre_error = np.linalg.norm(gt_registered_target - predicted_registered_target, axis=0)
        tre_list.append(np.mean(tre_error))
        init_tre_error = np.linalg.norm(gt_registered_target - vertebra_target, axis=0)
        impr_tre_list.append(np.mean((init_tre_error - tre_error)/init_tre_error))  # calculate the percentage of improvement

    return quaternion_distance_list, translation_distance_list, tre_list, impr_tre_list

