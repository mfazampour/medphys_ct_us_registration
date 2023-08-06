import numpy as np
from pycpd import RigidRegistration


class BiomechanicalCpd(RigidRegistration):
    def __init__(self, target_pc, source_pc, springs=(), max_iterations=10):
        """
        :param: target_pc (np.ndarray): An np.ndarray with size (N, 3) containing the target point cloud
        :param: source_pc (np.ndarray): An np.ndarray with size (M, 3) containing the target point cloud
        :param: springs (list( (int, np.ndarray) )): is a list containing tuples, where each tuple contains:
            1. The index of the point connected to a spring in the source points cloud
            2. The position of the point the spring is connected to as a np.ndarray os size (3, )
        :param: max_iterations (int): The maximum number of iteration of the CPD algorithm
        """

        if len(springs) > 0:
            connected_points = [item[1] for item in springs]
            target_pc = np.concatenate([target_pc, connected_points], axis=0)

        super(BiomechanicalCpd, self).__init__(X=target_pc, Y=source_pc, mamax_iterations=max_iterations)
        self.alpha = 2 ** 5
        self.springs_idxs = springs  # contains the springs indexes on the y dataset
        self.max_iterations = max_iterations
        self.biomechanical_constraint = len(springs) > 0

        self.constrained = len(springs) > 0

        if self.biomechanical_constraint:
            P_from_springs = np.zeros(shape=(len(self.springs_idxs), self.P.shape[1]))
            for i, spring in enumerate(springs):
                P_from_springs[i, spring[0]] = 2 * self.sigma2 * self.alpha  # 2*sigma*alpha

            self.P = np.concatenate([self.P, P_from_springs], axis=0)

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.
        """

        # target point cloud mean
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        self.YPY = np.dot(np.transpose(self.P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))

        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        # self.s = np.trace(np.dot(np.transpose(self.A), np.transpose(self.R))) / self.YPY
        self.t = np.transpose(muX) - self.s * np.dot(np.transpose(self.R), np.transpose(muY))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q

        trAR = np.trace(np.dot(self.A, self.R))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
                 (2 * self.sigma2) + self.D * self.Np / 2 * np.log(self.sigma2)

        self.diff = np.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
