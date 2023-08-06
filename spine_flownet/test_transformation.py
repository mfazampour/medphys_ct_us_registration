import numpy as np


def make_homogeneous(pc):
    if pc.shape[0] != 3:
        pc = np.transpose(pc)

    assert pc.shape[0] == 3

    return np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)

root = "C:/Repo/thesis/raycastedCPDRes/temps/"
predicted_T_it1 = np.loadtxt(root + "predicted_T_it1.txt")
source_pc = np.loadtxt(root + "source_pc.txt")
source_pc_it1 = np.loadtxt(root + "source_pc_it1.txt")

Y = np.copy(source_pc)
source_pc = make_homogeneous(source_pc)

a = np.dot(predicted_T_it1, source_pc)
b = np.matmul(predicted_T_it1, source_pc)


R = predicted_T_it1[0:3, 0:3]
t = predicted_T_it1[0:3, -1]


f = np.matmul(Y, R) + t
f2 = np.transpose(np.matmul(np.transpose(R), np.transpose(Y)))

predicted_T_it1[0:3, 0:3] = np.transpose(predicted_T_it1[0:3, 0:3])
res = np.matmul(predicted_T_it1, source_pc)
res_t = np.transpose(res[0:3, ...])

print(np.sum(f-res_t))

#f2 = np.transpose(np.dot(R, source_pc[0:3, ...]))
# print(np.sum(f-f2))
#
# print(np.sum(a-b))
# print(np.sum(a[0:3, ...]-np.transpose(f)))