import chamferdist
import numpy as np
import torch
import torch.nn.functional as F


def calculate_loss(batch_size, constraint, flow, flow_pred, loss_opt, pc1, pc2, position1, loss_coeff):
    if 'all' in loss_opt:
        for loss in ["biomechanical", "rigidity", "chamfer", 'mse']:
            if loss not in loss_coeff.keys():
                loss_coeff[loss] = 1.0
    loss = torch.tensor([0.0], device=flow.device, dtype=flow.dtype)
    bio_loss, rig_loss, cham_loss, mse_loss = torch.zeros_like(loss), torch.zeros_like(loss), torch.zeros_like(loss), torch.zeros_like(loss)
    if "mse" in loss_opt or 'all' in loss_opt:
        mse_loss = F.mse_loss(flow_pred.float(), flow.float())
        loss += mse_loss
    if "biomechanical" in loss_opt or 'all' in loss_opt:
        for idx in range(batch_size):
            bio_loss += biomechanical_loss(constraint, flow, flow_pred, idx, pc1, coeff=loss_coeff["biomechanical"])
        bio_loss /= batch_size
        loss += bio_loss
        bio_loss /= loss_coeff["biomechanical"]
    if "rigidity" in loss_opt or 'all' in loss_opt:
        rig_loss = rigidity_loss(flow, flow_pred, pc1, position1, coeff=loss_coeff["rigidity"])
        loss += rig_loss
        rig_loss /= loss_coeff["rigidity"]
    if "chamfer" in loss_opt or 'all' in loss_opt:
        cham_loss = chamfer_loss(flow, flow_pred, pc1, pc2, coeff=loss_coeff["chamfer"])
        loss += cham_loss
        cham_loss /= loss_coeff["chamfer"]
    return bio_loss, cham_loss, loss, mse_loss, rig_loss


chamfer = chamferdist.ChamferDistance()


def chamfer_loss(flow, flow_pred, pc1, pc2, coeff=1):
    predicted = pc1 + flow_pred

    loss = chamfer(predicted.type(torch.float), pc2.type(torch.float), bidirectional=True) * 1e-7
    return loss * coeff


def rigidity_loss(flow, flow_pred, pc1, position1, coeff=1):
    source_point1 = torch.Tensor().cuda()
    source_point2 = torch.Tensor().cuda()
    predict_point1 = torch.Tensor().cuda()
    predict_point2 = torch.Tensor().cuda()
    dist_source = torch.Tensor().cuda()
    dist_pred = torch.Tensor().cuda()
    for idx in range(pc1.shape[0]):
        for p1 in position1:
            p1 = p1.type(torch.int).cuda()
            points_source = torch.index_select(pc1[idx, ...], 1, p1[idx, :]).T[None, ...]
            dist_source = torch.cat((dist_source, torch.cdist(points_source, points_source).view(-1)), dim=0)
            points_pred = torch.index_select(pc1[idx, ...] + flow_pred[idx, ...], 1, p1[idx, :]).T[None, ...]
            dist_pred = torch.cat((dist_pred, torch.cdist(points_pred, points_pred).view(-1)), dim=0)

    loss = F.mse_loss(dist_pred, dist_source)

    return loss * coeff


def biomechanical_loss(constraint, flow, flow_pred, idx, pc1, coeff=1):
    source = pc1[idx, :, constraint[idx]]
    predicted = pc1[idx, :, constraint[idx]] + flow_pred[idx, :, constraint[idx]]
    # loss = torch.tensor([0.0], device=flow_pred.device, dtype=flow_pred.dtype)
    loss = torch.abs(torch.linalg.norm(source[:, 0::2] - source[:, 1::2], dim=0) - torch.linalg.norm(
        predicted[:, 0::2] - predicted[:, 1::2], dim=0)).mean()
    # for j in range(0, constraint.size(1) - 1, 2):
    #     loss += torch.abs(torch.linalg.norm(source[:, j] - source[:, j + 1]) -
    #                       torch.linalg.norm(predicted[:, j] - predicted[:, j + 1]))
    return loss * coeff


def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)  # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05) * mask, (error / gtflow_len <= 0.05) * mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1) * mask, (error / gtflow_len <= 0.1) * mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2
