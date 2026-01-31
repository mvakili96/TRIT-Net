import torch
import torch.nn as nn
import torch.nn.functional as F


def L1_loss(x_est, x_gt, n_chann, b_sigmoid=False):
    if n_chann == 3:
        peak_val = 1 - 1e-4
    elif n_chann == 1:
        peak_val = 100

    if b_sigmoid is True:
        x_est = torch.clamp(x_est, min=1e-4, max=peak_val)

    loss_a = nn.L1Loss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b


def MSE_loss(x_est, x_gt):
    x_est = torch.clamp(torch.sigmoid(x_est), min=1e-4, max=1 - 1e-4)
    loss_a = nn.MSELoss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b


def neg_loss(preds, targets):
    param_gamma = 4

    size_preds = preds.size()
    totnum_pixels = size_preds[0]*size_preds[1]*size_preds[2]*size_preds[3]

    pos_inds = targets.ge(0.5).float()          # pos_inds: (bs, 1, h, w)
    neg_inds = 1.0 - pos_inds                   # neg_inds: (bs, 1, h, w)

    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        pos_loss = -1.0 * torch.log(pred) * torch.pow(1 - pred, param_gamma) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * torch.pow(pred, param_gamma) * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (pos_loss + neg_loss)

    return loss / totnum_pixels


def neg_loss_cb(preds, targets):
    param_gamma = 2

    size_preds = preds.size()

    pos_inds = targets.ge(0.5).float()          # pos_inds: (bs, 1, h, w)
    neg_inds = 1.0 - pos_inds                   # neg_inds: (bs, 1, h, w)

    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()
    totnum_all = num_pos + num_neg

    alpha_pos  = num_neg/totnum_all
    alpha_neg  = num_pos/totnum_all



    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = -1.0 * torch.log(pred) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (alpha_pos*pos_loss + alpha_neg*neg_loss)

    return loss/totnum_all


def _neg_loss_ver0(preds, targets):
    pos_inds = targets == 1     
    neg_inds = targets < 1     
    neg_weights = torch.pow(1 - targets[neg_inds], 4)
    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)




