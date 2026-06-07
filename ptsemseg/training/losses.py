from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from ptsemseg.models.registry import AUX_OUTPUT_MODELS


@dataclass
class TrainingLosses:
    total: object
    segmentation: object
    centerline: object
    afm: Optional[object] = None


def L1_loss(x_est: torch.Tensor, x_gt: torch.Tensor, n_chann: int, b_sigmoid: bool = False) -> torch.Tensor:
    """Compute an L1 loss with optional clamping for sigmoid outputs."""
    if n_chann == 3:
        peak_val = 1 - 1e-4
    elif n_chann == 1:
        peak_val = 100

    if b_sigmoid is True:
        x_est = torch.clamp(x_est, min=1e-4, max=peak_val)

    loss_a = nn.L1Loss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b


def MSE_loss(x_est: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """Mean-squared error after sigmoid-clamping of estimates."""
    x_est = torch.clamp(torch.sigmoid(x_est), min=1e-4, max=1 - 1e-4)
    loss_a = nn.MSELoss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b


def neg_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Focal-like negative log-likelihood used for centerline supervision."""
    param_gamma = 4

    size_preds = preds.size()
    totnum_pixels = size_preds[0] * size_preds[1] * size_preds[2] * size_preds[3]

    pos_inds = targets.ge(0.5).float()
    neg_inds = 1.0 - pos_inds

    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        pos_loss = -1.0 * torch.log(pred) * torch.pow(1 - pred, param_gamma) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * torch.pow(pred, param_gamma) * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (pos_loss + neg_loss)

    return loss / totnum_pixels


def neg_loss_cb(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-balanced negative log-likelihood variant."""
    param_gamma = 2

    pos_inds = targets.ge(0.5).float()
    neg_inds = 1.0 - pos_inds

    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()
    totnum_all = num_pos + num_neg

    alpha_pos = num_neg / totnum_all
    alpha_neg = num_pos / totnum_all

    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = -1.0 * torch.log(pred) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (alpha_pos * pos_loss + alpha_neg * neg_loss)

    return loss / totnum_all


def _neg_loss_ver0(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Legacy neg loss implementation kept for reference."""
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


def compute_segmentation_loss(loss_fn, model_outputs, gt_imgs_label_seg, device, arch: str):
    if arch not in AUX_OUTPUT_MODELS:
        return loss_fn(input=model_outputs.segmentation, target=gt_imgs_label_seg, dev=device)

    loss_seg_unique = loss_fn(input=model_outputs.segmentation, target=gt_imgs_label_seg, dev=device)
    loss_seg = (
        loss_fn(input=model_outputs.aux_outputs[0], target=gt_imgs_label_seg, dev=device)
        + loss_fn(input=model_outputs.aux_outputs[1], target=gt_imgs_label_seg, dev=device)
        + loss_fn(input=model_outputs.aux_outputs[2], target=gt_imgs_label_seg, dev=device)
        + loss_fn(input=model_outputs.aux_outputs[3], target=gt_imgs_label_seg, dev=device)
        + loss_seg_unique
    )
    loss_seg = loss_seg_unique
    return loss_seg


def compute_centerline_loss(model_outputs, gt_labelmap_centerline):
    return L1_loss(
        x_est=model_outputs.centerline,
        x_gt=gt_labelmap_centerline,
        n_chann=1,
        b_sigmoid=True,
    )


def compute_afm_loss(model_outputs, gt_afm):
    if model_outputs.afm is None:
        return None

    return L1_loss(
        x_est=model_outputs.afm,
        x_gt=gt_afm,
        n_chann=1,
        b_sigmoid=True,
    )


def compute_training_losses(
    loss_fn,
    model_outputs,
    gt_imgs_label_seg,
    gt_labelmap_centerline,
    gt_afm,
    device,
    arch: str,
    n_classes_segmentation: int,
):
    loss_seg = compute_segmentation_loss(
        loss_fn=loss_fn,
        model_outputs=model_outputs,
        gt_imgs_label_seg=gt_imgs_label_seg,
        device=device,
        arch=arch,
    )
    loss_centerline = compute_centerline_loss(
        model_outputs=model_outputs,
        gt_labelmap_centerline=gt_labelmap_centerline,
    )

    if n_classes_segmentation == 4:
        loss_afm = compute_afm_loss(model_outputs=model_outputs, gt_afm=gt_afm)
        loss_total = 1 * loss_seg + 1 * loss_centerline + 1 * loss_afm
        return TrainingLosses(
            total=loss_total,
            segmentation=loss_seg,
            centerline=loss_centerline,
            afm=loss_afm,
        )

    loss_total = 1 * loss_seg + 1 * loss_centerline
    return TrainingLosses(
        total=loss_total,
        segmentation=loss_seg,
        centerline=loss_centerline,
        afm=None,
    )
