from dataclasses import dataclass
from typing import Optional

from helpers_my import my_loss
from ptsemseg.models.registry import AUX_OUTPUT_MODELS


@dataclass
class TrainingLosses:
    total: object
    segmentation: object
    centerline: object
    afm: Optional[object] = None


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
    return my_loss.L1_loss(
        x_est=model_outputs.centerline,
        x_gt=gt_labelmap_centerline,
        n_chann=1,
        b_sigmoid=True,
    )


def compute_afm_loss(model_outputs, gt_afm):
    if model_outputs.afm is None:
        return None

    return my_loss.L1_loss(
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
