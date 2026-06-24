"""Shared visualization helpers with verified demo/eval compatibility."""

from __future__ import annotations

import numpy as np
import torch

from ptsemseg.loader.visualization import (
    compute_centerness_from_leftright as compute_training_centerness_from_leftright,
)
from ptsemseg.loader.visualization import (
    decode_output_centerline as decode_training_output_centerline,
)

DEMO_EVAL_SEGMENTATION_RGB_LABELS = np.array(
    [
        [0, 64, 128],
        [0, 35, 35],
        [200, 170, 70],
        [240, 35, 35],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [230, 150, 140],
        [0, 0, 142],
        [0, 0, 70],
        [90, 40, 40],
        [0, 80, 100],
        [0, 254, 254],
        [0, 68, 63],
    ]
)


def decode_demo_eval_segmap_bgr_uint8(labelmap: np.ndarray) -> np.ndarray:
    """Decode labels using the legacy demo/eval BGR uint8 palette."""
    rgb = np.full((*labelmap.shape, 3), 250, dtype=np.uint8)
    for label, color in enumerate(DEMO_EVAL_SEGMENTATION_RGB_LABELS):
        rgb[labelmap == label] = color
    return rgb[:, :, ::-1]


def decode_demo_eval_relu_heatmap(res_in):
    """Decode the legacy one-channel ReLU/clamped regression output."""
    res_relu = torch.relu(res_in)
    res_a = torch.clamp(res_relu[0], min=0.0, max=100.0)
    res_out = res_a[0].detach().cpu().numpy()
    return res_out, res_out.astype(np.uint8)


def decode_demo_eval_leftright(res_in):
    """Decode legacy left/right outputs with the original 0.1 minimum clamp."""
    res0 = torch.relu(res_in)[0]

    res_left = torch.clamp(res0[0], min=0.1, max=255.0).detach().cpu().numpy()
    res_right = torch.clamp(res0[1], min=0.1, max=255.0).detach().cpu().numpy()

    return (
        res_left,
        res_right,
        res_left.astype(np.uint8),
        res_right.astype(np.uint8),
    )


def decode_demo_eval_sigmoid_heatmap(res_in):
    """Decode demo/eval sigmoid heatmaps using the verified shared behavior."""
    return decode_training_output_centerline(res_in)


def compute_demo_eval_centerness_from_leftright(
    res_left: np.ndarray,
    res_right: np.ndarray,
):
    """Reuse the cleaned training helper where behavior is identical."""
    return compute_training_centerness_from_leftright(res_left, res_right)
