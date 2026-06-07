from typing import Tuple

import numpy as np
import torch

from ptsemseg.loader.constants import IGNORE_LABEL
from ptsemseg.loader.constants import RAILSEM19_NUM_CLASSES
from ptsemseg.loader.constants import RAILSEM19_RGB_LABELS


def decode_segmap(labelmap: np.ndarray, plot: bool = False) -> np.ndarray:
    rgb_labels = np.array(RAILSEM19_RGB_LABELS)

    r = np.ones_like(labelmap) * IGNORE_LABEL
    g = np.ones_like(labelmap) * IGNORE_LABEL
    b = np.ones_like(labelmap) * IGNORE_LABEL

    for l in range(0, RAILSEM19_NUM_CLASSES):
        idx_set = (labelmap == l)
        r[idx_set] = rgb_labels[l, 0]
        g[idx_set] = rgb_labels[l, 1]
        b[idx_set] = rgb_labels[l, 2]

    img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
    img_label_rgb[:, :, 0] = r / 255.0
    img_label_rgb[:, :, 1] = g / 255.0
    img_label_rgb[:, :, 2] = b / 255.0

    return img_label_rgb


def decode_output_centerline(res_in: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Decode model centerline output to (float array, uint8 image)."""
    res_sigmoid = torch.clamp(torch.sigmoid(res_in), min=1e-4, max=1 - 1e-4)

    res_a = res_sigmoid[0].permute(1, 2, 0)
    res_b = res_a[:, :, 0]
    res_c = res_b * 255.0
    res_d = torch.clamp(res_c, min=0.0, max=255.0)
    res_e = res_d.detach().cpu().numpy()

    res_out = res_b.detach().cpu().numpy()
    img_res_out = res_e.astype(np.uint8)

    return res_out, img_res_out


def decode_output_leftright(res_in: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode left/right outputs into numpy arrays and uint8 images."""
    res_relu = torch.relu(res_in)
    res0 = res_relu[0]

    res_left_a = res0[0, :, :]
    res_left_b = torch.clamp(res_left_a, min=0.0, max=255.0)
    res_left_c = res_left_b.detach().cpu().numpy()
    img_res_left = res_left_c.astype(np.uint8)

    res_right_a = res0[1, :, :]
    res_right_b = torch.clamp(res_right_a, min=0.0, max=255.0)
    res_right_c = res_right_b.detach().cpu().numpy()
    img_res_right = res_right_c.astype(np.uint8)

    res_left = res_left_c
    res_right = res_right_c

    return res_left, res_right, img_res_left, img_res_right


def compute_centerness_from_leftright(res_left: np.ndarray, res_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centerness weight from left/right predictions."""
    res_delta = abs(res_left - res_right)
    res_sum = abs(res_left) + abs(res_right)

    res_ratio = res_delta / res_sum
    res_ratio[np.isnan(res_ratio)] = 1.0
    res_ratio[res_sum <= 1.0] = 1.0

    res_weight = 1.0 - res_ratio

    res_weight_b = res_weight * 255.0
    img_res_weight = res_weight_b.astype(np.uint8)

    return res_weight, img_res_weight
