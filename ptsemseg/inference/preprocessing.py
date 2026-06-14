"""Behavior-preserving preprocessing helpers for demo/eval integration."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from ptsemseg.inference.model_adapter import DEMO_EVAL_LOCAL_ONLY_ARCH_NAME
from ptsemseg.inference.model_adapter import get_demo_eval_architecture_name
from ptsemseg.loader.io import convert_img_ori_to_img_data as convert_training_img_to_model_input


DEMO_EVAL_DEFAULT_RGB_MEAN = np.array([113.95, 118.05, 110.18]) / 255.0
DEMO_EVAL_DEFAULT_RGB_STD = np.array([78.37, 68.79, 65.80]) / 255.0


def read_demo_eval_image_uint8(
    full_fname_img_raw_jpg: str,
    size_img_rsz: Dict[str, int],
) -> np.ndarray:
    """Read and resize an image while preserving legacy demo/eval behavior."""
    img_raw = cv2.imread(full_fname_img_raw_jpg)
    return cv2.resize(img_raw, (size_img_rsz["w"], size_img_rsz["h"]))


def convert_demo_eval_img_to_model_input(
    img_ori_uint8: np.ndarray,
    architecture_code: int,
    rgb_mean: np.ndarray = DEMO_EVAL_DEFAULT_RGB_MEAN,
    rgb_std: np.ndarray = DEMO_EVAL_DEFAULT_RGB_STD,
) -> np.ndarray:
    """Convert a demo/eval image to model input format.

    Shared architectures reuse the cleaned training repo's conversion helper.
    The copied repo's local-only ``TPEnet_a`` path stays local but follows the
    same numerical behavior the copied demo/eval code already used.
    """
    arch_name = get_demo_eval_architecture_name(architecture_code)

    if arch_name == DEMO_EVAL_LOCAL_ONLY_ARCH_NAME:
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0
        img_ori_fl_n = img_ori_fl - rgb_mean
        img_ori_fl_n = img_ori_fl_n / rgb_std
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        return img_ori_fl_n.astype(np.float32)

    return convert_training_img_to_model_input(
        img_ori_uint8,
        arch_name,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
    )
