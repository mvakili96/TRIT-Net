"""Behavior-preserving preprocessing helpers for demo/eval integration."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import cv2
import numpy as np

from ptsemseg.inference.config import load_demo_eval_config
from ptsemseg.inference.model_adapter import DEMO_EVAL_LOCAL_ONLY_ARCH_NAME
from ptsemseg.inference.model_adapter import get_demo_eval_architecture_name
from ptsemseg.loader.io import convert_img_ori_to_img_data as convert_training_img_to_model_input


@lru_cache(maxsize=1)
def _get_demo_eval_rgb_stats():
    data_config = load_demo_eval_config()["data"]
    return (
        np.array(data_config["rgb_mean"]) / 255.0,
        np.array(data_config["rgb_std"]) / 255.0,
    )


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
    rgb_mean: np.ndarray | None = None,
    rgb_std: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a demo/eval image to model input format.

    Shared architectures reuse the cleaned training repo's conversion helper.
    The legacy ``TPEnet_a`` architecture name is kept as a demo/eval alias for
    shared ``rpnet_c`` and follows the same numerical behavior as the copied
    demo/eval code.
    """
    if rgb_mean is None or rgb_std is None:
        default_mean, default_std = _get_demo_eval_rgb_stats()
        if rgb_mean is None:
            rgb_mean = default_mean
        if rgb_std is None:
            rgb_std = default_std

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
