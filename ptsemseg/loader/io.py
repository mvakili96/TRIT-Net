import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from torchvision.transforms import ToTensor

from ptsemseg.models.registry import DLINKNET_PREPROCESS_MODELS
from ptsemseg.models.registry import RGB_MEAN_STD_PREPROCESS_MODELS
from ptsemseg.models.registry import TO_TENSOR_PREPROCESS_MODELS


def read_img_raw_jpg_from_file(
    full_fname_img_raw_jpg: str,
    size_img_rsz: Dict[str, int],
    arch: str,
    rgb_mean: Optional[List[float]],
    rgb_std: Optional[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Read and resize an RGB image, then normalize to model input."""
    img_raw = cv2.imread(full_fname_img_raw_jpg)

    if img_raw is None or img_raw.size == 0:
        fsz = os.path.getsize(full_fname_img_raw_jpg)
        raise RuntimeError(
            f"[read_img_raw_jpg_from_file] Failed to read image: {full_fname_img_raw_jpg} "
            f"(cv2.imread returned None/empty, size_on_disk={fsz} bytes)"
        )

    img_raw_rsz_uint8 = cv2.resize(img_raw, (size_img_rsz["w"], size_img_rsz["h"]))
    img_raw_rsz_fl_n_final = convert_img_ori_to_img_data(
        img_raw_rsz_uint8, arch, rgb_mean=rgb_mean, rgb_std=rgb_std
    )

    return img_raw_rsz_uint8, img_raw_rsz_fl_n_final


def convert_img_ori_to_img_data(
    img_ori_uint8: np.ndarray,
    arch: str,
    rgb_mean: Optional[List[float]] = None,
    rgb_std: Optional[List[float]] = None,
) -> np.ndarray:
    """Convert an original uint8 image to the model input float array."""
    if arch in RGB_MEAN_STD_PREPROCESS_MODELS:
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0
        img_ori_fl_n = img_ori_fl - rgb_mean
        img_ori_fl_n = img_ori_fl_n / rgb_std
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        img_data_fl_n_final = img_ori_fl_n.astype(np.float32)
    elif arch in DLINKNET_PREPROCESS_MODELS:
        img_data_fl_n_final = (
            np.array(img_ori_uint8, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        )
    elif arch in TO_TENSOR_PREPROCESS_MODELS:
        img_data_fl_n_final = ToTensor()(img_ori_uint8)
        img_data_fl_n_final = img_data_fl_n_final.numpy()
    else:
        raise RuntimeError("No model found for converting image to data")

    return img_data_fl_n_final


def read_label_seg_png_from_file(full_fname_label_seg_png: str, size_out: Dict[str, int]) -> np.ndarray:
    """Read a segmentation PNG as a 2D array and resize if necessary."""
    img_raw = cv2.imread(full_fname_label_seg_png, cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise FileNotFoundError(f"Could not read {full_fname_label_seg_png}")

    h, w = img_raw.shape

    if w != size_out["w"] or h != size_out["h"]:
        img_raw = cv2.resize(
            img_raw, (size_out["w"], size_out["h"]), interpolation=cv2.INTER_NEAREST
        )

    return img_raw


def read_triplet_image_from_file(full_fname_triplet_image_png: str, size_out: Dict[str, int]) -> np.ndarray:
    """Read a single-channel triplet image (centerline / AFM) and return shape (1,H,W)."""
    img_raw = cv2.imread(full_fname_triplet_image_png, cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise FileNotFoundError(f"Could not read {full_fname_triplet_image_png}")

    h, w = img_raw.shape

    if w != size_out["w"] or h != size_out["h"]:
        img_raw = cv2.resize(
            img_raw, (size_out["w"], size_out["h"]), interpolation=cv2.INTER_NEAREST
        )

    return np.array([img_raw])
