"""Demo/eval result-image writers."""

import os
import re

import cv2
import numpy as np


def _output_map_stem(fname_img_in, img_idx, index_offset):
    if not fname_img_in:
        return "resluting_image_" + str(img_idx)

    stem = os.path.splitext(os.path.basename(fname_img_in))[0]
    if index_offset == 0:
        return stem

    matches = list(re.finditer(r"\d+", stem))
    if not matches:
        raise ValueError("Cannot offset filename without a numeric index: " + fname_img_in)
    match = matches[-1]
    shifted = str(int(match.group()) + index_offset).zfill(len(match.group()))
    return stem[:match.start()] + shifted + stem[match.end():]


def save_demo_eval_result_images(
    output_subdirs,
    img_idx,
    image_showing_evaluation_res,
    img_res_seg,
    img_res_centerness,
    img_res_AFM_direct,
    fname_img_in=None,
    index_offset=0,
):
    """Write demo/eval images, naming output maps after their input image."""

    name = _output_map_stem(fname_img_in, img_idx, index_offset)
    cv2.imwrite(os.path.join(output_subdirs["img"], "resluting_image_" + str(img_idx) + ".jpg"), image_showing_evaluation_res)
    cv2.imwrite(os.path.join(output_subdirs["seg"], name + ".bmp"), img_res_seg)
    cv2.imwrite(os.path.join(output_subdirs["cen"], name + ".png"), img_res_centerness)
    if img_res_AFM_direct is not None:
        cv2.imwrite(os.path.join(output_subdirs["afm"], name + ".png"), img_res_AFM_direct)


def make_rail_area_mask(list_res_paths, image_shape):
    """Fill the area between each extracted left/right rail, including both rails."""

    mask = np.zeros(image_shape, dtype=np.uint8)
    for path in list_res_paths:
        left = np.asarray(path["extracted"]["xy_left_img"])
        right = np.asarray(path["extracted"]["xy_right_img"])
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
            continue

        count = min(len(left), len(right))
        if count == 0:
            continue

        left = np.rint(left[:count]).astype(np.int32)
        right = np.rint(right[:count]).astype(np.int32)
        if count == 1:
            cv2.line(mask, tuple(left[0]), tuple(right[0]), 255, 1)
            continue

        for idx in range(count - 1):
            polygon = np.array((left[idx], left[idx + 1], right[idx + 1], right[idx]), dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)

    return mask


def save_rail_area_mask_image(output_subdirs, img_idx, list_res_paths, image_shape, fname_img_in=None, index_offset=0):
    """Write the optional single-channel rail-area PNG alongside legacy outputs."""

    output_dir = output_subdirs.get("rail_area_mask", "MASK")
    os.makedirs(output_dir, exist_ok=True)
    mask = make_rail_area_mask(list_res_paths, image_shape)
    name = _output_map_stem(fname_img_in, img_idx, index_offset)
    output_path = os.path.join(output_dir, name + ".png")
    if not cv2.imwrite(output_path, mask):
        raise OSError("Could not write rail-area mask: " + output_path)
