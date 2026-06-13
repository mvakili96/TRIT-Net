"""Centralized runtime defaults for the copied demo/eval repo.

This module intentionally preserves the copied repo's current behavior while
reducing duplicated hard-coded constants across the demo/eval entry points and
helpers. Paths remain relative to ``evaluation/code_TPEnet_PathExtraction/``.
"""

from __future__ import annotations

from typing import Dict


ARCH_TPENET_A = 0
ARCH_DLINKNET_34 = 1
ARCH_ERFNET = 2
ARCH_BISENET_V2 = 3
ARCH_SEGFORMER = 4
ARCH_SEGHARDNET = 5


ARCHITECTURE_TO_IMAGE_SIZE = {
    ARCH_TPENET_A: {"h": 540, "w": 960},
    ARCH_DLINKNET_34: {"h": 540, "w": 960},
    ARCH_ERFNET: {"h": 540, "w": 960},
    ARCH_BISENET_V2: {"h": 540, "w": 960},
    ARCH_SEGFORMER: {"h": 544, "w": 960},
    ARCH_SEGHARDNET: {"h": 544, "w": 960},
}

ARCHITECTURE_TO_MODEL_ARCH = {
    ARCH_TPENET_A: "TPEnet_a",
    ARCH_DLINKNET_34: "dlinknet_34",
    ARCH_ERFNET: "erfnet",
    ARCH_BISENET_V2: "bisenet_v2",
    ARCH_SEGFORMER: "segformer",
    ARCH_SEGHARDNET: "seghardnet",
}

DATASET_TO_INPUT_DIR = {
    0: "./sample_input_imgs/resized_railsem",
    1: "./RailSet/jpgs",
    2: "./RailDB/jpgs",
    3: "./sample_input_imgs/ydhr",
    4: "./sample_input_imgs/nyc_yt",
}

DEFAULT_OUTPUT_DIR = "./res_imgs"
DEFAULT_CAMERA_CALIB_PATH = "./camera_calib/params_cam_calib_960_540.pickle"

ALGORITHM_DEFAULT_FILE_WEIGHT_BY_DATASET = {
    0: "./net_weight/Mybest_219gooo01.pkl",
    1: "./net_weight/TPEnet_a_weights_20800.pkl",
}

DEMO_PRESETS = {
    "TEST7_RUN2_NRS_GOPRO": {
        "fname_pathlabel_gt_in": "../in/gt_label/res_pathlabel_test7_nrs2_20210824.pickle",
        "format_fname_img_in": "../in/img_in/test7_run2_normal_reverse_switch_in_ori/img_in_%d.png",
        "format_fname_img_out": "../out/img_res_%d.png",
        "dx_valid_a": -100,
        "dx_valid_b": 100,
    }
}


def get_default_processing_size(architecture: int) -> Dict[str, int]:
    return dict(ARCHITECTURE_TO_IMAGE_SIZE[architecture])


def get_model_arch_for_code(architecture: int) -> str:
    return ARCHITECTURE_TO_MODEL_ARCH[architecture]


def get_default_input_dir(dataset_for_use: int) -> str:
    return DATASET_TO_INPUT_DIR[dataset_for_use]


def get_default_output_dir() -> str:
    return DEFAULT_OUTPUT_DIR


def get_default_algorithm_file_weight(dataset_for_use: int) -> str:
    if dataset_for_use == 0:
        return ALGORITHM_DEFAULT_FILE_WEIGHT_BY_DATASET[0]
    return ALGORITHM_DEFAULT_FILE_WEIGHT_BY_DATASET[1]


def get_camera_calibration_path() -> str:
    return DEFAULT_CAMERA_CALIB_PATH


def get_demo_preset(title_testrun_this: str) -> Dict[str, object]:
    return dict(DEMO_PRESETS[title_testrun_this])


def get_override_weight_path(net_type: int, num_seg_classes: int, num_channel_reg: int) -> str:
    if net_type == ARCH_TPENET_A:
        if num_seg_classes == 3:
            return "./net_weight/Mybest_32100.pkl"
        if num_seg_classes == 4:
            if num_channel_reg == 1:
                return "./net_weight/Association/CRV2ISPRS-runs/CASE-on-RailSem19/Mybest_10000.pkl"
            if num_channel_reg == 3:
                return "./net_weight/Mybest_9033.pkl"
        if num_seg_classes == 19:
            if num_channel_reg == 1:
                return "./net_weight/Mybest_90119.pkl"
            if num_channel_reg == 3:
                return "./net_weight/Mybest_90319.pkl"
    elif net_type == ARCH_DLINKNET_34:
        return "./net_weight/Mybest_11000.pkl"
    elif net_type == ARCH_ERFNET:
        return "./net_weight/Association/CRV2ISPRS-runs/ERFNet-on-RailSem19/Mybest_90000.pkl"
    elif net_type == ARCH_BISENET_V2:
        if num_seg_classes == 3:
            return "./net_weight/Mybest_10000.pkl"
        if num_seg_classes == 4:
            return "./net_weight/Association/CRV2ISPRS-runs/BiSeNet-on-RailSem19/Mybest_90000.pkl"
    elif net_type == ARCH_SEGFORMER:
        return "./net_weight/Association/CRV2ISPRS-runs/SegFormer-on-RailSem19/Mybest_90000.pkl"
    elif net_type == ARCH_SEGHARDNET:
        return "./net_weight/Association/SegHarDNet/cur/7/Mybest_80000.pkl"

    raise ValueError(
        "Unsupported net_type/num_seg_classes/num_channel_reg combination: "
        f"{net_type}, {num_seg_classes}, {num_channel_reg}"
    )
