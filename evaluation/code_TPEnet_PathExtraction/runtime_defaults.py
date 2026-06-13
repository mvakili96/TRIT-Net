"""Centralized runtime defaults for the copied demo/eval repo.

This module intentionally preserves the copied repo's current behavior while
reducing duplicated hard-coded constants across the demo/eval entry points and
helpers. Paths remain relative to ``evaluation/code_TPEnet_PathExtraction/``.
"""

from __future__ import annotations

import copy
import os
import sys
from functools import lru_cache
from typing import Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ptsemseg.inference.config import get_default_demo_eval_config_path
from ptsemseg.inference.config import load_demo_eval_config


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


@lru_cache(maxsize=1)
def _load_demo_eval_config_cached() -> Dict:
    try:
        return load_demo_eval_config()
    except FileNotFoundError:
        return {}


def _get_config_value(default, *path):
    cfg = _load_demo_eval_config_cached()
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return copy.deepcopy(cur)


def _get_algorithm_defaults_key(dataset_for_use: int) -> str:
    if dataset_for_use == 0:
        return "dataset_0"
    return "dataset_other"


def get_loaded_demo_eval_config() -> Dict:
    return copy.deepcopy(_load_demo_eval_config_cached())


def get_demo_eval_config_path() -> str:
    return get_default_demo_eval_config_path()


def get_default_processing_size(architecture: int) -> Dict[str, int]:
    size_from_runtime = _get_config_value(None, "runtime", "operation", "size_img_process")
    if size_from_runtime is not None:
        return dict(size_from_runtime)

    size_from_cfg = _get_config_value(None, "network_image_sizes", architecture)
    if size_from_cfg is not None:
        return dict(size_from_cfg)

    return dict(ARCHITECTURE_TO_IMAGE_SIZE[architecture])


def get_model_arch_for_code(architecture: int) -> str:
    return ARCHITECTURE_TO_MODEL_ARCH[architecture]


def get_default_input_dir(dataset_for_use: int) -> str:
    configured = _get_config_value(None, "runtime", "operation", "dir_input")
    if configured is not None:
        return configured

    configured = _get_config_value(None, "dataset_defaults", "input_dirs", dataset_for_use)
    if configured is not None:
        return configured

    return DATASET_TO_INPUT_DIR[dataset_for_use]


def get_default_output_dir() -> str:
    configured = _get_config_value(None, "runtime", "operation", "dir_output")
    if configured is not None:
        return configured

    return DEFAULT_OUTPUT_DIR


def get_default_algorithm_file_weight(dataset_for_use: int) -> str:
    configured = _get_config_value(None, "dataset_defaults", "algorithm_file_weight_by_dataset", dataset_for_use)
    if configured is not None:
        return configured

    if dataset_for_use == 0:
        return ALGORITHM_DEFAULT_FILE_WEIGHT_BY_DATASET[0]
    return ALGORITHM_DEFAULT_FILE_WEIGHT_BY_DATASET[1]


def get_camera_calibration_path() -> str:
    configured = _get_config_value(None, "paths", "camera_calibration")
    if configured is not None:
        return configured

    return DEFAULT_CAMERA_CALIB_PATH


def get_demo_preset(title_testrun_this: str) -> Dict[str, object]:
    configured = _get_config_value(None, "demo_presets", title_testrun_this)
    if configured is not None:
        return dict(configured)

    return dict(DEMO_PRESETS[title_testrun_this])


def get_metrics_output_dir() -> str:
    configured = _get_config_value(None, "paths", "metrics_output_dir")
    if configured is not None:
        return configured

    return "./Performance Metrics"


def get_output_subdirs() -> Dict[str, str]:
    configured = _get_config_value(None, "paths", "output_subdirs")
    if configured is not None:
        return dict(configured)

    return {"img": "IMG", "seg": "SEG", "cen": "CEN", "afm": "AFM"}


def get_operation_runtime_defaults(dataset_for_use: int, architecture: int) -> Dict[str, object]:
    defaults = {
        "size_img_process": get_default_processing_size(architecture),
        "dir_input": get_default_input_dir(dataset_for_use),
        "dir_output": get_default_output_dir(),
        "b_save_res_imgs_as_file": False,
    }
    configured = _get_config_value(None, "runtime", "operation")
    if configured is not None:
        defaults.update(configured)
    return defaults


def get_algorithm_runtime_defaults(dataset_for_use: int) -> Dict[str, object]:
    defaults_key = _get_algorithm_defaults_key(dataset_for_use)
    configured = _get_config_value(None, "algorithm_defaults", defaults_key)
    if configured is not None:
        return dict(configured)

    if dataset_for_use == 0:
        return {
            "b_create_imgs_res_interim": False,
            "file_weight": get_default_algorithm_file_weight(dataset_for_use),
            "param_triplet_nms_alpha": 235.0 / 270.0,
            "param_triplet_nms_beta": -220.0,
            "param_triplet_nms_min": 15.0,
            "param_triplet_nms_scale": 0.1,
            "param_3D_ipm_camera_pitch_angle": -90.0 - 3.5,
            "param_3D_ipm_camera_pos_wrt_ground_plane": [0.0, 0.0, 1.5],
            "param_3D_ipm_img_pixel_per_meter": 20.0,
            "param_3D_ipm_img_height": 1000,
            "param_3D_ipm_img_width": 400,
            "param_rpg_subedge_thres_dx_3d": 10.0,
            "param_rpg_subedge_thres_dy_img": 10,
            "param_rpg_subedge_height_section": 5,
            "param_rpg_nodeedge_thres_dist_img_for_seed": 296,
            "param_rpg_nodeedge_thres_dx_3d": 25.0,
            "param_rpg_nodeedge_thres_dy_img": 20,
            "param_rpg_path_vertices_valid_y_min": 100.0,
            "param_rpg_poly_fitting_y_max": 150.0,
            "param_rpg_poly_fitting_degree": 2,
        }

    return {
        "b_create_imgs_res_interim": False,
        "file_weight": get_default_algorithm_file_weight(dataset_for_use),
        "param_triplet_nms_alpha": 145.0 / 270.0,
        "param_triplet_nms_beta": -140.0,
        "param_triplet_nms_min": 5.0,
        "param_triplet_nms_scale": 0.5,
        "param_3D_ipm_camera_pitch_angle": -90.0,
        "param_3D_ipm_camera_pos_wrt_ground_plane": [0.0, 0.0, 1.75],
        "param_3D_ipm_img_pixel_per_meter": 20.0,
        "param_3D_ipm_img_height": 1000,
        "param_3D_ipm_img_width": 400,
        "param_rpg_subedge_thres_dx_3d": 0.5,
        "param_rpg_subedge_thres_dy_img": 10,
        "param_rpg_subedge_height_section": 5,
        "param_rpg_nodeedge_thres_dist_img_for_seed": 96,
        "param_rpg_nodeedge_thres_dx_3d": 1.0,
        "param_rpg_nodeedge_thres_dy_img": 20,
        "param_rpg_path_vertices_valid_y_min": 10.0,
        "param_rpg_poly_fitting_y_max": 150.0,
        "param_rpg_poly_fitting_degree": 2,
    }


def get_demo_runtime_settings() -> Dict[str, object]:
    settings = {
        "title_testrun_this": "TEST7_RUN2_NRS_GOPRO",
        "architecture": ARCH_SEGHARDNET,
        "architecture_name": get_model_arch_for_code(ARCH_SEGHARDNET),
        "num_seg_classes": 4,
        "num_channel_reg": 1,
        "seg_in_pp": False,
        "flag_miou": False,
        "flag_save_img": 1,
        "flag_save_data": 1,
        "flag_single_multiple_path_evaluation": 1,
        "data_in_use": 0,
        "dataset_for_use": 0,
    }

    model_cfg = _get_config_value({}, "model")
    runtime_cfg = _get_config_value({}, "runtime")

    if model_cfg:
        settings["architecture"] = model_cfg.get("architecture_code", settings["architecture"])
        settings["architecture_name"] = model_cfg.get(
            "architecture_name",
            get_model_arch_for_code(settings["architecture"]),
        )
        settings["num_seg_classes"] = model_cfg.get("num_seg_classes", settings["num_seg_classes"])
        settings["num_channel_reg"] = model_cfg.get("num_channel_reg", settings["num_channel_reg"])
        settings["seg_in_pp"] = model_cfg.get("seg_in_pp", settings["seg_in_pp"])

    if runtime_cfg:
        settings["title_testrun_this"] = runtime_cfg.get("title_testrun_this", settings["title_testrun_this"])
        settings["data_in_use"] = runtime_cfg.get("data_in_use", settings["data_in_use"])
        settings["dataset_for_use"] = runtime_cfg.get("dataset_for_use", settings["dataset_for_use"])
        settings["flag_miou"] = runtime_cfg.get("flag_miou", settings["flag_miou"])
        settings["flag_save_img"] = runtime_cfg.get("flag_save_img", settings["flag_save_img"])
        settings["flag_save_data"] = runtime_cfg.get("flag_save_data", settings["flag_save_data"])
        settings["flag_single_multiple_path_evaluation"] = runtime_cfg.get(
            "flag_single_multiple_path_evaluation",
            settings["flag_single_multiple_path_evaluation"],
        )

    settings["size_img_process"] = get_default_processing_size(settings["architecture"])
    settings["dir_input"] = get_default_input_dir(settings["data_in_use"])
    settings["dir_output"] = get_default_output_dir()
    settings["file_weight_arg_default"] = get_default_algorithm_file_weight(settings["dataset_for_use"])
    settings["file_weight_override"] = get_override_weight_path(
        settings["architecture"],
        settings["num_seg_classes"],
        settings["num_channel_reg"],
    )
    settings["camera_calibration_path"] = get_camera_calibration_path()
    settings["metrics_output_dir"] = get_metrics_output_dir()
    settings["output_subdirs"] = get_output_subdirs()
    settings["demo_preset"] = get_demo_preset(settings["title_testrun_this"])

    return settings


def get_override_weight_path(net_type: int, num_seg_classes: int, num_channel_reg: int) -> str:
    arch_name = get_model_arch_for_code(net_type)
    configured = _get_config_value(None, "checkpoint", "override_file_weight_by_architecture", arch_name)
    if configured is not None:
        class_config = configured.get(num_seg_classes, configured.get("any"))
        if class_config is not None:
            channel_config = class_config.get(num_channel_reg, class_config.get("any"))
            if channel_config is not None:
                return channel_config

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
