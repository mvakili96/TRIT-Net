"""Execution setup helpers for the legacy demo/eval runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional

import cv2
import nums_from_string
import numpy as np

from ptsemseg.evaluation import calculate_demo_eval_segmentation_iou
from ptsemseg.evaluation import create_VSAObject_from_PE_results
from ptsemseg.evaluation import evaluate_demo_eval_image
from ptsemseg.evaluation import load_demo_eval_ground_truth_inputs
from ptsemseg.evaluation import save_demo_eval_result_images
from ptsemseg.inference.path_extraction import PathExtraction_TPEnet
from ptsemseg.inference.preprocessing import read_demo_eval_image_uint8


@dataclass
class DemoEvalExecutionContext:
    """Resolved input list and processor used by the demo/eval loop."""

    list_fnames_img: List[str]
    path_extractor: PathExtraction_TPEnet


def initialize_demo_eval_execution(
    args_oper: Any,
    args_alg: Any,
    num_seg_classes: int,
    num_channel_reg: int,
    seg_in_pp: bool,
    architecture: int,
) -> DemoEvalExecutionContext:
    """Preserve legacy input discovery and path extractor construction."""

    print("Process all image inside : {}".format(args_oper.dir_input))

    list_fnames_img = os.listdir(args_oper.dir_input)
    list_fnames_img.sort(key=lambda f: int(re.sub(r"\D", "", f)))

    path_extractor = PathExtraction_TPEnet(
        args_alg,
        num_seg_classes,
        num_channel_reg,
        seg_in_pp,
        architecture,
    )

    return DemoEvalExecutionContext(
        list_fnames_img=list_fnames_img,
        path_extractor=path_extractor,
    )


def process_demo_eval_image(
    my_idx: int,
    fname_img_in: str,
    list_fnames_img: List[str],
    args_oper: Any,
    path_extractor: PathExtraction_TPEnet,
    data_in_use: int,
    num_seg_classes: int,
    flag_miou: bool,
    architecture: int,
    list_pathlabel_gt_in: List[Any],
    obj_helper_GT: Any,
    output_subdirs: Dict[str, str],
    flag_save_img: int,
) -> Optional[Dict[str, Any]]:
    """Run one legacy demo/eval image iteration and return its metric record."""

    full_fname_img_ori = os.path.join(args_oper.dir_input, fname_img_in)
    print("Read Input Image from : {}".format(full_fname_img_ori))

    img_raw_rsz_uint8 = read_demo_eval_image_uint8(
        full_fname_img_ori,
        args_oper.size_img_process,
    )
    img_raw_this = copy.deepcopy(img_raw_rsz_uint8)
    # img_raw_rsz_uint8 = cv2.rotate(img_raw_rsz_uint8, cv2.ROTATE_180)

    list_res_paths, \
    dict_res_time, \
    dict_res_imgs, \
    img_res_center_combined, \
    img_res_seg, \
    model_seg_output, \
    model_cen_output, \
    img_res_centerness, \
    img_res_AFM_direct = path_extractor.process(img_raw_rsz_uint8)    # img_raw_rsz_uint8: sensor data

    labels_seg_predicted = np.squeeze(model_seg_output.data.max(1)[1].cpu().numpy(), axis=0)
    if args_oper.size_img_process["h"] != 540:
        # labels_seg_predicted = cv2.resize(labels_seg_predicted.astype(float), (540, 960))
        img_raw_rsz_uint8 = cv2.resize(img_raw_rsz_uint8, (960, 540))

    # centerness_image_on_raw_image = path_extractor.show_centerness_on_raw_image(img_raw_rsz_uint8,img_res_center_combined)
    # cv2.imshow('centerness_image_on_raw_image', centerness_image_on_raw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    final_im = path_extractor.show_final_path_on_ori_v0(list_res_paths, img_raw_rsz_uint8)
    # path_extractor.show_final_path_on_ori_v1(list_res_paths, img_raw_rsz_uint8)
    # path_extractor.show_final_path_on_ipm(list_res_paths, img_raw_rsz_uint8)

    # cv2.imshow('final_im_kang', final_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if dict_res_time is not None:
        print("    duration [part-a: feedforward in net] (%f)(s)" % dict_res_time["dtime_ab"])
        print("    duration [part-b: decode & visualize] (%f)(s)" % dict_res_time["dtime_bc"])

    if dict_res_imgs is not None:
        path_extractor.show_imgs_res_interim(args_oper.dir_output, fname_img_in,
                                             dict_res_imgs["img_raw_in"], dict_res_imgs["img_res_seg"],
                                             dict_res_imgs["img_res_centerness_combined"], dict_res_imgs["img_res_triplet_localmax"],
                                             args_oper.b_save_res_imgs_as_file)

    create_VSAObject_from_PE_results(list_res_paths)

    img_idx = nums_from_string.get_nums(list_fnames_img[my_idx])[0]
    metric_record = None
    if data_in_use != 4:
        Class_0 = 0
        Class_1 = 0
        Class_2 = 0
        Class_3 = 0
        ground_truth_inputs = load_demo_eval_ground_truth_inputs(
            data_in_use=data_in_use,
            num_seg_classes=num_seg_classes,
            my_idx=my_idx,
            img_idx=img_idx,
            list_pathlabel_gt_in=list_pathlabel_gt_in,
            obj_helper_GT=obj_helper_GT,
        )
        gt_final_dict_xs_img_rail_LR = ground_truth_inputs["gt_final_dict_xs_img_rail_LR"]
        gt_segmentation = ground_truth_inputs["gt_segmentation"]
        GT_3 = ground_truth_inputs["GT_3"]

        segmentation_iou = calculate_demo_eval_segmentation_iou(
            data_in_use=data_in_use,
            flag_miou=flag_miou,
            gt_segmentation=gt_segmentation,
            labels_seg_predicted=labels_seg_predicted,
            image_height=img_raw_rsz_uint8.shape[0],
            image_width=img_raw_rsz_uint8.shape[1],
            num_seg_classes=num_seg_classes,
            class_0=Class_0,
            class_1=Class_1,
            class_2=Class_2,
            class_3=Class_3,
            gt_3=GT_3,
        )
        gt_segmentation = segmentation_iou["gt_segmentation"]
        Class_0 = segmentation_iou["Class_0"]
        Class_1 = segmentation_iou["Class_1"]
        Class_2 = segmentation_iou["Class_2"]
        Class_3 = segmentation_iou["Class_3"]
        GT_3 = segmentation_iou["GT_3"]

        image_showing_evaluation_res, metric_record = evaluate_demo_eval_image(
            gt_final_dict_xs_img_rail_LR=gt_final_dict_xs_img_rail_LR,
            list_res_paths=list_res_paths,
            final_im=final_im,
            image_height=img_raw_rsz_uint8.shape[0],
            image_width=img_raw_rsz_uint8.shape[1],
            architecture=architecture,
            my_idx=my_idx,
            img_idx=img_idx,
            class_0=Class_0,
            class_1=Class_1,
            class_2=Class_2,
            class_3=Class_3,
            gt_3=GT_3,
            dict_res_time=dict_res_time,
        )
    else:
        # pass
        image_showing_evaluation_res = path_extractor.show_final_path_on_ori_noGTdata(img_raw_rsz_uint8, list_res_paths)

    # cv2.imshow('final_res', image_showing_evaluation_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if flag_save_img == 1:
        save_demo_eval_result_images(
            output_subdirs=output_subdirs,
            img_idx=img_idx,
            image_showing_evaluation_res=image_showing_evaluation_res,
            img_res_seg=img_res_seg,
            img_res_centerness=img_res_centerness,
            img_res_AFM_direct=img_res_AFM_direct,
        )

    return metric_record


__all__ = [
    "DemoEvalExecutionContext",
    "initialize_demo_eval_execution",
    "process_demo_eval_image",
]
