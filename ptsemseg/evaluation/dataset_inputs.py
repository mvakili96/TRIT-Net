"""Dataset-specific input loading for demo/eval.

The paths in this module intentionally preserve the legacy demo/eval runner's
relative-path rules. Callers are still expected to run from
``evaluation/code_TPEnet_PathExtraction`` for the existing defaults.
"""

import json

import cv2


def load_demo_eval_ground_truth_inputs(
    data_in_use,
    num_seg_classes,
    my_idx,
    img_idx,
    list_pathlabel_gt_in,
    obj_helper_GT,
):
    """Load legacy ground-truth path labels and optional segmentation masks."""

    gt_final_dict_xs_img_rail_LR = None
    gt_segmentation = None
    gt_3 = None

    if data_in_use == 3:
        gt_3 = 0.1
        dict_pathlabel_gt_this = list_pathlabel_gt_in[img_idx]

        _gt_idx_time_this = dict_pathlabel_gt_this['idx_time_this']
        _gt_fname_img_in_only = dict_pathlabel_gt_this['fname_img_in_only']
        gt_raw_dict_xs_img_rail_LR = dict_pathlabel_gt_this['dict_rail_pnt_x_img']
        gt_raw_dict_XYZ_pnt_in_cam_rail_L = dict_pathlabel_gt_this['dict_xyz_pnt_rail_left_in_cam']
        gt_raw_dict_XYZ_pnt_in_cam_rail_R = dict_pathlabel_gt_this['dict_xyz_pnt_rail_right_in_cam']

        gt_final_dict_xs_img_rail_LR, _, _ = obj_helper_GT.get_gt_final(
            gt_raw_dict_xs_img_rail_LR,
            gt_raw_dict_XYZ_pnt_in_cam_rail_L,
            gt_raw_dict_XYZ_pnt_in_cam_rail_R,
        )

    elif data_in_use == 2:
        gt_final_dict_xs_img_rail_LR = json.load(open("./RailDB/test/" + f"{img_idx}" + ".json", 'r'))
        if num_seg_classes == 3:
            gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
        if num_seg_classes == 4:
            gt_segmentation = cv2.imread("./RailDB/rs19_val_link_4class+ydhr/" + f"{img_idx}" + ".png", cv2.IMREAD_GRAYSCALE)

    elif data_in_use == 1:
        gt_final_dict_xs_img_rail_LR = json.load(open("RailSet/test/" + str(img_idx) + ".json", 'r'))
        if num_seg_classes == 3:
            gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
        if num_seg_classes == 4:
            gt_segmentation = cv2.imread("./RailSet/rs19_val_link_4class+ydhr/" + f"{img_idx}" + ".png", cv2.IMREAD_GRAYSCALE)

    elif data_in_use == 0:
        gt_final_dict_xs_img_rail_LR = json.load(
            open("railsem_jsons_test_modified2/railsem_jsons_test_modified" + str(my_idx) + ".json", 'r')
        )
        if num_seg_classes == 3:
            gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
        if num_seg_classes == 4:
            gt_segmentation = cv2.imread("./Direction_Map_4class/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
        if num_seg_classes == 19:
            gt_segmentation = cv2.imread("./rs19_val/rs" + f"{my_idx + 7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)

    return {
        "gt_final_dict_xs_img_rail_LR": gt_final_dict_xs_img_rail_LR,
        "gt_segmentation": gt_segmentation,
        "GT_3": gt_3,
    }
