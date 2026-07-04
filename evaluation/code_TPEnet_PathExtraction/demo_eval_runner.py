# 2020/8/11
# Jungwon Kang


import os
import re
import pickle
import cv2
import numpy as np
import copy
import sys
import nums_from_string
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ptsemseg.evaluation import MyHelper_GT
from ptsemseg.evaluation import calculate_demo_eval_segmentation_iou
from ptsemseg.evaluation import create_VSAObject_from_PE_results
from ptsemseg.evaluation import evaluate_demo_eval_image
from ptsemseg.evaluation import load_demo_eval_ground_truth_inputs
from ptsemseg.evaluation import save_demo_eval_metric_outputs
from ptsemseg.evaluation import save_demo_eval_result_images
from ptsemseg.inference import PathExtraction_TPEnet
from ptsemseg.inference import read_demo_eval_image_uint8
from ptsemseg.inference.demo_eval_args import define_args_algorithm
from ptsemseg.inference.demo_eval_args import define_args_operation
from ptsemseg.inference.demo_eval_args import set_value_for_args_algorithm
from ptsemseg.inference.runtime_defaults import get_demo_preset
from ptsemseg.inference.runtime_defaults import get_demo_runtime_settings
from ptsemseg.inference.runtime_defaults import get_metrics_output_dir
from ptsemseg.inference.runtime_defaults import get_output_subdirs


def run_demo_eval():
    """Run the legacy TPEnet demo/eval flow."""

    ###=====================================================================================================================
    ### 0. setting
    ###=====================================================================================================================
    runtime_settings = get_demo_runtime_settings()
    
    title_testrun_this = runtime_settings["title_testrun_this"]
    
    fname_pathlabel_gt_in = None
    format_fname_img_in   = None
    format_fname_img_out  = None
    w_img = 960
    dx_valid_a = 0
    dx_valid_b = 0
    metrics_output_dir = get_metrics_output_dir()
    output_subdirs = get_output_subdirs()
    
    demo_preset = runtime_settings.get("demo_preset", get_demo_preset(title_testrun_this))
    fname_pathlabel_gt_in = demo_preset["fname_pathlabel_gt_in"]
    format_fname_img_in   = demo_preset["format_fname_img_in"]
    format_fname_img_out  = demo_preset["format_fname_img_out"]
    dx_valid_a = demo_preset["dx_valid_a"]
    dx_valid_b = demo_preset["dx_valid_b"]
    
    obj_helper_GT = MyHelper_GT(title_testrun_this, w_img, dx_valid_a, dx_valid_b)
    
    with open(fname_pathlabel_gt_in, 'rb') as fh:
        list_pathlabel_gt_in = pickle.load(fh)
    #end
    
    totnum_steps = len(list_pathlabel_gt_in)
    
    ###==================================================================================================================
    ### 1. set parameters
    ###==================================================================================================================
    architecture    = runtime_settings["architecture"]    # 0 for TPE-Net - 1 for DLink-Net34 - 2 for erfnet - 3 for BisenetV2 - 4 for segformer - 5 SegHarDNet
    
    num_seg_classes = runtime_settings["num_seg_classes"]
    num_channel_reg = runtime_settings["num_channel_reg"]
    
    seg_in_pp       = runtime_settings["seg_in_pp"]
    flag_miou       = runtime_settings["flag_miou"]
    
    flag_save_img   = runtime_settings["flag_save_img"]
    flag_save_data  = runtime_settings["flag_save_data"]
    flag_single_multiple_path_evaluation = runtime_settings["flag_single_multiple_path_evaluation"]
    
    data_in_use     = runtime_settings["data_in_use"]      # 0 for RailSem19 - 1 for RailSet - 2 for RailDB - 3 for YDHR - 4 for others without GT data
    
    ### define args
    DATASET_for_use = runtime_settings["dataset_for_use"]
    parser_oper = define_args_operation(data_in_use, architecture)
    parser_alg  = define_args_algorithm(DATASET_for_use, architecture)
    
    ### parse
    args_oper = parser_oper.parse_args()
    args_alg  = parser_alg.parse_args()
    
    ### set values for some args
    args_alg = set_value_for_args_algorithm(DATASET_for_use, args_alg)
    
    
    ###==================================================================================================================
    ### 2. init
    ###==================================================================================================================
    ###==================================================================================================================
    ### 3. loop
    ###==================================================================================================================
    print("Process all image inside : {}".format(args_oper.dir_input))
    
    list_fnames_img = os.listdir(args_oper.dir_input)
    list_fnames_img.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    
    res_eval = []
    PathExtractor = PathExtraction_TPEnet(args_alg, num_seg_classes, num_channel_reg, seg_in_pp, architecture)
    for my_idx,fname_img_in in enumerate(list_fnames_img):
    
        # if my_idx == 250:
        #     pass
        # else:
        #     continue
        ##------------------------------------------------------------------------------------------------
        ### 3-1. read img from file
        ###------------------------------------------------------------------------------------------------
        full_fname_img_ori = os.path.join(args_oper.dir_input, fname_img_in)
        print("Read Input Image from : {}".format(full_fname_img_ori))
    
        img_raw_rsz_uint8 = read_demo_eval_image_uint8(
            full_fname_img_ori,
            args_oper.size_img_process,
        )
        img_raw_this = copy.deepcopy(img_raw_rsz_uint8)
        # img_raw_rsz_uint8 = cv2.rotate(img_raw_rsz_uint8, cv2.ROTATE_180)
    
    
        ###------------------------------------------------------------------------------------------------
        ### 3-2. process
        ###------------------------------------------------------------------------------------------------
        list_res_paths, \
        dict_res_time, \
        dict_res_imgs, \
        img_res_center_combined, \
        img_res_seg,\
        model_seg_output,\
        model_cen_output, \
        img_res_centerness, \
        img_res_AFM_direct = PathExtractor.process(img_raw_rsz_uint8)    # img_raw_rsz_uint8: sensor data
    
    
        labels_seg_predicted = np.squeeze(model_seg_output.data.max(1)[1].cpu().numpy(), axis=0)
        if args_oper.size_img_process["h"] != 540:
            # labels_seg_predicted = cv2.resize(labels_seg_predicted.astype(float), (540, 960))
            img_raw_rsz_uint8    = cv2.resize(img_raw_rsz_uint8, (960, 540))
    
    
        # if flag_save_data == 1:
        #     seg_validator = seg_validation("test_seg/", my_idx, 0, PathExtractor.m_device)
        #     loss_seg = seg_validator.calculate_loss(model_seg_output, 8192, 0.3, weight=None, size_average=True)
        #     loss_seg_accum += loss_seg.item()
        #
        #     cen_validator = cen_validation("test_cen/", my_idx, PathExtractor.m_device)
        #     loss_cen = cen_validator.calculate_loss(model_cen_output)
        #     # loss_cen_regional = cen_validator.calculate_loss_regional(model_cen_output, seg_validator.GT_image_final)
        #     loss_cen_at_peaks = cen_validator.calculate_loss_at_peaks(model_cen_output)
        #     loss_cen_accum += loss_cen_at_peaks.item()
    
    
    
        ### 3-2.1 show centerness result on raw image
        # centerness_image_on_raw_image = PathExtractor.show_centerness_on_raw_image(img_raw_rsz_uint8,img_res_center_combined)
        # cv2.imshow('centerness_image_on_raw_image', centerness_image_on_raw_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        ###------------------------------------------------------------------------------------------------
        ### 3-3. visualize results
        ###------------------------------------------------------------------------------------------------
    
        ### visualize final paths
        final_im = PathExtractor.show_final_path_on_ori_v0(list_res_paths, img_raw_rsz_uint8)
        # PathExtractor.show_final_path_on_ori_v1(list_res_paths, img_raw_rsz_uint8)
        # PathExtractor.show_final_path_on_ipm(list_res_paths, img_raw_rsz_uint8)
    
        # cv2.imshow('final_im_kang', final_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        ### show time
        if dict_res_time is not None:
            print("    duration [part-a: feedforward in net] (%f)(s)" % dict_res_time["dtime_ab"])
            print("    duration [part-b: decode & visualize] (%f)(s)" % dict_res_time["dtime_bc"])
        #end
    
        ### show interim result (from TPE net only)
        if dict_res_imgs is not None:
            PathExtractor.show_imgs_res_interim(args_oper.dir_output, fname_img_in,
                                                dict_res_imgs["img_raw_in"], dict_res_imgs["img_res_seg"],
                                                dict_res_imgs["img_res_centerness_combined"], dict_res_imgs["img_res_triplet_localmax"],
                                                args_oper.b_save_res_imgs_as_file)
        #end
    
    
        ###------------------------------------------------------------------------------------------------
        ### 3.4 create VSAObject from results
        ###------------------------------------------------------------------------------------------------
        vsaobject_path = create_VSAObject_from_PE_results(list_res_paths)
    
    
        ###------------------------------------------------------------------------------------------------
        ### 3.5 PERFORMANCE METRICS CREATION
        ###------------------------------------------------------------------------------------------------
        img_idx = nums_from_string.get_nums(list_fnames_img[my_idx])[0]
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
            res_eval.append(metric_record)
        else:
            # pass
            image_showing_evaluation_res = PathExtractor.show_final_path_on_ori_noGTdata(img_raw_rsz_uint8,list_res_paths)
    
    
        # cv2.imshow('final_res', image_showing_evaluation_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        ### 3.5.6 save image
        if flag_save_img == 1:
            save_demo_eval_result_images(
                output_subdirs=output_subdirs,
                img_idx=img_idx,
                image_showing_evaluation_res=image_showing_evaluation_res,
                img_res_seg=img_res_seg,
                img_res_centerness=img_res_centerness,
                img_res_AFM_direct=img_res_AFM_direct,
            )
    
    
            # gt_final_dict_xs_img_rail_LR = json.load(open("railsem_jsons_test_modified2/railsem_jsons_test_modified" + str(my_idx) + ".json", 'r'))
            # evaluator_topolgy = eval_object_topology(gt_final_dict_xs_img_rail_LR, list_res_paths, image_height = img_raw_rsz_uint8.shape[0], image_width = img_raw_rsz_uint8.shape[1], arch=architecture)
            # annotated_im, y_minimum = evaluator_topolgy.annotate_gt(final_im)
    
    
            # multiplier  = 1
            # regression_hmap_gt_rgb = multiplier*cv2.imread('my_triplet_image/rs' +  f"{my_idx:05d}" + '.png')
            #
            # img_gt_seg = cv2.imread('rs19_val_train/rs' +  f"{my_idx:05d}" + '.png')
            # img_gt_seg = cv2.resize(img_gt_seg,(960,540))
            #
            # for i,row in enumerate(img_gt_seg):
            #     for j,col in enumerate(row):
            #         if col[0] == 1:
            #             regression_hmap_gt_rgb[i,j] = [0,200,0]
            #
            #
            # regression_hmap_rgb = cv2.cvtColor(multiplier*img_res_centerness,cv2.COLOR_GRAY2RGB)
            # for i,row in enumerate(img_res_seg):
            #     for j,col in enumerate(row):
            #         if col[0] == 232:
            #             regression_hmap_rgb[i,j] = [0,0,200]
            #
            # single_image_0 = cv2.vconcat([img_raw_this, regression_hmap_rgb])
            # # cv2.imshow("A",single_image_0)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            # single_image_final = cv2.vconcat([single_image_0, regression_hmap_gt_rgb])
            # cv2.imwrite("CEN_train/resluting_image_" + str(img_idx) + ".png", single_image_final)
    
    
    ### 3.5.7 save performance metrics computation results
    save_demo_eval_metric_outputs(
        res_eval=res_eval,
        metrics_output_dir=metrics_output_dir,
        flag_save_data=flag_save_data,
        data_in_use=data_in_use,
        flag_single_multiple_path_evaluation=flag_single_multiple_path_evaluation,
    )
    
    
    ########################################################################################################################
    ########################################################################################################################
    
    
        #---------------------------------------------------------------------------------------------------
        # list_paths_out:
        #   list_paths_out[i]: ith path, is {dict:3}
        #       'extracted': having the following
        #            -> set in def _convert_to_paths_as_vertices_v2(..):
        #            dict_path_this = {"id_edge": [],
        #                              "xy_cen_img": [],
        #                              "xy_left_img": [],
        #                              "xy_right_img": [],
        #                              "xyz_cen_3d": [],
        #                              "xyz_left_3d": [],
        #                              "xyz_right_3d": [],
        #                              ###
        #                              "id_node_switch": [],  # switch (id_node)
        #                              "xy_switch_img": [],  # switch (img)
        #                              "xyz_switch_3d": [],  # switch (3d)
        #                              ###
        #                              "xy_switch_img_edge_start": [],
        #                              "xyz_switch_3d_edge_start": [],
        #                              "xy_switch_img_edge_end": [],  # equal to "xy_switch_img"
        #                              "xyz_switch_3d_edge_end": []  # equal to "xyz_switch_3d"
        #                              }
        #
        #       'polynomial': having the following dict
        #           -> set in def _get_paths_by_polynomial_fitting(..):
        #            dict_path_poly_this = {"xyz_cen_3d": sample_arr_xyz_cen_ori,
        #                                   "xyz_left_3d": sample_arr_xyz_left_ori,
        #                                   "xyz_right_3d": sample_arr_xyz_right_ori,
        #                                   "coeff_poly_cen_3d_new": coeff_poly_cen,
        #                                   "coeff_poly_left_3d_new": coeff_poly_left,
        #                                   "coeff_poly_right_3d_new": coeff_poly_right}
        #
    #       'type_path': list_type_paths[idx_path] -> EGO or NON-EGO
    #
    #---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    run_demo_eval()
