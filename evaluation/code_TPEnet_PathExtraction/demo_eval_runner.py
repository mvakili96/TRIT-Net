# 2020/8/11
# Jungwon Kang


import os
import re
import pickle
import cv2
import json
import numpy as np
import copy
import sys
import nums_from_string
import torch

from runtime_defaults import get_demo_runtime_settings
from runtime_defaults import get_demo_preset
from runtime_defaults import get_metrics_output_dir
from runtime_defaults import get_output_subdirs

from ptsemseg.evaluation import MyHelper_GT
from ptsemseg.evaluation import create_VSAObject_from_PE_results
from ptsemseg.evaluation.metrics import cen_validation
from ptsemseg.evaluation.metrics import eval_object_all_pixel_level
from ptsemseg.evaluation.metrics import eval_object_topology
from ptsemseg.evaluation.metrics import eval_seg_object
from ptsemseg.evaluation.metrics import seg_validation
from ptsemseg.inference import read_demo_eval_image_uint8

import PE_TPEnet
import my_args_TPEnet


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
    parser_oper = my_args_TPEnet.define_args_operation(data_in_use, architecture)
    parser_alg  = my_args_TPEnet.define_args_algorithm(DATASET_for_use, architecture)
    
    ### parse
    args_oper = parser_oper.parse_args()
    args_alg  = parser_alg.parse_args()
    
    ### set values for some args
    args_alg = my_args_TPEnet.set_value_for_args_algorithm(DATASET_for_use, args_alg)
    
    
    ###==================================================================================================================
    ### 2. init
    ###==================================================================================================================
    ###==================================================================================================================
    ### 3. loop
    ###==================================================================================================================
    print("Process all image inside : {}".format(args_oper.dir_input))
    
    list_fnames_img = os.listdir(args_oper.dir_input)
    list_fnames_img.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    res_eval = []
    No_GT_Class3_counter = 0
    PathExtractor = PE_TPEnet.PathExtraction_TPEnet(args_alg, num_seg_classes, num_channel_reg, seg_in_pp, architecture)
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
            if data_in_use == 3:
                GT_3    = 0.1
                dict_pathlabel_gt_this = list_pathlabel_gt_in[img_idx]
    
    
                gt_idx_time_this                  = dict_pathlabel_gt_this['idx_time_this']         # sequential index (0,1,2...)
                gt_fname_img_in_only              = dict_pathlabel_gt_this['fname_img_in_only']
                gt_raw_dict_xs_img_rail_LR        = dict_pathlabel_gt_this['dict_rail_pnt_x_img']
                gt_raw_dict_XYZ_pnt_in_cam_rail_L = dict_pathlabel_gt_this['dict_xyz_pnt_rail_left_in_cam']
                gt_raw_dict_XYZ_pnt_in_cam_rail_R = dict_pathlabel_gt_this['dict_xyz_pnt_rail_right_in_cam']
    
                gt_final_dict_xs_img_rail_LR, \
                gt_final_dict_XYZ_pnt_in_cam_rail_L,\
                gt_final_dict_XYZ_pnt_in_cam_rail_R = obj_helper_GT.get_gt_final(gt_raw_dict_xs_img_rail_LR,
                                                                                 gt_raw_dict_XYZ_pnt_in_cam_rail_L,
                                                                                 gt_raw_dict_XYZ_pnt_in_cam_rail_R)
    
    
            elif data_in_use == 2:
                gt_final_dict_xs_img_rail_LR = json.load(open("./RailDB/test/" + f"{img_idx}" + ".json", 'r'))
                if num_seg_classes == 3:
                    gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                if num_seg_classes == 4:
                    gt_segmentation = cv2.imread("./RailDB/rs19_val_link_4class+ydhr/" + f"{img_idx}" + ".png", cv2.IMREAD_GRAYSCALE)
                # GT_3 = 0.1
    
    
            elif data_in_use == 1:
                gt_final_dict_xs_img_rail_LR = json.load(open("RailSet/test/" + str(img_idx) + ".json", 'r'))
                if num_seg_classes == 3:
                    gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                if num_seg_classes == 4:
                    gt_segmentation = cv2.imread("./RailSet/rs19_val_link_4class+ydhr/" + f"{img_idx}" + ".png", cv2.IMREAD_GRAYSCALE)
    
            
            elif data_in_use == 0:
                gt_final_dict_xs_img_rail_LR = json.load(open("railsem_jsons_test_modified2/railsem_jsons_test_modified" + str(my_idx) + ".json", 'r'))
                if num_seg_classes == 3:
                    gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                if num_seg_classes == 4:
                    gt_segmentation = cv2.imread("./Direction_Map_4class/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                if num_seg_classes == 19:
                    gt_segmentation = cv2.imread("./rs19_val/rs" + f"{my_idx + 7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
    
    
            if data_in_use <= 2:
                if flag_miou:
                    gt_segmentation = cv2.resize(gt_segmentation, (img_raw_rsz_uint8.shape[1], img_raw_rsz_uint8.shape[0]))
    
                    evaluator_seg = eval_seg_object(gt_segmentation, labels_seg_predicted,
                                                    image_height=img_raw_rsz_uint8.shape[0],
                                                    image_width=img_raw_rsz_uint8.shape[1])
    
                    if num_seg_classes == 3:
                        _,Class_0 = evaluator_seg.calculate_IoU(class_this=0)               #IoU_rail_region
                        _,Class_1 = evaluator_seg.calculate_IoU(class_this=1)               #IoU_rail
                        _,Class_2 = evaluator_seg.calculate_IoU(class_this=2)               #IoU_background
                        GT_3 = 0.1
    
                    if num_seg_classes == 4:
                        _,Class_0 = evaluator_seg.calculate_IoU(class_this=0)               #Left
                        _,Class_1 = evaluator_seg.calculate_IoU(class_this=1)               #Background
                        _,Class_2 = evaluator_seg.calculate_IoU(class_this=2)               #Right
                        GT_3,Class_3 = evaluator_seg.calculate_IoU(class_this=3)             #Right
    
                    if num_seg_classes == 19:
                        _,Class_0 = evaluator_seg.calculate_IoU(class_this=12)
                        _,Class_1 = evaluator_seg.calculate_IoU(class_this=17)
                        _,Class_2 = evaluator_seg.calculate_IoU(class_this=3)
    
                else:
                    GT_3 = 0.1
    
    
            ### 3.5.1 create evaluator objects
            num_GT_paths = len(gt_final_dict_xs_img_rail_LR)
            evaluator_topolgy = eval_object_topology(gt_final_dict_xs_img_rail_LR, list_res_paths, image_height = img_raw_rsz_uint8.shape[0], image_width = img_raw_rsz_uint8.shape[1], arch=architecture)

            # evaluator_seg = eval_seg_object(gt_segmentation, labels_seg_predicted, image_height = img_raw_rsz_uint8.shape[0], image_width = img_raw_rsz_uint8.shape[1])
    
            ### 3.5.2 annotate ground-truth rail area
            annotated_im, y_minimum = evaluator_topolgy.annotate_gt(final_im)
    
            ### 3.5.3 find correspondences between ground-truth and detected rail
            matching_mat, matched_ones = evaluator_topolgy.find_matches(4, y_minimum)
    
            ### 3.5.4 find true positives, false positives, and false negatives
            TP,FP,FN = evaluator_topolgy.performance_metrics_values_TP_level(matching_mat,matched_ones)
            path_level_prec, path_level_recall = evaluator_topolgy.performance_metrics_values_path_level(matched_ones, min_rate=0)
            all_pixel_prec, all_pixel_recall = evaluator_topolgy.performance_metrics_values_all_pixel_level(matching_mat,matched_ones)
    
            if TP == 0 or y_minimum == -1:
                print("************************************************************************************")
                print(my_idx)
                print("************************************************************************************")
    
    
            # evaluator_all_pixel_level = eval_object_all_pixel_level(gt_final_dict_xs_img_rail_LR, list_res_paths, image_height= img_raw_rsz_uint8.shape[0], image_width=img_raw_rsz_uint8.shape[1])
            # all_pixel_prec, all_pixel_recall   = evaluator_all_pixel_level.find_matches(6, y_minimum)
    
            ### 3.5.5 show performance evaluation results on the annotated image
            image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V2(annotated_im, matching_mat, matched_ones)
            # image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V1(final_im, matched_ones)
            # image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V0(final_im)
    
            # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Image index: %d' % img_idx, (400, 25),
            #                                            cv2.FONT_HERSHEY_SIMPLEX,
            #                                            0.75, (255, 0, 0), 1, cv2.LINE_AA)
    
            ### 3.5.6 measure segmentation IoU
            # if num_seg_classes == 3:
            #     IoU_rail_region = evaluator_seg.calculate_IoU(class_this=0)
            #     IoU_rail        = evaluator_seg.calculate_IoU(class_this=1)
            #     IoU_background  = evaluator_seg.calculate_IoU(class_this=2)
            #
            # if num_seg_classes == 19:
            #     IoU_rail_region = evaluator_seg.calculate_IoU(class_this=12)
            #     IoU_rail        = evaluator_seg.calculate_IoU(class_this=17)
            #     IoU_background  = evaluator_seg.calculate_IoU(class_this=3)
    
            if (TP+FP) > 0:
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'TP Pixel-level Precision: %f' % (TP/(TP+FP)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (0, 0, 255), 1, cv2.LINE_AA)
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'TP Pixel-level Recall: %f' % (TP/(TP+FN)), (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (0, 0, 255), 1, cv2.LINE_AA)
                res_eval.append(
                    {"id": img_idx, "num_GT_paths": num_GT_paths, "TP": TP, "FP": FP, "FN": FN, "precision": (TP / (TP + FP)), "recall": (TP / (TP + FN)),
                     "Class_0": Class_0, "Class_1": Class_1, "Class_2": Class_2, "Class_3": Class_3, "GT_3": GT_3,
                     "path_level_prec": path_level_prec, "path_level_recall": path_level_recall,
                     "all_pixel_prec": all_pixel_prec, "all_pixel_recall": all_pixel_recall,
                     "time_net": dict_res_time["dtime_ab"], "time_pp": dict_res_time["dtime_bc"]})
            else:
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Precision: %f' % 0, (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (255, 0, 0), 1, cv2.LINE_AA)
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Recall: %f' % 0, (400, 75), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (255, 0, 0), 1, cv2.LINE_AA)
                res_eval.append(
                    {"id": img_idx, "num_GT_paths": num_GT_paths, "TP": TP, "FP": FP, "FN": FN, "precision": 0, "recall": 0,
                     "Class_0": Class_0, "Class_1": Class_1, "Class_2": Class_2, "Class_3": Class_3, "GT_3": GT_3,
                     "path_level_prec": path_level_prec, "path_level_recall": path_level_recall,
                     "all_pixel_prec": all_pixel_prec, "all_pixel_recall": all_pixel_recall,
                     "time_net": dict_res_time["dtime_ab"], "time_pp": dict_res_time["dtime_bc"]})
        else:
            # pass
            image_showing_evaluation_res = PathExtractor.show_final_path_on_ori_noGTdata(img_raw_rsz_uint8,list_res_paths)
    
    
        # cv2.imshow('final_res', image_showing_evaluation_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        ### 3.5.6 save image
        if flag_save_img == 1:
    
            cv2.imwrite(os.path.join(output_subdirs["img"], "resluting_image_" + str(img_idx) + ".jpg"), image_showing_evaluation_res)
            cv2.imwrite(os.path.join(output_subdirs["seg"], "resluting_image_" + str(img_idx) + ".bmp"), img_res_seg)
            cv2.imwrite(os.path.join(output_subdirs["cen"], "resluting_image_" + str(img_idx) + ".png"), img_res_centerness)
            cv2.imwrite(os.path.join(output_subdirs["afm"], "resluting_image_" + str(img_idx) + ".png"), img_res_AFM_direct)
    
    
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
    
    sum_prec = 0
    sum_rec = 0
    
    sum_path_prec = 0
    sum_path_recall = 0
    
    sum_all_prec = 0
    sum_all_recall = 0
    
    sum_IoU_Class_0 = 0
    sum_IoU_Class_1 = 0
    sum_IoU_Class_2 = 0
    sum_IoU_Class_3 = 0
    sum_GT_3        = 0
    
    sum_time_net = 0
    sum_time_pp  = 0
    
    if flag_save_data == 1 and data_in_use <= 3:
        with open(os.path.join(metrics_output_dir, 'precision_1.txt'), 'w') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths>1:
                    f.write('%f' % prec)
                    f.write("\n")
    
                sum_prec = sum_prec + prec
    
                sum_IoU_Class_0   += item["Class_0"]
                sum_IoU_Class_1   += item["Class_1"]
                sum_IoU_Class_2   += item["Class_2"]
                sum_IoU_Class_3   += item["GT_3"]*item["Class_3"]
                sum_GT_3          += item["GT_3"]
    
    
                sum_path_prec   += item["path_level_prec"]
                sum_path_recall += item["path_level_recall"]
    
                sum_all_prec   += item["all_pixel_prec"]
                sum_all_recall += item["all_pixel_recall"]
    
                sum_time_net += item["time_net"]
                sum_time_pp  += item["time_pp"]
    
    
    
        with open(os.path.join(metrics_output_dir, 'recall.txt'), 'w') as f:
            for item in res_eval:
                rec = item["recall"]
                f.write('%f' % rec)
                f.write("\n")
                sum_rec = sum_rec + rec
    
        with open(os.path.join(metrics_output_dir, 'TP.txt'), 'w') as f:
            for item in res_eval:
                TP = item["TP"]
                f.write('%d' % TP)
                f.write("\n")
    
        with open(os.path.join(metrics_output_dir, 'FP.txt'), 'w') as f:
            for item in res_eval:
                FP = item["FP"]
                f.write('%d' % FP)
                f.write("\n")
    
        with open(os.path.join(metrics_output_dir, 'FN.txt'), 'w') as f:
            for item in res_eval:
                FN = item["FN"]
                f.write('%d' % FN)
                f.write("\n")
    
    
        avg_precision = sum_prec/len(res_eval)
        avg_recall = sum_rec/len(res_eval)
    
        mIoU_Class_0  = sum_IoU_Class_0 /len(res_eval)
        mIoU_Class_1  = sum_IoU_Class_1 / len(res_eval)
        mIoU_Class_2  = sum_IoU_Class_2 / len(res_eval)
        # mIoU_Class_3  = sum_IoU_Class_3 / max(1,(len(res_eval) - No_GT_Class3_counter))
        mIoU_Class_3 = sum_IoU_Class_3 / sum_GT_3
    
        # print(No_GT_Class3_counter)
    
        avg_path_precision = sum_path_prec / len(res_eval)
        avg_path_recall    = sum_path_recall / len(res_eval)
    
        avg_all_precision = sum_all_prec / len(res_eval)
        avg_all_recall    = sum_all_recall / len(res_eval)
    
        avg_time_net = sum_time_net / len(res_eval)
        avg_time_pp  = sum_time_pp / len(res_eval)
    
        print("TP PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_precision)
        print(avg_recall)
        print("SEGMENTATION PERFORMANCE")
        print(mIoU_Class_0)
        print(mIoU_Class_1)
        print(mIoU_Class_2)
        print(mIoU_Class_3)
    
        print("ALL PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_all_precision)
        print(avg_all_recall)
        print("PATH LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_path_precision)
        print(avg_path_recall)
        print("DURATION RESULTS")
        print(avg_time_net)
        print(avg_time_pp)
    
    
    
    precision_TP_1   = []
    recall_TP_1      = []
    precision_all_1  = []
    recall_all_1     = []
    precision_path_1 = []
    recall_path_1    = []
    
    
    if flag_single_multiple_path_evaluation == 1:
        with open(os.path.join(metrics_output_dir, 'precision_TP_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_TP_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_TP_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_TP_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_all_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["all_pixel_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_all_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_all_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["all_pixel_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_all_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_path_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["path_level_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_path_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_path_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["path_level_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_path_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
    
    
        precision_TP_multi   = []
        recall_TP_multi      = []
        precision_all_multi  = []
        recall_all_multi     = []
        precision_path_multi = []
        recall_path_multi    = []
    
        with open(os.path.join(metrics_output_dir, 'precision_TP_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_TP_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_TP_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_TP_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_all_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["all_pixel_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_all_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_all_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["all_pixel_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_all_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_path_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["path_level_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_path_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_path_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["path_level_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_path_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
    
        print("######################################")
        print("single-track versus multi-track evaluation")
        print("######################################")
        
        print("TP SINGLE PRECISION")
        print(np.mean(precision_TP_1))
        print("TP SINGLE RECALL")
        print(np.mean(recall_TP_1))
        print("ALL SINGLE PRECISION")
        print(np.mean(precision_all_1))
        print("ALL SINGLE RECALL")
        print(np.mean(recall_all_1))
        print("PATH SINGLE PRECISION")
        print(np.mean(precision_path_1))
        print("PATH SINGLE RECALL")
        print(np.mean(recall_path_1))
        print("***************************************")
        print("TP MULTI PRECISION")
        print(np.mean(precision_TP_multi))
        print("TP MULTI RECALL")
        print(np.mean(recall_TP_multi))
        print("ALL MULTI PRECISION")
        print(np.mean(precision_all_multi))
        print("ALL MULTI RECALL")
        print(np.mean(recall_all_multi))
        print("PATH MULTI PRECISION")
        print(np.mean(precision_path_multi))
        print("PATH MULTI RECALL")
        print(np.mean(recall_path_multi))  
    
    
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
