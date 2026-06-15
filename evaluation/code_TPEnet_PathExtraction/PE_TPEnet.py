# 2020/8/11
# Jungwon Kang


import torch
import numpy as np
import time
import cv2
import copy
import math
import os
import sys

import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ptsemseg.inference import get_demo_eval_architecture_name

from helpers.models import get_model
from helpers.utils  import my_utils_img
from helpers.utils  import my_utils_net
from helpers.utils  import my_utils_3D
from helpers.utils  import my_utils_RPG
from helpers.utils  import my_utils_vis
from scipy.signal import find_peaks


########################################################################################################################
###
########################################################################################################################
class PathExtraction_TPEnet:

    def __init__(self, args, num_seg_classes, num_channel_reg, seg_in_pp, architecture):


        self.num_seg_classes = num_seg_classes
        self.num_channel_reg = num_channel_reg
        self.seg_in_pp       = seg_in_pp
        self.architecture    = architecture


        self.m_b_create_imgs_res_interim = args.b_create_imgs_res_interim



        dict_args_net, \
        dict_args_triplet, \
        dict_args_3D_ipm, \
        dict_args_rpg = self.arrange_args(args)



        self.m_obj_utils_net = my_utils_net.MyUtils_Net(dict_args_net, architecture, num_seg_classes, num_channel_reg)

        self.m_obj_utils_img = my_utils_img.MyUtils_Image(dict_args_triplet)
        self.m_obj_utils_3D  = my_utils_3D.MyUtils_3D(dict_args_3D_ipm)
        self.m_obj_utils_rpg = my_utils_RPG.MyUtils_RailPathGraph(dict_args_rpg)


        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = {"arch": get_demo_eval_architecture_name(architecture)}

        self.m_model = get_model(model_dict, n_classes=num_seg_classes, n_channels_reg=num_channel_reg)

        self.m_obj_utils_net.load_weights_to_model(self.m_model)

        self.m_model.eval()
        self.m_model.to(self.m_device)

        return


    ###=========================================================================================================
    ### arrange args
    ###=========================================================================================================
    def arrange_args(self, args):
        """
        See <my_args_TPEnet.py>
        :param args:
        :return:
        """

        ###
        dict_args_net = {}
        dict_args_triplet = {}
        dict_args_3D_ipm = {}
        dict_args_rpg = {}


        ###---------------------------------------------------------------------------------------
        ###
        ###---------------------------------------------------------------------------------------
        dict_args_net["file_weight"] = args.file_weight


        ###---------------------------------------------------------------------------------------
        ###
        ###---------------------------------------------------------------------------------------
        dict_args_triplet["param_triplet_nms_alpha"] = args.param_triplet_nms_alpha
        dict_args_triplet["param_triplet_nms_beta"]  = args.param_triplet_nms_beta
        dict_args_triplet["param_triplet_nms_min"]   = args.param_triplet_nms_min
        dict_args_triplet["param_triplet_nms_scale"] = args.param_triplet_nms_scale


        ###---------------------------------------------------------------------------------------
        ###
        ###---------------------------------------------------------------------------------------
        dict_args_3D_ipm["param_3D_ipm_camera_intrinsic_matrix"]     = args.param_3D_ipm_camera_intrinsic_matrix
        dict_args_3D_ipm["param_3D_ipm_camera_pitch_angle"]          = args.param_3D_ipm_camera_pitch_angle
        dict_args_3D_ipm["param_3D_ipm_camera_pos_wrt_ground_plane"] = args.param_3D_ipm_camera_pos_wrt_ground_plane
        dict_args_3D_ipm["param_3D_ipm_img_pixel_per_meter"]         = args.param_3D_ipm_img_pixel_per_meter
        dict_args_3D_ipm["param_3D_ipm_img_height"]                  = args.param_3D_ipm_img_height
        dict_args_3D_ipm["param_3D_ipm_img_width"]                   = args.param_3D_ipm_img_width


        ###---------------------------------------------------------------------------------------
        ###
        ###---------------------------------------------------------------------------------------
        dict_args_rpg["param_rpg_subedge_thres_dx_3d"]       = args.param_rpg_subedge_thres_dx_3d
        dict_args_rpg["param_rpg_subedge_thres_dy_img"]      = args.param_rpg_subedge_thres_dy_img
        dict_args_rpg["param_rpg_subedge_height_section"]    = args.param_rpg_subedge_height_section

        dict_args_rpg["param_rpg_nodeedge_thres_dist_img_for_seed"] = args.param_rpg_nodeedge_thres_dist_img_for_seed
        dict_args_rpg["param_rpg_nodeedge_thres_dx_3d"]             = args.param_rpg_nodeedge_thres_dx_3d
        dict_args_rpg["param_rpg_nodeedge_thres_dy_img"]            = args.param_rpg_nodeedge_thres_dy_img

        dict_args_rpg["param_rpg_path_vertices_valid_y_min"] = args.param_rpg_path_vertices_valid_y_min

        dict_args_rpg["param_rpg_poly_fitting_y_max"]        = args.param_rpg_poly_fitting_y_max
        dict_args_rpg["param_rpg_poly_fitting_degree"]       = args.param_rpg_poly_fitting_degree



        return dict_args_net, dict_args_triplet, dict_args_3D_ipm, dict_args_rpg
    #end


    ###=========================================================================================================
    ### process()
    ###=========================================================================================================
    def process(self, img_raw_rsz_uint8):

        width  = 960
        height = 540
        ###------------------------------------------------------------------------------------------------
        ### 1. input image
        ###------------------------------------------------------------------------------------------------
        img_raw_rsz_fl_n = self.m_obj_utils_img.convert_img_ori_to_img_data(img_raw_rsz_uint8, self.architecture)

        img_raw = np.expand_dims(img_raw_rsz_fl_n, 0)
        img_raw = torch.from_numpy(img_raw).float()

        images = img_raw.to(self.m_device)

        time_a = time.time()
        ###------------------------------------------------------------------------------------------------
        ### 3. do feed-forwarding to get centerness/left-right/segmentation output
        ###------------------------------------------------------------------------------------------------
        if self.num_channel_reg == 3:
            output_seg, \
            output_centerness, \
            output_leftright = self.m_model(images)

        elif self.num_channel_reg == 1:
            if self.num_seg_classes == 4:
                output_seg, \
                output_centerness, \
                output_AFM        = self.m_model(images)
            else:
                output_seg, \
                output_centerness = self.m_model(images)

        ### sample time
        time_b   = time.time()
        dtime_ab = time_b - time_a


        ###------------------------------------------------------------------------------------------------
        ### 4. decode network_output (& create img for visualization)
        ###------------------------------------------------------------------------------------------------
        labels_seg_predicted = np.squeeze(output_seg.data.max(1)[1].cpu().numpy(), axis=0)


        img_res_seg = None
        self.m_b_create_imgs_res_interim = True
        if self.m_b_create_imgs_res_interim is True:
            img_res_seg = self.m_obj_utils_img.decode_segmap(labels_seg_predicted)

            # cv2.imshow("Seg",img_res_seg)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            self.m_b_create_imgs_res_interim = False

        num_ch_reg = 1

        ### decode output_centerness
        res_centerness_direct, \
        img_res_centerness_direct = self.m_obj_utils_img.decode_output_centerness(output_centerness, num_channel_reg=num_ch_reg)

        # img = img_res_centerness_direct.astype(np.float32)

        # p1, p99 = np.percentile(img, (1, 99.99))
        # img_clip = np.clip(img, p1, p99)

        # vis = cv2.normalize(img_clip, None, 0, 255, cv2.NORM_MINMAX)
        # vis = vis.astype(np.uint8)



        # cv2.imshow("Cen", vis)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        ### decode output_AFM
        img_res_AFM_direct = None
        if self.num_seg_classes == 4:
            res_AFM_direct, \
            img_res_AFM_direct = self.m_obj_utils_img.decode_output_centerness(output_AFM, num_channel_reg=num_ch_reg)


        
        # cv2.imshow("AFM",np.uint8(2*img_res_AFM_direct))
        # cv2.waitKey()
        # cv2.destroyAllWindows()


        ### decode output_leftright
        if num_ch_reg== 3:
            res_left, \
            res_right, \
            img_res_left, \
            img_res_right = self.m_obj_utils_img.decode_output_leftright(output_leftright)
        elif num_ch_reg == 1:
            res_left  = None
            res_right = None


        ###------------------------------------------------------------------------------------------------
        ### 5. get centerness by combining centerness-direct and centerness-from-left-right)
        ###------------------------------------------------------------------------------------------------
        if num_ch_reg == 3:
            res_centerness_from_LR,\
            img_res_centerness_from_LR = self.m_obj_utils_img.compute_centerness_from_leftright(res_left, res_right)


        ###
        if num_ch_reg == 3:
            res_centerness_combined = res_centerness_direct*res_centerness_from_LR
            res_centerness_combined_b = res_centerness_combined*255.0
            img_res_centerness_combined = res_centerness_combined_b.astype(np.uint8)
        elif num_ch_reg == 1:
            res_centerness_combined = res_centerness_direct                       
            res_centerness_combined_b = res_centerness_combined                    
            img_res_centerness_combined = res_centerness_combined_b.astype(np.uint8)



        ###------------------------------------------------------------------------------------------------
        ### 6. extract triplet points
        ###------------------------------------------------------------------------------------------------
        if img_raw_rsz_uint8.shape[0] !=540:
            res_centerness_combined = cv2.resize(res_centerness_combined,(width,height))
            img_res_centerness_combined = cv2.resize(img_res_centerness_combined, (width, height))
            img_raw_rsz_uint8       = cv2.resize(img_raw_rsz_uint8,(width,height))
            img_res_seg             = cv2.resize(img_res_seg,(width,height))
            img_res_centerness_direct = cv2.resize(img_res_centerness_direct,(width,height))

            if res_left is not None:
                res_left = cv2.resize(res_left, (width, height))
                res_right = cv2.resize(res_right, (width, height))



        list_dict_triplet_pnts_local_max = self.m_obj_utils_img.extract_triplet_pnts_localmax(res_centerness_combined, res_left, res_right, self.m_obj_utils_3D)

        ### visualize result (triplet points)
        img_res_triplet_localmax = None

        if self.m_b_create_imgs_res_interim is True:
            img_res_triplet_localmax = self.m_obj_utils_img.visualize_res_triplet_localmax(img_raw_rsz_uint8, res_centerness_combined, res_left, res_right)
        #end


        ###------------------------------------------------------------------------------------------------
        ### 7. create paths from triplet points
        ###------------------------------------------------------------------------------------------------
        use_PP = False

        if self.num_seg_classes == 4 and use_PP is False:
            list_paths_final = self.m_obj_utils_img.remove_post_process(res_centerness_direct,labels_seg_predicted,img_raw_rsz_uint8, res_AFM_direct)
        
        else:
            list_paths_final = self.m_obj_utils_rpg.process(list_dict_triplet_pnts_local_max,
                                                            self.m_obj_utils_3D, img_raw_rsz_uint8, img_res_seg, self.num_seg_classes, self.seg_in_pp)

            rejected = []
            accepted = []
            for id_path in range(len(list_paths_final)-1):
                enumerator = 0
                pnts_this = list_paths_final[id_path]["extracted"]["xy_cen_img"]
                pnts_next = list_paths_final[id_path+1]["extracted"]["xy_cen_img"]
                for counter in range(min(len(pnts_this),len(pnts_next))):
                    if pnts_this[counter][0] == pnts_next[counter][0] and pnts_this[counter][1] == pnts_next[counter][1]:
                        enumerator += 1
            
                if enumerator/min(len(pnts_this),len(pnts_next)) > 0.8 and (min(len(pnts_this),len(pnts_next))/max(len(pnts_this),len(pnts_next))<0.8 or min(len(pnts_this),len(pnts_next))/max(len(pnts_this),len(pnts_next))>0.996):
                    # print(id_path)
                    # print(id_path+1)
                    # print("*****************************************")
                    if len(pnts_this)<len(pnts_next):
                        if (id_path+1 not in accepted) and (id_path+1 not in rejected):
                            accepted.append(id_path+1)
                        rejected.append(id_path)
                    else:
                        if (id_path not in accepted) and (id_path not in rejected):
                            accepted.append(id_path)
                        rejected.append(id_path+1)
            
                else:
                    if id_path not in accepted and id_path not in rejected:
                        accepted.append(id_path)
                    if id_path+1 not in accepted and id_path+1 not in rejected:
                        accepted.append(id_path+1)
            list_paths_final = [list_paths_final[i] for i in range(len(list_paths_final)) if i not in rejected]
            
            if len(list_paths_final) > 1:
                rejected = []
                accepted = []
                lengths = []
                for id_path in range(len(list_paths_final)):
                    pnts_this = list_paths_final[id_path]["extracted"]["xy_cen_img"]
                    len_this = abs(pnts_this[0][1]-pnts_this[-1][1])
                    lengths.append(len_this)
                max_len = max(lengths)
                for id_path in range(len(list_paths_final)):
                    pnts_this = list_paths_final[id_path]["extracted"]["xy_cen_img"]
                    len_this = abs(pnts_this[0][1]-pnts_this[-1][1])
                    if len_this/max_len < 0.86:
                        rejected.append(id_path)
                list_paths_final = [list_paths_final[i] for i in range(len(list_paths_final)) if i not in rejected]
            
            
            
            for id_path in range(len(list_paths_final)):
                id_start_point = -1
                pnts_left = list_paths_final[id_path]["polynomial"]['xyz_left_3d']
                pnts_right = list_paths_final[id_path]["polynomial"]['xyz_right_3d']
                for id_point in range(min(len(pnts_right),len(pnts_left))):
                    if pnts_right[id_point,0] - pnts_left[id_point,0]<0:
                        id_start_point = id_point
                if id_start_point>0:
                    list_paths_final[id_path]["polynomial"]['xyz_left_3d'] = list_paths_final[id_path]["polynomial"]['xyz_left_3d'][id_start_point:-1]
                    list_paths_final[id_path]["polynomial"]['xyz_right_3d'] = list_paths_final[id_path]["polynomial"]['xyz_right_3d'][id_start_point:-1]



        ### sample time
        time_c   = time.time()
        dtime_bc = time_c - time_b


        ###------------------------------------------------------------------------------------------------
        ### 8. output
        ###------------------------------------------------------------------------------------------------
        dict_res_time = {"dtime_ab": dtime_ab,
                         "dtime_bc": dtime_bc
                         }

        dict_res_imgs = None

        if self.m_b_create_imgs_res_interim is True:
            dict_res_imgs = {"img_raw_in"                  : img_raw_rsz_uint8,
                             "img_res_seg"                 : img_res_seg,
                             "img_res_centerness_combined" : img_res_centerness_combined,
                             "img_res_triplet_localmax"    : img_res_triplet_localmax
                             }


        return list_paths_final, \
               dict_res_time, \
               dict_res_imgs, \
               img_res_centerness_combined, \
               img_res_seg, \
               output_seg, \
               output_centerness, \
               img_res_centerness_direct, \
               img_res_AFM_direct


    ###=========================================================================================================
    ### convert_one_point_from_world_to_img
    ###=========================================================================================================
    def convert_one_point_from_world_to_img(self, x_in, y_in, z_in = 1.0):
        x_out, y_out = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(
            np.array([[x_in], [y_in], [z_in]]))
        return [x_out, y_out]

    def show_centerness_on_raw_image(self, raw_image, centerness_image):
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        gray_image_3channel = np.zeros_like(raw_image)  # img_ori_gray3: 3-ch gray img
        gray_image_3channel[:, :, 0] = gray_image
        gray_image_3channel[:, :, 1] = gray_image
        gray_image_3channel[:, :, 2] = gray_image

        # print(gray_image.shape)
        # base_added_red_channel_val = 100
        coeff_red_channel_val = 500
        for i,row in enumerate(centerness_image):
            for j,pixel in enumerate(row):
                added_red_channel_val = int( ((pixel/255)**0.6)*coeff_red_channel_val )
                if gray_image_3channel[i, j, 0] >= 255-added_red_channel_val:
                    gray_image_3channel[i, j, :] = [255, 255-added_red_channel_val, 255-added_red_channel_val]
                else:
                    gray_image_3channel[i, j, :] = gray_image_3channel[i, j, :] + [added_red_channel_val, 0, 0]

        return gray_image_3channel

    def show_final_path_on_ori_noGTdata(self, raw_image, list_paths_final):
        points_ref = []
        for detected_path_index in range(len(list_paths_final)):
            detected_3d_left = list_paths_final[detected_path_index]["polynomial"]["xyz_left_3d"]
            detected_3d_right = list_paths_final[detected_path_index]["polynomial"]["xyz_right_3d"]
            for idx_pnt in range(detected_3d_left.shape[0]):  # Loop for detected points (left) to create binary image
                x3d_detected_left = detected_3d_left[idx_pnt][0]
                y3d_detected_left = detected_3d_left[idx_pnt][1]

                x3d_detected_right = detected_3d_right[idx_pnt][0]
                y3d_detected_right = detected_3d_right[idx_pnt][1]

                ximg_detected_int_left = int(round(x3d_detected_left))
                yimg_detected_int_left = int(round(y3d_detected_left))

                ximg_detected_int_right = int(round(x3d_detected_right))
                yimg_detected_int_right = int(round(y3d_detected_right))

                if (ximg_detected_int_left < 0) or (ximg_detected_int_left >= 960) or (yimg_detected_int_left < 0) or (
                        yimg_detected_int_left >= 540) or (yimg_detected_int_left < 270):
                    continue
                if (ximg_detected_int_right < 0) or (ximg_detected_int_right >= 960) or (yimg_detected_int_right < 0) or (
                        yimg_detected_int_right >= 540) or (yimg_detected_int_right < 270):
                    continue

                for x_this in range(ximg_detected_int_left, (ximg_detected_int_right + 1)):
                    if [yimg_detected_int_left, x_this] not in points_ref:
                        if raw_image[yimg_detected_int_left, x_this, 1] >= 200:
                            raw_image[yimg_detected_int_left, x_this, :] = [200, 255, 200]
                        else:
                            raw_image[yimg_detected_int_left, x_this, :] = raw_image[yimg_detected_int_left, x_this, :] + [0, 55, 0]

                        points_ref.append([yimg_detected_int_left, x_this])
                    else:
                        continue

                cv2.circle(raw_image, center=(ximg_detected_int_left, yimg_detected_int_left), radius=2,
                           color=(255, 255, 0), thickness=-1)
                cv2.circle(raw_image, center=(ximg_detected_int_right, yimg_detected_int_right), radius=2,
                           color=(0, 70, 255), thickness=-1)
            # all_res_images.append(img_cp)

        # if len(all_res_images) == 1:
        #     single_image_final = all_res_images[0]
        # else:
        #     input_to_concat = all_res_images[0]
        #     for cnt in range(len(all_res_images) - 1):
        #         single_image_final = cv2.vconcat([input_to_concat, all_res_images[cnt + 1]])
        #         input_to_concat = single_image_final

        return raw_image
    ###=========================================================================================================
    ### visualize final path on ori-img
    ###=========================================================================================================
    def show_final_path_on_ori_v0(self, list_paths_final, img_bg, res_for_show=1):
        # ---------------------------------------------------------------------------------------------
        # dict_path_final = {"extracted": dict_path_this,
        #                    "polynomial": dict_path_poly_this}
        #
        #    "extracted": dict_path_this = {"xy_cen_img": [],
        #                                   "xy_left_img": [],
        #                                   "xy_right_img": [],
        #                                   "xyz_cen_3d": [],
        #                                   "xyz_left_3d": [],
        #                                   "xyz_right_3d": [],
        #                                   "xy_switch_img"  : [],  # switch (img)
        #                                   "xyz_switch_3d"  : [],  # switch (3d)
        #                                   "id_node_switch" : []   # switch (id_node)
        #
        #    "polynomial": dict_path_poly_this = {"xyz_cen_3d": arr_xyz_sample_ori,
        #                                         "xyz_left_3d": [],
        #                                         "xyz_right_3d": []}
        # ---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        # ---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        # ---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)
        assert (img_bg is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_ori_rgb = copy.deepcopy(img_bg)
        img_vis_ori_gray1 = cv2.cvtColor(img_vis_ori_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ori_gray3 = np.zeros_like(img_vis_ori_rgb)  # img_ori_gray3: 3-ch gray img
        img_vis_ori_gray3[:, :, 0] = img_vis_ori_gray1
        img_vis_ori_gray3[:, :, 1] = img_vis_ori_gray1
        img_vis_ori_gray3[:, :, 2] = img_vis_ori_gray1


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        h_img = img_bg.shape[0]
        w_img = img_bg.shape[1]


        ###------------------------------------------------------------------------------------------------
        ### process each path
        ###------------------------------------------------------------------------------------------------

        for idx_path in range(len(list_paths_final)):

            if idx_path == 10:
                pass
            else:
                continue

            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted = dict_extracted["xy_right_img"]

            #################################################
            # arr_xy_switch_img = dict_extracted["xy_switch_img_edge_end"]
            #################################################

            # arr_xyz_switch_3d_extracted = dict_extracted["xyz_switch_3d"]


            ###
            # arr_xyz_cen_3d_temp = dict_extracted["xyz_cen_3d"]
            # arr_y_cen_3d_temp = arr_xyz_cen_3d_temp[:, 1]
            #
            # max_y_cen_3d_extracted = max(arr_y_cen_3d_temp)


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end

            totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ###
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ###
                    x_cen_3d_this = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_3d_this = arr_xy_cen_img_extracted[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_3d_this = arr_xy_left_img_extracted[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_3d_this = arr_xy_right_img_extracted[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ###
                    x_cen_img = x_cen_3d_this
                    y_cen_img = y_cen_3d_this

                    x_left_img = x_left_3d_this
                    y_left_img = y_left_3d_this

                    x_right_img = x_right_3d_this
                    y_right_img = y_right_3d_this

                    # x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    # x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    # x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))

                    # x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    # y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]
                    #
                    # x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    # y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]
                    #
                    # x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    # y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                #end


                ###
                # if y_cen_3d_this >= max_y_cen_3d_extracted:
                #     continue
                # # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))



                ###------------------------------------------------------------------------------
                ### DRAW
                ###------------------------------------------------------------------------------
                val_b_cen = 128
                val_g_cen = 0
                val_r_cen = 0

                val_b_left = 20
                val_g_left = 100
                val_r_left = 250

                val_b_right = 250
                val_g_right = 250
                val_r_right = 0


                # for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                #     y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]
                #
                #     dy_abs = abs(y_cen_3d_this - y_switch_3d_this)
                #
                #     if dy_abs < 12.0:
                #         val_b_cen = 0
                #         val_g_cen = 0
                #         val_r_cen = 255
                #     #end
                # #end


                ### draw pnt (cen)
                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25 ):
                        cv2.circle(img_vis_ori_gray3, center=(x_cen_img_int, y_cen_img_int), radius=2,
                               color=(val_b_cen, val_g_cen, val_r_cen), thickness=-1)
                # end


                ## draw pnt (left)
                if (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25):
                        cv2.circle(img_vis_ori_gray3, center=(x_left_img_int, y_left_img_int), radius=1,
                               color=(val_b_left, val_g_left, val_r_left), thickness=-1)
                # end


                ### draw pnt (right)
                if (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25):
                        cv2.circle(img_vis_ori_gray3, center=(x_right_img_int, y_right_img_int), radius=1,
                               color=(val_b_right, val_g_right, val_r_right), thickness=-1)
                # end
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### show
        # ###------------------------------------------------------------------------------------------------
        # cv2.imshow('final_path_on_ori', img_vis_ori_gray3)
        # cv2.waitKey(0)

        ### save (temp)
        if 0:
            fname_output_temp = "./res_imgs" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.jpg'
            cv2.imwrite(fname_output_temp, img_vis_ori_gray3)
            self.m_temp_idx_res_img_on_ori += 1
        #end

        return img_vis_ori_gray3

    #end


    ###=========================================================================================================
    ### visualize final path on ori-img
    ###=========================================================================================================
    def show_final_path_on_ori_v1(self, list_paths_final, img_bg_in, res_for_show=1):
        # ---------------------------------------------------------------------------------------------------
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
        # ---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        # ---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        # ---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)
        assert (img_bg_in is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_res = copy.deepcopy(img_bg_in)

        h_img = img_vis_res.shape[0]
        w_img = img_vis_res.shape[1]


        ###------------------------------------------------------------------------------------------------
        ### set path sequence for drawing (for first drawing non-ego paths, then drawing ego path)
        ###------------------------------------------------------------------------------------------------
        list_idx_path_for_drawing = []

        ### for non-ego path
        for idx_path in range(len(list_paths_final)):
            type_path_this = list_paths_final[idx_path]["type_path"]

            if type_path_this is my_utils_RPG.TYPE_path.NON_EGO:
                list_idx_path_for_drawing.append(idx_path)
            #end
        #end

        ### for ego path
        for idx_path in range(len(list_paths_final)):
            type_path_this = list_paths_final[idx_path]["type_path"]

            if type_path_this is my_utils_RPG.TYPE_path.EGO:
                list_idx_path_for_drawing.append(idx_path)
            #end
        #end
            # completed to set
            #   list_idx_path_for_drawing




        ###------------------------------------------------------------------------------------------------
        ### step 0: draw (filled) path-region
        ###------------------------------------------------------------------------------------------------
        for idx_draw in range(len(list_paths_final)):
            idx_path = list_idx_path_for_drawing[idx_draw]
            type_path_this = list_paths_final[idx_path]["type_path"]


            # ---------------------------------------------------------------------------------------
            # get info ("extracted") for this path
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted      = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted     = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted    = dict_extracted["xy_right_img"]

            arr_xyz_switch_3d_extracted   = dict_extracted["xyz_switch_3d"]
            info_xyz_switch_3d_edge_start = dict_extracted["xyz_switch_3d_edge_start"]
            info_xyz_switch_3d_edge_end   = dict_extracted["xyz_switch_3d_edge_end"]


            ### just for setting max y for visualization
            arr_xyz_cen_3d_temp           = dict_extracted["xyz_cen_3d"]
            arr_y_cen_3d_temp             = arr_xyz_cen_3d_temp[:, 1]
            max_y_cen_3d_extracted        = max(arr_y_cen_3d_temp)
                # max y for visualization


            # ---------------------------------------------------------------------------------------
            # get info ("polynomial-fitted") for this path
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end


            # ---------------------------------------------------------------------------------------
            # draw (filled) path-region
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ### get info
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ### 3d coordinate
                    x_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 0]
                    y_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 0]
                    y_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 0]
                    y_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ### img coordinate
                    x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))
                #end


                ### check if it is over max-y (for visualization)
                if y_cen_3d_this >= max_y_cen_3d_extracted:
                    continue
                # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))


                ###------------------------------------------------------------------------------------
                ### check if it is in a valid image region
                ###------------------------------------------------------------------------------------
                b_draw = False

                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img) and \
                   (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img) and \
                   (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):

                    b_draw = True
                #end

                ###
                if b_draw is False:
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is a switch edge
                ###------------------------------------------------------------------------------------
                b_switch_edge = False
                b_switch_edge_of_non_ego_path = False

                for idx_sw in range(len(info_xyz_switch_3d_edge_start)):
                    y_switch_edge_3d_this_start = info_xyz_switch_3d_edge_start[idx_sw][1]
                    y_switch_edge_3d_this_end   = info_xyz_switch_3d_edge_end[idx_sw][1]

                    if (y_switch_edge_3d_this_start <= y_cen_3d_this) and (y_cen_3d_this <= y_switch_edge_3d_this_end):
                        b_switch_edge = True
                    #end
                #end

                ###
                if (b_switch_edge is True) and (type_path_this is my_utils_RPG.TYPE_path.NON_EGO):
                    b_switch_edge_of_non_ego_path = True
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is in switch region
                ###------------------------------------------------------------------------------------
                b_switchregion = False

                for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                    y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]

                    dy = y_cen_3d_this - y_switch_3d_this

                    if (-12.0 <= dy) and (dy <= 5.0):
                        b_switchregion = True
                    #end
                #end


                ###
                if (b_switch_edge_of_non_ego_path is True) and (b_switchregion is False):
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### draw path region
                ###------------------------------------------------------------------------------------
                type_region = 0     # 0(normal region), 1(switch region)

                if b_switchregion is True:
                    type_region = 1
                #end

                ###
                for x_img in range(x_left_img_int, x_right_img_int + 1):
                    b_old, g_old, r_old = img_bg_in[y_cen_img_int, x_img, :]
                    b_new, g_new, r_new = my_utils_vis.adjust_rgb_for_region(b_old, g_old, r_old, type_region)
                    img_vis_res[y_cen_img_int, x_img, :] = [b_new, g_new, r_new]
                #end
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### step 1: draw paths
        ###------------------------------------------------------------------------------------------------
        for idx_draw in range(len(list_paths_final)):
            idx_path = list_idx_path_for_drawing[idx_draw]
            type_path_this = list_paths_final[idx_path]["type_path"]

            # ---------------------------------------------------------------------------------------
            # get info ("extracted") for this path
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted      = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted     = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted    = dict_extracted["xy_right_img"]

            arr_xyz_switch_3d_extracted   = dict_extracted["xyz_switch_3d"]

            info_xyz_switch_3d_edge_start = dict_extracted["xyz_switch_3d_edge_start"]
            info_xyz_switch_3d_edge_end   = dict_extracted["xyz_switch_3d_edge_end"]


            ### just for setting max y for visualization
            arr_xyz_cen_3d_temp     = dict_extracted["xyz_cen_3d"]
            arr_y_cen_3d_temp       = arr_xyz_cen_3d_temp[:, 1]
            max_y_cen_3d_extracted  = max(arr_y_cen_3d_temp)
                # max y for visualization


            # ---------------------------------------------------------------------------------------
            # get info ("polynomial-fitted") for this path
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end


            # ---------------------------------------------------------------------------------------
            # draw paths
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ### get info
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ###
                    x_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 0]
                    y_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 0]
                    y_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 0]
                    y_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ###
                    x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))
                #end


                ### check if it is over max-y (for visualization)
                if y_cen_3d_this >= max_y_cen_3d_extracted:
                    continue
                # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))



                ###------------------------------------------------------------------------------
                ### DRAW
                ###------------------------------------------------------------------------------
                val_b_cen = 0
                val_g_cen = 128
                val_r_cen = 0

                val_b_left = 20
                val_g_left = 100
                val_r_left = 250

                val_b_right = 250
                val_g_right = 200
                val_r_right = 0


                ###
                b_draw = False

                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img) and \
                   (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img) and \
                   (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):

                    b_draw = True
                #end

                if b_draw is False:
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is a switch edge
                ###------------------------------------------------------------------------------------
                b_switch_edge = False
                b_switch_edge_of_non_ego_path = False

                for idx_sw in range(len(info_xyz_switch_3d_edge_start)):
                    y_switch_edge_3d_this_start = info_xyz_switch_3d_edge_start[idx_sw][1]
                    y_switch_edge_3d_this_end   = info_xyz_switch_3d_edge_end[idx_sw][1]

                    if (y_switch_edge_3d_this_start <= y_cen_3d_this) and (y_cen_3d_this <= y_switch_edge_3d_this_end):
                        b_switch_edge = True
                    #end
                #end

                if (b_switch_edge is True) and (type_path_this is my_utils_RPG.TYPE_path.NON_EGO):
                    b_switch_edge_of_non_ego_path = True
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is in switch region
                ###------------------------------------------------------------------------------------
                b_switchregion = False

                for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                    y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]

                    dy = y_cen_3d_this - y_switch_3d_this

                    if (-12.0 <= dy) and (dy <= 5.0):
                        b_switchregion = True
                    #end
                #end


                ###
                if (b_switch_edge_of_non_ego_path is True) and (b_switchregion is False):
                    continue
                #end


                ###
                val_radius = 1

                if type_path_this is my_utils_RPG.TYPE_path.EGO:
                    val_radius = 3
                #end

                #cv2.circle(img_vis_res, center=(x_cen_img_int, y_cen_img_int), radius=2, color=(val_b_cen, val_g_cen, val_r_cen), thickness=-1)
                cv2.circle(img_vis_res, center=(x_left_img_int, y_left_img_int), radius=val_radius, color=(val_b_left, val_g_left, val_r_left), thickness=-1)
                cv2.circle(img_vis_res, center=(x_right_img_int, y_right_img_int), radius=val_radius, color=(val_b_right, val_g_right, val_r_right), thickness=-1)
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('final_path_on_ori', img_vis_res)
        cv2.waitKey(1)

        ### save (temp)
        if 0:
            #fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_on_ori/" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.jpg'
            fname_output_temp = "/home/yu1/Desktop/dir_res_path_full/nyc_temp/" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.png'
            cv2.imwrite(fname_output_temp, img_vis_res)
            self.m_temp_idx_res_img_on_ori += 1
        #end

    #end



    ###=========================================================================================================
    ### visualize final path on ipm
    ###=========================================================================================================
    def show_final_path_on_ipm(self, list_paths_final, img_bg, res_for_show=1):
        #---------------------------------------------------------------------------------------------
        # dict_path_final = {"extracted": dict_path_this,
        #                    "polynomial": dict_path_poly_this}
        #
        #    "extracted": dict_path_this = {"xy_cen_img": [],
        #                                   "xy_left_img": [],
        #                                   "xy_right_img": [],
        #                                   "xyz_cen_3d": [],
        #                                   "xyz_left_3d": [],
        #                                   "xyz_right_3d": []}
        #
        #    "polynomial": dict_path_poly_this = {"xyz_cen_3d": arr_xyz_sample_ori,
        #                                         "xyz_left_3d": [],
        #                                         "xyz_right_3d": []}
        #---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        #---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        #---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)

        #img_raw_rsz_uint8 = self.m_obj_utils_rpg.get_img_raw_rsz_uint8()
        #assert (img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        #img_vis_ipm_rgb   = self.m_obj_utils_3D.create_img_IPM(img_raw_rsz_uint8)
        img_vis_ipm_rgb = self.m_obj_utils_3D.create_img_IPM(img_bg)
        img_vis_ipm_gray1 = cv2.cvtColor(img_vis_ipm_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ipm_gray3 = np.zeros_like(img_vis_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
        img_vis_ipm_gray3[:, :, 0] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 1] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 2] = img_vis_ipm_gray1

        ###
        h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        # h_img_temp = img_raw_rsz_uint8.shape[0]
        # w_img_temp = img_raw_rsz_uint8.shape[1]
        h_img_temp = img_bg.shape[0]
        w_img_temp = img_bg.shape[1]

        x_pnt_bev, y_pnt_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev(np.array([[w_img_temp/2], [h_img_temp - 1], [1.0]]))

        x_bev_visualize_int = int(round(x_pnt_bev))
        y_bev_visualize_int = int(round(y_pnt_bev))


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        for idx_path in range(len(list_paths_final)):
            # val_r = int(self.m_rgb_cluster[idx_path, 0])
            # val_g = int(self.m_rgb_cluster[idx_path, 1])
            # val_b = int(self.m_rgb_cluster[idx_path, 2])

            #------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            arr_xyz_cen_temp = dict_extracted["xyz_cen_3d"]
            arr_y_cen_temp   = arr_xyz_cen_temp[:, 1]

            max_y_cen_extracted = max(arr_y_cen_temp)


            #------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            arr_xyz_cen = None
            arr_xyz_left = None
            arr_xyz_right = None

            if res_for_show == 0:
                arr_xyz_cen = dict_extracted["xyz_cen_3d"]
                arr_xyz_left = dict_extracted["xyz_left_3d"]
                arr_xyz_right = dict_extracted["xyz_right_3d"]
            else:
                arr_xyz_cen = dict_polynomial["xyz_cen_3d"]
                arr_xyz_left = dict_polynomial["xyz_left_3d"]
                arr_xyz_right = dict_polynomial["xyz_right_3d"]
            #end


            for idx_pnt in range(arr_xyz_cen.shape[0]):
                ###
                x_cen_3d = arr_xyz_cen[idx_pnt, 0]
                y_cen_3d = arr_xyz_cen[idx_pnt, 1]

                x_left_3d = arr_xyz_left[idx_pnt, 0]
                y_left_3d = arr_xyz_left[idx_pnt, 1]

                x_right_3d = arr_xyz_right[idx_pnt, 0]
                y_right_3d = arr_xyz_right[idx_pnt, 1]

                if y_cen_3d >= max_y_cen_extracted:
                    continue
                #end


                ###
                x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_cen_3d], [y_cen_3d], [1.0]]))
                x_left_bev, y_left_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_left_3d], [y_left_3d], [1.0]]))
                x_right_bev, y_right_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_right_3d], [y_right_3d], [1.0]]))


                ###
                x_cen_bev_int = int(round(x_cen_bev))
                y_cen_bev_int = int(round(y_cen_bev))

                x_left_bev_int = int(round(x_left_bev))
                y_left_bev_int = int(round(y_left_bev))

                x_right_bev_int = int(round(x_right_bev))
                y_right_bev_int = int(round(y_right_bev))


                ###
                if y_cen_bev_int >= y_bev_visualize_int:
                    continue
                #end


                ### draw pnt (cen)
                val_b = 0
                val_g = 128
                val_r = 0

                if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_cen_bev_int, y_cen_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

                ### draw pnt (left)
                val_b = 20
                val_g = 100
                val_r = 250

                if (0 <= x_left_bev_int) and (x_left_bev_int < w_img_bev) and (0 <= y_left_bev_int) and (y_left_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_left_bev_int, y_left_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

                ### draw pnt (right)
                val_b = 250
                val_g = 250
                val_r = 0

                if (0 <= x_right_bev_int) and (x_right_bev_int < w_img_bev) and (0 <= y_right_bev_int) and (y_right_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_right_bev_int, y_right_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('final_path_on_ipm', img_vis_ipm_gray3)
        cv2.waitKey(1)


        ### save (temp)
        if 0:
            #fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_on_ipm/" + "final_path_ipm_" + str(self.m_temp_idx_res_img_on_ipm) + '.jpg'
            fname_output_temp = "/home/yu1/Desktop/dir_res_path/NYC/final_path_on_ipm/" + "final_path_ipm_" + str(self.m_temp_idx_res_img_on_ori) + '.png'
            cv2.imwrite(fname_output_temp, img_vis_ipm_gray3)
            self.m_temp_idx_res_img_on_ipm += 1
        #end
    #end


    ###=========================================================================================================
    ###
    ###=========================================================================================================
    def show_imgs_res_interim(self, dir_output, fname_img_in, img_raw_in, img_res_seg, img_res_centerness_combined, img_res_triplet_localmax, b_save=False):
        """
        show result images and save them as files (if wanted)

        :param dir_output:
        :param fname_img_in:
        :param img_raw_in:
        :param img_res_seg:
        :param img_res_centerness_combined:
        :param img_res_triplet_localmax:
        :param b_save:
        """


        ###------------------------------------------------------------------------------------------------
        ### set fname (for saving)
        ###------------------------------------------------------------------------------------------------

        ###
        # fname_out_img_raw_in                  = dir_output + 'in_' + fname_img_in
        # fname_out_img_res_seg                 = dir_output + 'seg_' + fname_img_in
        # fname_out_img_res_centerness_combined = dir_output + 'centerness_' + fname_img_in
        # fname_out_img_res_vis_temp2           = dir_output + 'triplet_' + fname_img_in
        fname_out_img_raw_in                  = dir_output + '/in/'         + 'in_'         + fname_img_in
        fname_out_img_res_seg                 = dir_output + '/seg/'        + 'seg_'        + fname_img_in
        fname_out_img_res_centerness_combined = dir_output + '/centerness/' + 'centerness_' + fname_img_in
        fname_out_img_res_vis_temp2           = dir_output + '/triplet/'    + 'triplet_'    + fname_img_in


        ###------------------------------------------------------------------------------------------------
        ### show and save
        ###------------------------------------------------------------------------------------------------

        ### show
        cv2.imshow('img_raw_in', img_raw_in)
        cv2.imshow('img_res_seg', img_res_seg)
        cv2.imshow('img_res_centerness', img_res_centerness_combined)  # combination of direct and LR
        cv2.imshow('img_res_triplet_localmax', img_res_triplet_localmax)  # center, and corresponding left, right


        ### save
        if b_save is True:
            cv2.imwrite(fname_out_img_raw_in, img_raw_in)
            cv2.imwrite(fname_out_img_res_seg, img_res_seg)
            cv2.imwrite(fname_out_img_res_centerness_combined, img_res_centerness_combined)
            cv2.imwrite(fname_out_img_res_vis_temp2, img_res_triplet_localmax)
        #end

        cv2.waitKey(0)

    #END



#end


########################################################################################################################
