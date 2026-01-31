import os
import torch
import numpy as np
import collections
import cv2
import math
import copy
from scipy.signal import find_peaks
import tifffile as tiff
from torchvision.transforms import ToTensor


def read_img_raw_jpg_from_file(full_fname_img_raw_jpg, size_img_rsz, arch, rgb_mean, rgb_std):

    img_raw = cv2.imread(full_fname_img_raw_jpg)
    
    if img_raw is None or img_raw.size == 0:
        fsz = os.path.getsize(full_fname_img_raw_jpg)
        raise RuntimeError(
            f"[read_img_raw_jpg_from_file] Failed to read image: {full_fname_img_raw_jpg} "
            f"(cv2.imread returned None/empty, size_on_disk={fsz} bytes)"
        )

    img_raw_rsz_uint8      = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
    img_raw_rsz_fl_n_final = convert_img_ori_to_img_data(img_raw_rsz_uint8, arch, rgb_mean=rgb_mean, rgb_std=rgb_std)

    return img_raw_rsz_uint8, img_raw_rsz_fl_n_final


def convert_img_ori_to_img_data(img_ori_uint8,
                                arch,
                                rgb_mean=None,
                                rgb_std=None):

    if arch == "rpnet_c" or arch == "bisenet_v2" or arch == "segformer" or arch == "seghardnet":
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0
        img_ori_fl_n = img_ori_fl - rgb_mean
        img_ori_fl_n = img_ori_fl_n / rgb_std
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        img_data_fl_n_final = img_ori_fl_n.astype(np.float32)
    elif arch == "dlinknet_34":
        img_data_fl_n_final = np.array(img_ori_uint8, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    elif arch == "erfnet":
        img_data_fl_n_final = ToTensor()(img_ori_uint8)
        img_data_fl_n_final = img_data_fl_n_final.numpy()
    else:
        raise ("No model found for converting image to data")

    return img_data_fl_n_final


def read_label_seg_png_from_file(full_fname_label_seg_png, size_out):
    img_raw = cv2.imread(full_fname_label_seg_png, cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise FileNotFoundError(f"Could not read {full_fname_label_seg_png}")

    h, w = img_raw.shape

    if w != size_out['w'] or h != size_out['h']:
        img_raw = cv2.resize(
            img_raw,
            (size_out['w'], size_out['h']), 
            interpolation=cv2.INTER_NEAREST
        )

    return img_raw


def read_triplet_image_from_file(full_fname_triplet_image_png,size_out):
    img_raw = cv2.imread(full_fname_triplet_image_png,cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise FileNotFoundError(f"Could not read {full_fname_triplet_image_png}")

    h, w = img_raw.shape

    if w != size_out['w'] or h != size_out['h']:
        img_raw = cv2.resize(
            img_raw,
            (size_out['w'], size_out['h']), 
            interpolation=cv2.INTER_NEAREST
        )

    img_reshaped = np.array([img_raw])

    return img_reshaped


def read_fnames_trainval(dir_this, idx_split):
    list_fname_ = os.listdir(dir_this)
    list_fname_ = [
        f for f in list_fname_
        if f != "@eaDir"                                  
        and not os.path.isdir(os.path.join(dir_this, f))  
    ]
    list_fname  = sorted(list_fname_)
    list_fname_train = list_fname[0:idx_split] 
    list_fname_val   = list_fname[idx_split:]

    dict_fnames = collections.defaultdict(list)
    dict_fnames["train"] = list_fname_train
    dict_fnames["val"]   = list_fname_val

    return dict_fnames


def decode_segmap(labelmap, plot=False):
    n_classes = 19

    rgb_class00 = [128,  64, 128]   # 00: road
    rgb_class01 = [244,  35, 232]   # 01: sidewalk
    rgb_class02 = [ 70,  70,  70]   # 02: construction
    rgb_class03 = [192,   0, 128]   # 03: tram-track
    rgb_class04 = [190, 153, 153]   # 04: fence
    rgb_class05 = [153, 153, 153]   # 05: pole
    rgb_class06 = [250, 170,  30]   # 06: traffic-light
    rgb_class07 = [220, 220,   0]   # 07: traffic-sign
    rgb_class08 = [107, 142,  35]   # 08: vegetation
    rgb_class09 = [152, 251, 152]   # 09: terrain
    rgb_class10 = [ 70, 130, 180]   # 10: sky
    rgb_class11 = [220,  20,  60]   # 11: human
    rgb_class12 = [230, 150, 140]   # 12: rail-track
    rgb_class13 = [  0,   0, 142]   # 13: car
    rgb_class14 = [  0,   0,  70]   # 14: truck
    rgb_class15 = [ 90,  40,  40]   # 15: trackbed
    rgb_class16 = [  0,  80, 100]   # 16: on-rails
    rgb_class17 = [  0, 254, 254]   # 17: rail-raised
    rgb_class18 = [  0,  68,  63]   # 18: rail-embedded

    rgb_labels = np.array(
        [
            rgb_class00,
            rgb_class01,
            rgb_class02,
            rgb_class03,
            rgb_class04,
            rgb_class05,
            rgb_class06,
            rgb_class07,
            rgb_class08,
            rgb_class09,
            rgb_class10,
            rgb_class11,
            rgb_class12,
            rgb_class13,
            rgb_class14,
            rgb_class15,
            rgb_class16,
            rgb_class17,
            rgb_class18,
        ]
    )

    r = np.ones_like(labelmap )*250         
    g = np.ones_like(labelmap )*250
    b = np.ones_like(labelmap )*250

    for l in range(0, n_classes):
        idx_set = (labelmap == l)           

        r[idx_set] = rgb_labels[l, 0]       
        g[idx_set] = rgb_labels[l, 1]       
        b[idx_set] = rgb_labels[l, 2]      

    img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
    img_label_rgb[:, :, 0] = r / 255.0
    img_label_rgb[:, :, 1] = g / 255.0
    img_label_rgb[:, :, 2] = b / 255.0

    return img_label_rgb


def decode_output_centerline(res_in):
    res_sigmoid = torch.clamp(torch.sigmoid(res_in), min=1e-4, max=1 - 1e-4)

    res_a = res_sigmoid[0].permute(1, 2, 0)    
    res_b = res_a[:, :, 0]                     
    res_c = res_b * 255.0
    res_d = torch.clamp(res_c, min=0.0, max=255.0)
    res_e = res_d.detach().cpu().numpy()

    res_out = res_b.detach().cpu().numpy()
    img_res_out = res_e.astype(np.uint8)

    return res_out, img_res_out


def decode_output_leftright(res_in):
    res_relu = torch.relu(res_in)
    res0     = res_relu[0]                                          

    res_left_a      = res0[0, :, :]                                 
    res_left_b      = torch.clamp(res_left_a, min=0.0, max=255.0)
    res_left_c      = res_left_b.detach().cpu().numpy()
    img_res_left    = res_left_c.astype(np.uint8)

    res_right_a     = res0[1, :, :]                                 
    res_right_b     = torch.clamp(res_right_a, min=0.0, max=255.0)
    res_right_c     = res_right_b.detach().cpu().numpy()
    img_res_right   = res_right_c.astype(np.uint8)

    res_left = res_left_c
    res_right = res_right_c

    return res_left, res_right, img_res_left, img_res_right


def compute_centerness_from_leftright(res_left, res_right):
    res_delta = abs(res_left - res_right)
    res_sum   = abs(res_left) + abs(res_right)

    res_ratio = res_delta/res_sum          
    res_ratio[np.isnan(res_ratio)] = 1.0
    res_ratio[res_sum <= 1.0] = 1.0

    res_weight = 1.0 - res_ratio

    res_weight_b   = res_weight*255.0
    img_res_weight = res_weight_b.astype(np.uint8)

    return res_weight, img_res_weight