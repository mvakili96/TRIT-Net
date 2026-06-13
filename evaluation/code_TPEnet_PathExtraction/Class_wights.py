import os
import re
import cv2
import numpy as np
import tifffile as tiff

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List
from typing import Iterable


# format_fname_data_in = "C:/Users/mmoei/PycharmProjects/TPE_Net_Training/TPE_training/proj_seg_hardnet_a_20210821_1/proj_seg_hardnet_a/train_py/uint8/rs19_val_link_4class+ydhr"
# list_fnames = os.listdir(format_fname_data_in)
# list_fnames.sort(key=lambda f: int(re.sub('\D', '', f)))
#
# class_pixel_counts = np.zeros(4, dtype=np.float32)
# for my_idx,fname_img_in in enumerate(list_fnames):
#     # if my_idx <= 100:
#     #     pass
#     # else:
#     #     continue
#
#     print(my_idx)
#
#     mask_img = cv2.imread(format_fname_data_in + '/rs' + f"{my_idx:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
#     flat_image = mask_img.flatten()
#     unique_classes, class_counts = np.unique(flat_image, return_counts=True)
#
#     for cls, count in zip(unique_classes, class_counts):
#         class_pixel_counts[cls] += count
#
#
# class_frequencies = class_pixel_counts / np.sum(class_pixel_counts)
# print(class_pixel_counts)
# print(class_frequencies)


# format_fname_data = "C:/Users/mmoei/PycharmProjects/TPE_Net_Training/TPE_training/proj_seg_hardnet_a_20210821_1/proj_seg_hardnet_a/train_py/LR_regu_railsem+ydhr/Start_heatmap/"
# list_fnames = os.listdir(format_fname_data)
# list_fnames.sort(key=lambda f: int(re.sub('\D', '', f)))
#
# for my_idx,fname_img_in in enumerate(list_fnames):
#     if my_idx == 4:
#         pass
#     else:
#         continue
#
#     img_raw = tiff.imread(format_fname_data+fname_img_in)
#     pos_inds = img_raw > 0.9
#
#     coordinates = np.argwhere(pos_inds)
#
#     print(coordinates)














