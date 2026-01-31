import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate
from ptsemseg.loader.myhelpers import myhelper_railsem19_b

class RailSem19_SegTriplet_b_Loader(data.Dataset):
    def __init__(self,
                configs  = None,
                type_trainval="train",
                network_input_size = None):


        self.type_trainval          = type_trainval

        self.rgb_mean               = np.array([128.0, 128.0, 128.0])/255.0     
        self.rgb_std                = np.array([1.0, 1.0, 1.0])               
        self.n_classes              = configs["training"]["num_seg_classes"]
        self.root_dataset           = configs["data"]["root"]
        self.train_split            = configs["data"]["train_split"]
        self.arch_this              = configs["model"]["arch"]
        self.size_img_rsz           = network_input_size
        self.size_out               = network_input_size

        rgb_mean                    = configs["data"]["rgb_mean"]
        self.rgb_mean               = np.array(rgb_mean) / 255.0
        rgb_std                     = configs["data"]["rgb_std"]
        self.rgb_std                = np.array(rgb_std) / 255.0

        self.dir_img_raw_jpg   = self.root_dataset + 'jpgs/'
        self.dir_img_AFM       = self.root_dataset + 'AFM/'
        self.dir_triplet_image = self.root_dataset + 'C_image/'
        if self.n_classes == 3:
            self.dir_label_seg_png  = self.root_dataset + 'Seg3/'
        elif self.n_classes == 4:
            self.dir_label_seg_png  = self.root_dataset + 'Seg4/'

        self.fnames_img_raw_jpg   = myhelper_railsem19_b.read_fnames_trainval(self.dir_img_raw_jpg, self.train_split)
        self.fnames_label_seg_png = myhelper_railsem19_b.read_fnames_trainval(self.dir_label_seg_png, self.train_split)
        self.fnames_triplet_image = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_image, self.train_split)
        self.fnames_AFM           = myhelper_railsem19_b.read_fnames_trainval(self.dir_img_AFM, self.train_split)


    def __len__(self):
        return len(self.fnames_img_raw_jpg[self.type_trainval])


    def __getitem__(self, index):
        full_fname_img_raw_jpg    = self.dir_img_raw_jpg    + self.fnames_img_raw_jpg[self.type_trainval][index]
        full_fnames_label_seg_png = self.dir_label_seg_png  + self.fnames_label_seg_png[self.type_trainval][index]
        full_fnames_img_AFM       = self.dir_img_AFM        + self.fnames_AFM[self.type_trainval][index]
        full_fname_triplet_image  = self.dir_triplet_image  + self.fnames_triplet_image[self.type_trainval][index]

        img_raw_rsz_uint8, \
        img_raw_rsz_fl_n = myhelper_railsem19_b.read_img_raw_jpg_from_file(full_fname_img_raw_jpg,
                                                                           self.size_img_rsz,
                                                                           self.arch_this,
                                                                           self.rgb_mean,
                                                                           self.rgb_std)

        img_label_seg_rsz_uint8 = myhelper_railsem19_b.read_label_seg_png_from_file(full_fnames_label_seg_png,
                                                                                    self.size_out)

        labelmap_centerline = myhelper_railsem19_b.read_triplet_image_from_file(full_fname_triplet_image,self.size_img_rsz)

        AFM  = myhelper_railsem19_b.read_triplet_image_from_file(full_fnames_img_AFM, self.size_img_rsz)

        set_idx_invalid = (img_label_seg_rsz_uint8 > 18)
        img_label_seg_rsz_uint8[set_idx_invalid] = 250

        output_img_raw             = torch.from_numpy(img_raw_rsz_fl_n).float()
        output_img_label_seg       = torch.from_numpy(img_label_seg_rsz_uint8).long()
        output_labelmap_centerline = torch.from_numpy(labelmap_centerline).float()
        output_AFM                 = torch.from_numpy(AFM).float()

        output_final = {'img_raw_fl_n'                   : output_img_raw,                              # (3, h_rsz, w_rsz)
                        'gt_img_label_seg'               : output_img_label_seg,                        # (h_rsz, w_rsz)
                        'gt_labelmap_centerline'         : output_labelmap_centerline,
                        'gt_AFM'                         : output_AFM}                                  # (1, h_rsz, w_rsz)


        return output_final

   








