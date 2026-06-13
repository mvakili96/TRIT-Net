# 2020/7/12
# Jungwon Kang


import torch
from collections import OrderedDict
import os
import sys

from runtime_defaults import get_override_weight_path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ptsemseg.inference import load_demo_eval_checkpoint


########################################################################################################################
###
########################################################################################################################
class MyUtils_Net:

    m_fname_weights_to_be_loaded = None


    ###############################################################################################################
    ###
    ###############################################################################################################
    def __init__(self, dict_args, net_type, num_seg_classes, num_channel_reg):
        dict_args["file_weight"] = get_override_weight_path(net_type, num_seg_classes, num_channel_reg)

        self.m_fname_weights_to_be_loaded = dict_args["file_weight"]



    ### helper: ./net_weight/Association/CRV2ISPRS-runs/OURS-on-RailDB/Mybest_90000.pkl


    ###############################################################################################################
    ###
    ###############################################################################################################
    def load_weights_to_model(self, model):
        return load_demo_eval_checkpoint(
            model=model,
            fname_weights_to_be_loaded=self.m_fname_weights_to_be_loaded,
            arch="demo_eval_legacy",
        )


########################################################################################################################

