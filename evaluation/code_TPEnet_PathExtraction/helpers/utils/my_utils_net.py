# 2020/7/12
# Jungwon Kang


import torch
from collections import OrderedDict

from runtime_defaults import get_override_weight_path


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


        ###================================================================================================
        ### 1. load weights-to-be-loaded from a file
        ###================================================================================================
        state_dict_weights0 = torch.load(self.m_fname_weights_to_be_loaded)["model_state"]
        print('loaded weights-to-be-loaded form %s !' % self.m_fname_weights_to_be_loaded)

        ###================================================================================================
        ### 2. selective copy (1): copy state_dict_weights0[] -> state_dict_weights[]
        ###================================================================================================
        state_dict_weights = OrderedDict()

        for key in state_dict_weights0:
            if key.startswith('module') and not key.startswith('module_list'):
                state_dict_weights[key] = state_dict_weights0[key]          # <edited by Jungwon>
            else:
                state_dict_weights[key] = state_dict_weights0[key]


        ###================================================================================================
        ### 3. selective copy (2): copy state_dict_model[] -> state_dict_weights[]
        ###================================================================================================
        state_dict_model = model.state_dict()

        for key in state_dict_weights:             
            if key in state_dict_model:     
                ###------------------------------------------------------------------------------
                ### if shape is not consistent, just skip.
                ###------------------------------------------------------------------------------
                if state_dict_weights[key].shape != state_dict_model[key].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        key, state_dict_model[key].shape, state_dict_weights[key].shape))
                    state_dict_weights[key] = state_dict_model[key]         # copy dummy into state_dict[]
            else:
                ###------------------------------------------------------------------------------
                ### if key in state dict[] does not exist in model_state_dict[], just ignore.
                ###------------------------------------------------------------------------------
                print('Drop parameter {}.'.format(key))


        for key in state_dict_model:
            if key not in state_dict_weights:
                ###------------------------------------------------------------------------------
                ### if key in state_dict_model{} does not exist in state_dict_weights{}, just ignore.
                ###------------------------------------------------------------------------------
                print('No param {}.'.format(key))
                state_dict_weights[key] = state_dict_model[key]         # copy dummy into state_dict_weights[]



        ###================================================================================================
        ### 4. fill in model with final weights_to_be_loaded
        ###================================================================================================
        model.load_state_dict(state_dict_weights, strict=False)


        return model


########################################################################################################################


