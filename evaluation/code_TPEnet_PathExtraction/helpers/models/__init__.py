# 2020/7/10
# Jungwon Kang

import copy


from helpers.models.TPEnet_a import TPEnet_a
from helpers.models.TPEnet_a import DinkNet34
from helpers.models.TPEnet_a import ERFNet
from helpers.models.TPEnet_a import Bisenet_v2
from helpers.models.TPEnet_a import SegFormer
from helpers.models.SegHarDNet import SegHarDNet

########################################################################################################################
###
########################################################################################################################
def get_model(model_dict, n_classes, n_channels_reg, version=None):
    """get model"""

    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes=n_classes, n_channels_reg = n_channels_reg, **param_dict)

    return model
#end


########################################################################################################################
###
########################################################################################################################
def _get_model_instance(name):
    """get model instance"""

    try:
        return {
            "TPEnet_a": TPEnet_a,
            "dlinknet_34": DinkNet34,
            "erfnet": ERFNet,
            "bisenet_v2": Bisenet_v2,
            "segformer": SegFormer,
            "seghardnet": SegHarDNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################
