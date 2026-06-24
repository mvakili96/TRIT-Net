# 2020/7/10
# Jungwon Kang

import copy


from helpers.models.TPEnet_a import TPEnet_a
from helpers.models.TPEnet_a import ERFNet
from helpers.models.TPEnet_a import Bisenet_v2
from helpers.models.TPEnet_a import SegFormer
from ptsemseg.inference.model_wrappers import DemoEvalDinkNet34
from ptsemseg.inference.model_wrappers import DemoEvalSegHarDNet

########################################################################################################################
###
########################################################################################################################
def get_model(model_dict, n_classes, n_channels_reg, version=None):
    """get model"""

    name        = model_dict["arch"]
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = _get_model_instance(name)
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
            "dlinknet_34": DemoEvalDinkNet34,
            "erfnet": ERFNet,
            "bisenet_v2": Bisenet_v2,
            "segformer": SegFormer,
            "seghardnet": DemoEvalSegHarDNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################
