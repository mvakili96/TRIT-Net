# 2020/7/10
# Jungwon Kang

import copy


from ptsemseg.inference.model_wrappers import DemoEvalBiSeNetV2
from ptsemseg.inference.model_wrappers import DemoEvalDinkNet34
from ptsemseg.inference.model_wrappers import DemoEvalERFNet
from ptsemseg.inference.model_wrappers import DemoEvalSegHarDNet
from ptsemseg.inference.model_wrappers import DemoEvalSegFormer
from ptsemseg.inference.model_wrappers import DemoEvalTPEnetA

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
            "TPEnet_a": DemoEvalTPEnetA,
            "dlinknet_34": DemoEvalDinkNet34,
            "erfnet": DemoEvalERFNet,
            "bisenet_v2": DemoEvalBiSeNetV2,
            "segformer": DemoEvalSegFormer,
            "seghardnet": DemoEvalSegHarDNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################
