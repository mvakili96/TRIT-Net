import copy
import torchvision.models as models

from ptsemseg.models.rpnet_c import rpnet_c
from ptsemseg.models.comparison_models import DinkNet34
from ptsemseg.models.comparison_models import ERFNet
from ptsemseg.models.comparison_models import Bisenet_v2
from ptsemseg.models.segformer import SegFormer
from ptsemseg.models.SegEncode_HarDDecode import SegHarDNet


def get_model(model_dict, n_classes_segmentation, version=None):
    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes_seg=n_classes_segmentation, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "rpnet_c": rpnet_c,
            "dlinknet_34": DinkNet34,
            "erfnet": ERFNet,
            "bisenet_v2": Bisenet_v2,
            "segformer": SegFormer,
            "seghardnet": SegHarDNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))


