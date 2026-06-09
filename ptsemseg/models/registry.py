MODEL_RPNET_C = "rpnet_c"
MODEL_DLINKNET_34 = "dlinknet_34"
MODEL_ERFNET = "erfnet"
MODEL_BISENET_V2 = "bisenet_v2"
MODEL_SEGFORMER = "segformer"
MODEL_SEGHARDNET = "seghardnet"

RGB_MEAN_STD_PREPROCESS_MODELS = frozenset(
    (MODEL_RPNET_C, MODEL_BISENET_V2, MODEL_SEGFORMER, MODEL_SEGHARDNET)
)
TO_TENSOR_PREPROCESS_MODELS = frozenset((MODEL_ERFNET,))
DLINKNET_PREPROCESS_MODELS = frozenset((MODEL_DLINKNET_34,))
CUSTOM_WEIGHT_INIT_MODELS = frozenset((MODEL_RPNET_C,))
AUX_OUTPUT_MODELS = frozenset((MODEL_BISENET_V2,))


def get_model_registry():
    from ptsemseg.models.rpnet_c import rpnet_c
    from ptsemseg.models.dlinknet import DinkNet34
    from ptsemseg.models.erfnet import ERFNet
    from ptsemseg.models.bisenet_v2 import Bisenet_v2
    from ptsemseg.models.segformer import SegFormer
    from ptsemseg.models.SegEncode_HarDDecode import SegHarDNet

    return {
        MODEL_RPNET_C: rpnet_c,
        MODEL_DLINKNET_34: DinkNet34,
        MODEL_ERFNET: ERFNet,
        MODEL_BISENET_V2: Bisenet_v2,
        MODEL_SEGFORMER: SegFormer,
        MODEL_SEGHARDNET: SegHarDNet,
    }


def get_registered_model_names():
    return tuple(get_model_registry().keys())
