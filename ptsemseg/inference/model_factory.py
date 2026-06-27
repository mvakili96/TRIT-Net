"""Demo/eval model factory backed by shared compatibility wrappers."""

from __future__ import annotations

import copy

from ptsemseg.inference.model_adapter import DEMO_EVAL_LOCAL_ONLY_ARCH_NAME
from ptsemseg.models.registry import MODEL_BISENET_V2
from ptsemseg.models.registry import MODEL_DLINKNET_34
from ptsemseg.models.registry import MODEL_ERFNET
from ptsemseg.models.registry import MODEL_SEGHARDNET
from ptsemseg.models.registry import MODEL_SEGFORMER
from ptsemseg.inference.model_wrappers import DemoEvalBiSeNetV2
from ptsemseg.inference.model_wrappers import DemoEvalDinkNet34
from ptsemseg.inference.model_wrappers import DemoEvalERFNet
from ptsemseg.inference.model_wrappers import DemoEvalSegHarDNet
from ptsemseg.inference.model_wrappers import DemoEvalSegFormer
from ptsemseg.inference.model_wrappers import DemoEvalTPEnetA


_DEMO_EVAL_MODEL_WRAPPERS = {
    DEMO_EVAL_LOCAL_ONLY_ARCH_NAME: DemoEvalTPEnetA,
    MODEL_DLINKNET_34: DemoEvalDinkNet34,
    MODEL_ERFNET: DemoEvalERFNet,
    MODEL_BISENET_V2: DemoEvalBiSeNetV2,
    MODEL_SEGFORMER: DemoEvalSegFormer,
    MODEL_SEGHARDNET: DemoEvalSegHarDNet,
}


def get_demo_eval_model(model_dict, n_classes, n_channels_reg, version=None):
    """Construct one demo/eval model through shared compatibility wrappers."""
    name = model_dict["arch"]
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = get_demo_eval_model_class(name)
    return model(n_classes=n_classes, n_channels_reg=n_channels_reg, **param_dict)


def get_demo_eval_model_class(name):
    """Return the audited demo/eval wrapper class for one architecture name."""
    try:
        return _DEMO_EVAL_MODEL_WRAPPERS[name]
    except KeyError as exc:
        supported = ", ".join(_DEMO_EVAL_MODEL_WRAPPERS)
        raise KeyError(
            f"Demo/eval model {name} not available. Supported models: {supported}"
        ) from exc


def get_demo_eval_model_names():
    """Return architecture names supported by the demo/eval factory."""
    return tuple(_DEMO_EVAL_MODEL_WRAPPERS)
