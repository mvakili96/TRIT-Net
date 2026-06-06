import copy

from ptsemseg.models.registry import get_model_registry
from ptsemseg.models.registry import get_registered_model_names


def get_model(model_dict, n_classes_segmentation, version=None):
    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes_seg=n_classes_segmentation, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return get_model_registry()[name]
    except KeyError as exc:
        supported = ", ".join(get_registered_model_names())
        raise KeyError(f"Model {name} not available. Supported models: {supported}") from exc


