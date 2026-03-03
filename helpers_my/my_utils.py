import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any


def weights_init(m: nn.Module) -> None:
    """Initialize weights for convolutional layers.

    Uses Xavier normal initialization for `nn.Conv2d` layers.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def load_my_state_dict(model: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
    """Copy matching parameters from ``state_dict`` into ``model``.

    Returns the input model (mutated in-place) for convenience.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue

        own_state[name].copy_(param)

    return model


def load_weights_to_model(model: torch.nn.Module, fname_weights_to_be_loaded: str, arch: str) -> torch.nn.Module:
    """Load a checkpoint and align its state dict to ``model``.

    Supports checkpoints with keys ``model_state`` or ``state_dict`` and will
    attempt to skip/replace parameters when shapes mismatch while preserving
    the model's existing parameters for missing entries.
    """
    ckpt: Any = torch.load(fname_weights_to_be_loaded, map_location="cpu")
    if "model_state" in ckpt:
        state_dict_weights0 = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict_weights0 = ckpt["state_dict"]
    else:
        raise KeyError(f"No 'model_state' or 'state_dict' in checkpoint. Found keys: {list(ckpt.keys())}")

    print('loaded weights-to-be-loaded form %s !' % fname_weights_to_be_loaded)

    state_dict_weights: OrderedDict = OrderedDict()

    for key in state_dict_weights0:
        # preserve original mapping behavior
        state_dict_weights[key] = state_dict_weights0[key]

    state_dict_model0 = model.state_dict()
    state_dict_model = state_dict_model0

    for key in state_dict_weights:
        if key in state_dict_model:
            if state_dict_weights[key].shape != state_dict_model[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, state_dict_model[key].shape, state_dict_weights[key].shape))
                state_dict_weights[key] = state_dict_model[key]
        else:
            print('Drop parameter {}.'.format(key))

    for key in state_dict_model:
        if key not in state_dict_weights:
            print('No param {}.'.format(key))
            state_dict_weights[key] = state_dict_model[key]

    model.load_state_dict(state_dict_weights, strict=False)

    return model




















