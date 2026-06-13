from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """Initialize weights for convolutional layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def load_my_state_dict(model: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
    """Copy matching parameters from ``state_dict`` into ``model``."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue

        own_state[name].copy_(param)

    return model


def align_and_load_state_dict(model: torch.nn.Module, state_dict_weights0: dict) -> torch.nn.Module:
    """Align a loaded state dict to ``model`` and load it with legacy-tolerant behavior."""
    state_dict_weights: OrderedDict = OrderedDict()

    for key in state_dict_weights0:
        state_dict_weights[key] = state_dict_weights0[key]

    state_dict_model = model.state_dict()

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


def load_weights_to_model(model: torch.nn.Module, fname_weights_to_be_loaded: str, arch: str) -> torch.nn.Module:
    """Load a checkpoint and align its state dict to ``model``."""
    ckpt: Any = torch.load(fname_weights_to_be_loaded, map_location="cpu")
    if "model_state" in ckpt:
        state_dict_weights0 = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict_weights0 = ckpt["state_dict"]
    else:
        raise KeyError(f"No 'model_state' or 'state_dict' in checkpoint. Found keys: {list(ckpt.keys())}")

    print('loaded weights-to-be-loaded form %s !' % fname_weights_to_be_loaded)

    return align_and_load_state_dict(model=model, state_dict_weights0=state_dict_weights0)
