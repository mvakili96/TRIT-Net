import torch
import torch.nn as nn
from collections import OrderedDict


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def load_my_state_dict(model, state_dict):  
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue

        own_state[name].copy_(param)

    return model


def load_weights_to_model(model, fname_weights_to_be_loaded, arch):
    ckpt = torch.load(fname_weights_to_be_loaded, map_location="cpu")
    if "model_state" in ckpt:
        state_dict_weights0 = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict_weights0 = ckpt["state_dict"]
    else:
        raise KeyError(f"No 'model_state' or 'state_dict' in checkpoint. Found keys: {list(ckpt.keys())}")

    print('loaded weights-to-be-loaded form %s !' % fname_weights_to_be_loaded)

    state_dict_weights = OrderedDict()

    for key in state_dict_weights0:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict_weights[key] = state_dict_weights0[key]          
        else:
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




















