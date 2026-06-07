import os
import random
import shutil
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from ptsemseg.utils import get_logger


DEFAULT_CONFIG_PATH = "./configs/trit_net.yml"
DEFAULT_LOGDIR = "./runs"
DEFAULT_SEED = 1337


def get_default_config_path() -> str:
    return DEFAULT_CONFIG_PATH


def get_default_logdir() -> str:
    return DEFAULT_LOGDIR


def configure_debug_environment() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def load_config(config_path: str) -> Dict:
    with open(config_path) as fp:
        return yaml.safe_load(fp)


def set_random_seeds(cfg: Dict) -> None:
    seed = cfg.get("seed", DEFAULT_SEED)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_network_input_size(cfg: Dict) -> Dict:
    arch = cfg.get("model", {}).get("arch")
    if "network_image_sizes" in cfg and arch in cfg["network_image_sizes"]:
        return cfg["network_image_sizes"][arch]
    raise KeyError(
        f"architecture '{arch}' not found in cfg['network_image_sizes'] and no default is available"
    )


def create_writer_and_logger(logdir: str, config_path: str) -> Tuple[SummaryWriter, object]:
    os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))

    try:
        shutil.copy(config_path, logdir)
    except Exception:
        # copying config is best-effort; don't fail the run if it can't be copied
        pass

    logger = get_logger(logdir)
    return writer, logger
