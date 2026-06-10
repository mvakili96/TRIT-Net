from torch.utils import data

from ptsemseg.loader import get_loader
from ptsemseg.loader.constants import TRAIN_SPLIT_NAME
from ptsemseg.loader.constants import VALID_NUM_SEG_CLASSES
from ptsemseg.loss import get_loss_function
from ptsemseg.models import get_model
from ptsemseg.models.registry import CUSTOM_WEIGHT_INIT_MODELS
from ptsemseg.optimizers import get_optimizer
from ptsemseg.schedulers import get_scheduler
from ptsemseg.training.weights import load_weights_to_model
from ptsemseg.training.weights import weights_init


def validate_num_segmentation_classes(n_classes_segmentation: int) -> None:
    if n_classes_segmentation not in VALID_NUM_SEG_CLASSES:
        raise ValueError(
            f"Invalid configuration: training.num_seg_classes={n_classes_segmentation}. "
            f"Expected one of {VALID_NUM_SEG_CLASSES}."
        )


def build_training_dataloader(cfg, network_input_size):
    data_loader = get_loader()
    train_dataset = data_loader(
        configs=cfg,
        split=TRAIN_SPLIT_NAME,
        network_input_size=network_input_size,
    )
    return data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )


def build_model(cfg, n_classes_segmentation, device):
    model = get_model(cfg["model"], n_classes_segmentation).to(device)

    if cfg["model"]["arch"] in CUSTOM_WEIGHT_INIT_MODELS:
        model.apply(weights_init)

    fname_weight_init = cfg["weight_init_t"][cfg["model"]["arch"]]
    if fname_weight_init != -1:
        load_weights_to_model(model, fname_weight_init, cfg["model"]["arch"])

    return model


def build_optimizer(cfg, model, logger):
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {
        k: v
        for k, v in cfg["training"]["optimizer"][cfg["training"]["optimizer"]["name"]].items()
    }

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    return optimizer


def build_scheduler(cfg, optimizer):
    scheduler_cfg = cfg["training"]["lr_schedule"]
    if scheduler_cfg is None:
        return get_scheduler(optimizer, None)

    resolved_scheduler_cfg = dict(scheduler_cfg)
    if "max_iter" not in resolved_scheduler_cfg:
        resolved_scheduler_cfg["max_iter"] = cfg["training"]["train_iters"]

    return get_scheduler(optimizer, resolved_scheduler_cfg)


def build_loss_function(cfg, logger):
    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))
    return loss_fn
