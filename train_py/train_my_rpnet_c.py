import os
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import sys
import cv2

# Insert repo root (relative) into sys.path so imports work when running from train_py/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from torch.utils                import data
from tqdm                       import tqdm
from tensorboardX               import SummaryWriter

from ptsemseg.metrics           import averageMeter
from ptsemseg.augmentations     import get_composed_augmentations
from ptsemseg.models.registry   import AUX_OUTPUT_MODELS
from helpers_my                 import my_loss               # Center/LeftRight Loss
from ptsemseg.training          import build_loss_function
from ptsemseg.training          import build_model
from ptsemseg.training          import build_optimizer
from ptsemseg.training          import build_scheduler
from ptsemseg.training          import build_training_dataloader
from ptsemseg.training          import configure_debug_environment
from ptsemseg.training          import create_writer_and_logger
from ptsemseg.training          import get_checkpoint_interval
from ptsemseg.training          import get_default_config_path
from ptsemseg.training          import get_default_logdir
from ptsemseg.training          import get_device
from ptsemseg.training          import load_config
from ptsemseg.training          import resolve_network_input_size
from ptsemseg.training          import save_checkpoint
from ptsemseg.training          import set_random_seeds
from ptsemseg.training          import validate_num_segmentation_classes

def train(cfg: dict, writer: SummaryWriter, logger) -> None:
    """Main training loop.

    Args:
        cfg: configuration dictionary loaded from YAML (see configs/)
        writer: TensorBoard SummaryWriter instance
        logger: logger instance from `ptsemseg.utils.get_logger`

    This function preserves existing training behavior. It reads `cfg["model"]["arch"]`
    and resolves `network_input_size` from `cfg['network_image_sizes']`. Seeds are set
    from `cfg['seed']` when present.
    """

    set_random_seeds(cfg)

    device = get_device()

    network_input_size = resolve_network_input_size(cfg)

    n_classes_segmentation = cfg["training"]["num_seg_classes"]
    validate_num_segmentation_classes(n_classes_segmentation)

    t_loader_batch = build_training_dataloader(cfg, network_input_size)

    model = build_model(cfg, n_classes_segmentation, device)
    optimizer = build_optimizer(cfg, model, logger)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_loss_function(cfg, logger)

    start_iter = 0

    time_meter = averageMeter()

    best_loss_hmap = 1000000000.0
    checkpoint_interval = get_checkpoint_interval(cfg)
    i = start_iter
    flag = True

    loss_accum_all         = 0
    loss_accum_seg         = 0
    loss_accum_centerline  = 0
    loss_accum_AFM         = 0

    num_loss = 0

    while i <= cfg["training"]["train_iters"] and flag:
        for data_batch in t_loader_batch:
            torch.cuda.empty_cache()
            i += 1
            start_ts = time.time()

            imgs_raw_fl_n                        = data_batch['img_raw_fl_n']                     
            gt_imgs_label_seg                    = data_batch['gt_img_label_seg']                 
            gt_labelmap_centerline               = data_batch['gt_labelmap_centerline']           
            gt_AFM                               = data_batch['gt_AFM']                           

            imgs_raw_fl_n           = imgs_raw_fl_n.to(device)
            gt_imgs_label_seg       = gt_imgs_label_seg.to(device)
            gt_labelmap_centerline  = gt_labelmap_centerline.to(device)
            gt_AFM                  = gt_AFM.to(device)

            scheduler.step()
            model.train()
            optimizer.zero_grad()

            if cfg["model"]["arch"] not in AUX_OUTPUT_MODELS:
                if n_classes_segmentation == 4:
                    outputs_seg, outputs_centerline, outputs_AFM = model(imgs_raw_fl_n)
                elif n_classes_segmentation == 3:
                    outputs_seg, outputs_centerline = model(imgs_raw_fl_n)

            else:
                if n_classes_segmentation == 4:
                    outputs_seg, outputs_centerline, outputs_AFM, aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n)
                elif n_classes_segmentation == 3:
                    outputs_seg, outputs_centerline, aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n)

            if cfg["model"]["arch"] not in AUX_OUTPUT_MODELS:
                loss_seg = loss_fn(input=outputs_seg, target=gt_imgs_label_seg, dev = device)
            else:
                loss_seg_unique = loss_fn(input=outputs_seg, target=gt_imgs_label_seg, dev=device)
                loss_seg = loss_fn(input=aux1, target=gt_imgs_label_seg, dev=device) + \
                loss_fn(input=aux2, target=gt_imgs_label_seg, dev=device) + \
                loss_fn(input=aux3, target=gt_imgs_label_seg, dev=device) + \
                loss_fn(input=aux4, target=gt_imgs_label_seg, dev=device) + \
                loss_seg_unique
                loss_seg = loss_seg_unique

            loss_centerline = my_loss.L1_loss(x_est=outputs_centerline, x_gt=gt_labelmap_centerline, n_chann = 1, b_sigmoid=True)

            if n_classes_segmentation == 4:
                loss_AFM = my_loss.L1_loss(x_est=outputs_AFM, x_gt=gt_AFM, n_chann=1,b_sigmoid=True)
                loss_this = 1*loss_seg  + 1*loss_centerline  + 1*loss_AFM
                loss_accum_AFM        += loss_AFM.item()

            elif n_classes_segmentation == 3:
                loss_this = 1*loss_seg  + 1*loss_centerline

            loss_this.backward()
            optimizer.step()

            c_lr = scheduler.get_lr()

            time_meter.update(time.time() - start_ts)

            loss_accum_all        += loss_this.item()
            loss_accum_seg        += loss_seg.item()
            loss_accum_centerline += loss_centerline.item()
            num_loss += 1

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss (all): {:.7f}, Loss (seg): {:.7f}, Loss (centerline): {:.7f}, Loss (AFM): {:.7f}, Time/Image: {:.7f}  lr={:.7f}"

                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_accum_all        / num_loss,
                    loss_accum_seg        / num_loss,
                    loss_accum_centerline / num_loss,
                    loss_accum_AFM        / num_loss,
                    time_meter.avg / cfg["training"]["batch_size"],
                    c_lr[0],
                )

                # print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss_this.item(), i + 1)
                time_meter.reset()

            if (i + 1) % checkpoint_interval == 0 or (i + 1) == cfg["training"]["train_iters"]:
                save_checkpoint(model, i + 1, best_loss=best_loss_hmap, logger=logger)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    # defaults (preserve prior behavior)
    default_config = get_default_config_path()
    default_logdir = get_default_logdir()

    configure_debug_environment()

    parser = argparse.ArgumentParser(description="TRIT-Net trainer")
    parser.add_argument('-c', '--config', help='path to config yaml', default=default_config)
    parser.add_argument('-l', '--logdir', help='log directory', default=default_logdir)
    args = parser.parse_args()

    cfg = load_config(args.config)
    writer, logger = create_writer_and_logger(args.logdir, args.config)
    logger.info("Let's begin...")

    train(cfg, writer, logger)




