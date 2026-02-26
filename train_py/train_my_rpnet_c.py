import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import sys
import cv2

sys.path.insert(0, "/home/m_vakili_am/Projects/TRIT-Net/")   # project root
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from torch.utils                import data
from tqdm                       import tqdm
from tensorboardX               import SummaryWriter

from ptsemseg.models            import get_model
from ptsemseg.loss              import get_loss_function     # Segmentation Loss
from ptsemseg.loader            import get_loader
from ptsemseg.utils             import get_logger
from ptsemseg.metrics           import averageMeter
from ptsemseg.augmentations     import get_composed_augmentations
from ptsemseg.schedulers        import get_scheduler
from ptsemseg.optimizers        import get_optimizer
from helpers_my                 import my_utils
from helpers_my                 import my_loss               # Center/LeftRight Loss

def train(cfg, writer, logger):

    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = cfg.get("model", {}).get("arch")
    if arch in cfg["network_image_sizes"]:
        network_input_size = cfg["network_image_sizes"][arch]
    else:
        raise KeyError(f"architecture '{arch}' not found in cfg['network_image_sizes'] and no default is available")

    data_loader   = get_loader()

    n_classes_segmentation = cfg["training"]["num_seg_classes"]

    t_loader_head = data_loader(configs=cfg, type_trainval="train", network_input_size=network_input_size)
    v_loader_head = data_loader(configs=cfg, type_trainval="val", network_input_size=network_input_size)

    t_loader_batch = data.DataLoader(
        t_loader_head,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        )

    v_loader_batch = data.DataLoader(
        v_loader_head,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"]
        )


    model = get_model(cfg["model"], n_classes_segmentation).to(device)

    if cfg["model"]["arch"] == "rpnet_c":
        model.apply(my_utils.weights_init)

    fname_weight_init = cfg["weight_init_t"][cfg["model"]["arch"]]

    if fname_weight_init != -1:
        my_utils.load_weights_to_model(model, fname_weight_init, cfg["model"]["arch"])

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"][cfg["training"]["optimizer"]["name"]].items()}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)

    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0

    time_meter = averageMeter()

    best_iou = -100.0
    best_loss_hmap = 1000000000.0
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

            if cfg["model"]["arch"] != "bisenet_v2":
                if n_classes_segmentation == 4:
                    outputs_seg, outputs_centerline, outputs_AFM = model(imgs_raw_fl_n)
                elif n_classes_segmentation == 3:
                    outputs_seg, outputs_centerline = model(imgs_raw_fl_n)

            else:
                if n_classes_segmentation == 4:
                    outputs_seg, outputs_centerline, outputs_AFM, aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n)
                elif n_classes_segmentation == 3:
                    outputs_seg, outputs_centerline, aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n)

            if cfg["model"]["arch"] != "bisenet_v2":
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

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                loss_accum_seg_validation = 0
                loss_accum_centerline_validation = 0
                loss_accum_AFM_validation = 0
                num_loss_validation = 0
                for data_batch_validation in v_loader_batch:
                    imgs_raw_fl_n_val                        = data_batch_validation['img_raw_fl_n']                     
                    gt_imgs_label_seg_val                    = data_batch_validation['gt_img_label_seg']               
                    gt_labelmap_centerline_val               = data_batch_validation['gt_labelmap_centerline']          
                    gt_AFM_val                               = data_batch_validation['gt_AFM']                           

                    imgs_raw_fl_n_val          = imgs_raw_fl_n_val.to(device)
                    gt_imgs_label_seg_val      = gt_imgs_label_seg_val.to(device)
                    gt_labelmap_centerline_val = gt_labelmap_centerline_val.to(device)
                    gt_AFM_val                 = gt_AFM_val.to(device)

                    if cfg["model"]["arch"] != "bisenet_v2":
                        if n_classes_segmentation == 4:
                            outs_seg, outs_centerline, outs_AFM =  model(imgs_raw_fl_n_val)
                        elif n_classes_segmentation == 3:
                            outs_seg, outs_centerline           =  model(imgs_raw_fl_n_val)
                    
                    else:
                        if n_classes_segmentation == 4:
                            outs_seg, outs_centerline, outs_AFM, aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n_val)
                        elif n_classes_segmentation == 3:
                            outs_seg, outs_centerline , aux1, aux2, aux3, aux4 = model(imgs_raw_fl_n_val)

                    loss_seg_val        = loss_fn(input=outs_seg, target=gt_imgs_label_seg_val, dev = device)
                    loss_centerline_val = my_loss.L1_loss(x_est=outs_centerline, x_gt=gt_labelmap_centerline_val, n_chann = 1, b_sigmoid=True)

                    if n_classes_segmentation == 4:
                        loss_AFM_val        = my_loss.L1_loss(x_est=outs_AFM, x_gt=gt_AFM_val, n_chann=1,b_sigmoid=True)
                        loss_accum_AFM_validation        += loss_AFM_val.item()


                    loss_accum_seg_validation        += loss_seg_val.item()
                    loss_accum_centerline_validation += loss_centerline_val.item()
                    
                    num_loss_validation += 1

                fmt_str = "(VALIDATION) Iter [{:d}/{:d}], Loss (seg): {:.7f}, Loss (centerline): {:.7f}, Loss (AFM): {:.7f}"

                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_accum_seg_validation / num_loss_validation,
                    loss_accum_centerline_validation / num_loss_validation,
                    loss_accum_AFM_validation / num_loss_validation,
                )

                print(print_str)


                state = {"epoch": i + 1,
                         "model_state": model.state_dict(),
                         "best_loss": best_loss_hmap}

                save_path = 'Mybest_' + str(i+1) + '.pkl'

                torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    fname_config = './configs/rpnet_c_railsem19_seg.yml'
    os.environ['CUDA_LAUNCH_BLOCKING']  = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    parser = argparse.ArgumentParser(description="config")
    args   = parser.parse_args()
    args.config = fname_config

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    run_id = random.randint(1, 100000)
    dir_log = './runs/rpnet_c_railsem19_seg/cur_20200725'

    writer = SummaryWriter(log_dir=dir_log)
    print("RUNDIR: {}".format(dir_log))

    shutil.copy(args.config, dir_log)

    logger = get_logger(dir_log)
    logger.info("Let's begin...")

    train(cfg, writer, logger)




