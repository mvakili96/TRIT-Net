import torch


DEFAULT_BEST_LOSS = 1000000000.0


def get_checkpoint_interval(cfg):
    return cfg["training"].get("checkpoint_interval", cfg["training"]["train_iters"])


def build_checkpoint_state(epoch, model_state, best_loss=DEFAULT_BEST_LOSS):
    return {
        "epoch": epoch,
        "model_state": model_state,
        "best_loss": best_loss,
    }


def get_checkpoint_path(iteration):
    return "Mybest_" + str(iteration) + ".pkl"


def save_checkpoint(model, iteration, best_loss=DEFAULT_BEST_LOSS, logger=None):
    state = build_checkpoint_state(
        epoch=iteration,
        model_state=model.state_dict(),
        best_loss=best_loss,
    )
    save_path = get_checkpoint_path(iteration)
    torch.save(state, save_path)
    if logger is not None:
        logger.info("Saved checkpoint {}".format(save_path))
    return save_path
