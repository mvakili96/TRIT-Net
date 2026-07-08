from typing import Any, Dict, Optional


DEFAULT_WANDB_PROJECT = "trit-net"


def get_wandb_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the optional W&B config block from the training config."""
    training_cfg = cfg.get("training", {})
    return dict(training_cfg.get("wandb", {}) or {})


def get_wandb_log_interval(cfg: Dict[str, Any]) -> int:
    wandb_cfg = get_wandb_config(cfg)
    default_interval = cfg.get("training", {}).get("print_interval", 100)
    return int(wandb_cfg.get("log_interval", default_interval))


def initialize_wandb(
    cfg: Dict[str, Any],
    logdir: Optional[str] = None,
    config_path: Optional[str] = None,
    logger: Optional[Any] = None,
):
    """Initialize a W&B run when enabled, otherwise return None."""
    wandb_cfg = get_wandb_config(cfg)
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled, but the 'wandb' package is not installed. "
            "Install it with 'pip install wandb' or set training.wandb.enabled=false."
        ) from exc

    init_kwargs = {
        "project": wandb_cfg.get("project", DEFAULT_WANDB_PROJECT),
        "config": cfg,
    }

    if logdir is not None:
        init_kwargs["dir"] = logdir
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("run_name"):
        init_kwargs["name"] = wandb_cfg["run_name"]
    if wandb_cfg.get("mode"):
        init_kwargs["mode"] = wandb_cfg["mode"]

    run = wandb.init(**init_kwargs)
    run.config.update(
        {
            "config_path": config_path,
            "logdir": logdir,
        },
        allow_val_change=True,
    )

    if logger is not None:
        logger.info(
            "W&B logging enabled: project=%s, run=%s",
            init_kwargs["project"],
            run.name,
        )

    return run
