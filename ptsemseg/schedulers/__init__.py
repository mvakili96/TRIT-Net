import logging
import inspect
from typing import Any, Dict, Optional

from ptsemseg.schedulers.schedulers import WarmUpLR, ConstantLR, PolynomialLR

logger = logging.getLogger("ptsemseg")

# Mapping from config key to scheduler constructor/class.
key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,

}


def _filter_and_validate_kwargs(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Filter cfg to only keys accepted by ``cls.__init__`` and validate
    that required (positional without defaults) args are present.

    Returns a new dict with only acceptable kwargs. Raises ``KeyError`` if a
    required parameter is missing.
    """
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())

    # parameter names to accept as kwargs (exclude 'self' and 'optimizer')
    accept_names = {p.name for p in params if p.name not in ("self", "optimizer")}

    # required params are those without default (and not VAR_POSITIONAL/VAR_KEYWORD)
    required = [
        p.name
        for p in params
        if p.name not in ("self", "optimizer")
        and p.default is inspect._empty
        and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    missing = [name for name in required if name not in cfg]
    if missing:
        raise KeyError(f"Missing required scheduler params for {cls.__name__}: {missing}")

    return {k: v for k, v in cfg.items() if k in accept_names}


def get_scheduler(optimizer, scheduler_dict: Optional[Dict[str, Any]]):
    """Create and return a learning-rate scheduler based on `scheduler_dict`.

    Behavior-preserving notes:
    - If ``scheduler_dict`` is ``None``, returns ``ConstantLR(optimizer)``.
    - Otherwise reads ``scheduler_dict['name']`` to select the scheduler.
    - Supports an optional warmup block in the config using keys
      ``warmup_iters``, ``warmup_mode``, ``warmup_factor``; when present a
      ``WarmUpLR`` wrapping the base scheduler is returned.

    The factory filters extraneous configuration keys and validates required
    parameters for each scheduler class, raising a helpful error if the
    configuration is incomplete.
    """

    if scheduler_dict is None:
        logger.info("Using No LR Scheduling")
        return ConstantLR(optimizer)

    # work on a copy so we don't mutate the caller's dict
    sched_cfg = dict(scheduler_dict)
    s_type = sched_cfg.pop("name")

    logging.info("Using %s scheduler with %s params", s_type, sched_cfg)

    warmup_dict: Dict[str, Any] = {}
    if "warmup_iters" in sched_cfg:
        warmup_dict["warmup_iters"] = sched_cfg.get("warmup_iters", 100)
        warmup_dict["mode"] = sched_cfg.get("warmup_mode", "linear")
        warmup_dict["gamma"] = sched_cfg.get("warmup_factor", 0.2)

        logger.info(
            "Using Warmup with %s iters %s gamma and %s mode",
            warmup_dict["warmup_iters"],
            warmup_dict["gamma"],
            warmup_dict["mode"],
        )

        # remove warmup keys before passing remaining args to base scheduler
        sched_cfg.pop("warmup_iters", None)
        sched_cfg.pop("warmup_mode", None)
        sched_cfg.pop("warmup_factor", None)

        cls = key2scheduler[s_type]
        filtered_kwargs = _filter_and_validate_kwargs(cls, sched_cfg)
        base_scheduler = cls(optimizer, **filtered_kwargs)

        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    cls = key2scheduler[s_type]
    filtered_kwargs = _filter_and_validate_kwargs(cls, sched_cfg)
    return cls(optimizer, **filtered_kwargs)
