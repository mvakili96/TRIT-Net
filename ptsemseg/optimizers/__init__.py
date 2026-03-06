import logging
from typing import Any, Dict, Type

from torch.optim import SGD, Adam

logger = logging.getLogger("ptsemseg")

# Mapping from config key to the corresponding torch optimizer class.
key2opt: Dict[str, Type] = {
    "sgd": SGD,
    "adam": Adam,
}


def get_optimizer(cfg: Dict[str, Any]) -> Type:
    """Return an optimizer class based on the provided config.

    Contract (preserve existing behavior):
    - If ``cfg['training']['optimizer']`` is ``None``, return :class:`SGD` and
      log the choice.
    - Otherwise, read ``cfg['training']['optimizer']['name']`` and return the
      corresponding optimizer class from ``key2opt``. If the name is unknown,
      raise ``NotImplementedError``.

    This function is intentionally lightweight and does not construct the
    optimizer instance (the caller passes model parameters and hyper-params).
    """

    if cfg["training"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    opt_name = cfg["training"]["optimizer"]["name"]
    if opt_name not in key2opt:
        raise NotImplementedError(f"Optimizer {opt_name} not implemented")

    logger.info(f"Using {opt_name} optimizer")
    return key2opt[opt_name]
