from typing import Type

from ptsemseg.loader.triplet_loader import Triplet_Loader


def get_loader() -> Type[Triplet_Loader]:
    """Loader factory used by the trainer.

    Returns the Dataset class (not an instance). The trainer calls it like:
        data_loader = get_loader()
        t_loader_head = data_loader(configs=cfg, type_trainval='train', network_input_size=...)

    """

    return Triplet_Loader

