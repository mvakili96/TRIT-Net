import os
from typing import List


def read_fnames_train(dir_this: str, max_train_index: int) -> List[str]:
    """List files in `dir_this` and keep training files up to `max_train_index`."""
    list_fname_ = os.listdir(dir_this)
    list_fname_ = [
        f for f in list_fname_
        if f != "@eaDir" and not os.path.isdir(os.path.join(dir_this, f))
    ]
    list_fname = sorted(list_fname_)

    return list_fname[0:max_train_index]
