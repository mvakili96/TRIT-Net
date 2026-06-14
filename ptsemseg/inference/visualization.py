"""Shared visualization helpers with verified demo/eval compatibility."""

from __future__ import annotations

import numpy as np

from ptsemseg.loader.visualization import (
    compute_centerness_from_leftright as compute_training_centerness_from_leftright,
)


def compute_demo_eval_centerness_from_leftright(
    res_left: np.ndarray,
    res_right: np.ndarray,
):
    """Reuse the cleaned training helper where behavior is identical."""
    return compute_training_centerness_from_leftright(res_left, res_right)
