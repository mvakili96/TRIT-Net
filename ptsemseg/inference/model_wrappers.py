"""Compatibility model wrappers for demo/eval integration."""

from __future__ import annotations

from ptsemseg.models.dlinknet import DinkNet34 as SharedDinkNet34
from ptsemseg.models.erfnet import ERFNet as SharedERFNet
from ptsemseg.models.SegEncode_HarDDecode import SegHarDNet as SharedSegHarDNet


DEMO_EVAL_FIXED_OUTPUT_SIZE = (540, 960)


class DemoEvalDinkNet34(SharedDinkNet34):
    """Shared DinkNet34 with the copied demo/eval regression contract."""

    def __init__(self, n_classes=19, n_channels_reg=1):
        super().__init__(
            n_classes_seg=n_classes,
            n_channels_reg=n_channels_reg,
        )


class DemoEvalERFNet(SharedERFNet):
    """Shared ERFNet with the copied demo/eval output contract."""

    def __init__(self, n_classes=19, n_channels_reg=1):
        super().__init__(
            n_classes_seg=n_classes,
            demo_eval_n_channels_reg=n_channels_reg,
            demo_eval_output_size=DEMO_EVAL_FIXED_OUTPUT_SIZE,
        )


class DemoEvalSegHarDNet(SharedSegHarDNet):
    """Shared SegHarDNet with the copied demo/eval output contract.

    Segmentation, centerline, and AFM outputs use the copied implementation's
    fixed ``(540, 960)`` size. Its optional left/right output remains
    input-sized. The shared training model keeps its existing behavior when
    ``n_channels_reg`` is omitted.
    """

    def __init__(self, n_classes=19, n_channels_reg=1):
        super().__init__(
            n_classes_seg=n_classes,
            output_size=DEMO_EVAL_FIXED_OUTPUT_SIZE,
            n_channels_reg=n_channels_reg,
        )
