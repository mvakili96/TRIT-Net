"""Compatibility model wrappers for demo/eval integration."""

from __future__ import annotations

from ptsemseg.models.SegEncode_HarDDecode import SegHarDNet as SharedSegHarDNet


DEMO_EVAL_FIXED_OUTPUT_SIZE = (540, 960)


class DemoEvalSegHarDNet(SharedSegHarDNet):
    """Shared SegHarDNet with the copied demo/eval output-size contract.

    The verified default demo/eval path expects segmentation, centerline, and
    AFM outputs resized to fixed ``(540, 960)`` tensors. The shared training
    model keeps input-sized outputs by default, so this wrapper narrows the
    compatibility behavior to the demo/eval default mode.
    """

    def __init__(self, n_classes=19, n_channels_reg=1):
        if n_classes != 4 or n_channels_reg != 1:
            raise ValueError(
                "DemoEvalSegHarDNet is verified only for "
                "n_classes=4 and n_channels_reg=1."
            )
        super().__init__(
            n_classes_seg=n_classes,
            output_size=DEMO_EVAL_FIXED_OUTPUT_SIZE,
        )
        self.n_channels_reg = n_channels_reg
