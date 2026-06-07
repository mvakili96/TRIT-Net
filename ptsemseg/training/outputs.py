from dataclasses import dataclass
from typing import Optional, Tuple

from ptsemseg.models.registry import AUX_OUTPUT_MODELS


@dataclass
class ModelOutputs:
    segmentation: object
    centerline: object
    afm: Optional[object] = None
    aux_outputs: Tuple[object, ...] = ()


def unpack_model_outputs(raw_outputs, arch: str, n_classes_segmentation: int) -> ModelOutputs:
    if arch not in AUX_OUTPUT_MODELS:
        if n_classes_segmentation == 4:
            outputs_seg, outputs_centerline, outputs_afm = raw_outputs
            return ModelOutputs(
                segmentation=outputs_seg,
                centerline=outputs_centerline,
                afm=outputs_afm,
            )

        outputs_seg, outputs_centerline = raw_outputs
        return ModelOutputs(
            segmentation=outputs_seg,
            centerline=outputs_centerline,
        )

    if n_classes_segmentation == 4:
        outputs_seg, outputs_centerline, outputs_afm, aux1, aux2, aux3, aux4 = raw_outputs
        return ModelOutputs(
            segmentation=outputs_seg,
            centerline=outputs_centerline,
            afm=outputs_afm,
            aux_outputs=(aux1, aux2, aux3, aux4),
        )

    outputs_seg, outputs_centerline, aux1, aux2, aux3, aux4 = raw_outputs
    return ModelOutputs(
        segmentation=outputs_seg,
        centerline=outputs_centerline,
        aux_outputs=(aux1, aux2, aux3, aux4),
    )


def forward_model(model, imgs_raw_fl_n, arch: str, n_classes_segmentation: int) -> ModelOutputs:
    raw_outputs = model(imgs_raw_fl_n)
    return unpack_model_outputs(raw_outputs, arch, n_classes_segmentation)
