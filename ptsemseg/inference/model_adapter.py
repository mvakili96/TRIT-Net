"""Compatibility helpers for gradual demo/eval integration.

This module is intentionally conservative:

- it reuses the cleaned training repo's registry constants and weight-loading
  helper where compatibility has been audited
- it does *not* force demo/eval to construct models through the cleaned
  training registry yet, because constructor signatures are not currently
  compatible for the copied demo/eval models
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import torch

from ptsemseg.models.registry import MODEL_BISENET_V2
from ptsemseg.models.registry import MODEL_DLINKNET_34
from ptsemseg.models.registry import MODEL_ERFNET
from ptsemseg.models.registry import MODEL_SEGHARDNET
from ptsemseg.models.registry import MODEL_SEGFORMER
from ptsemseg.models.registry import get_registered_model_names
from ptsemseg.training.weights import align_and_load_state_dict


DEMO_EVAL_ARCH_TPENET_A = 0
DEMO_EVAL_ARCH_DLINKNET_34 = 1
DEMO_EVAL_ARCH_ERFNET = 2
DEMO_EVAL_ARCH_BISENET_V2 = 3
DEMO_EVAL_ARCH_SEGFORMER = 4
DEMO_EVAL_ARCH_SEGHARDNET = 5

DEMO_EVAL_LOCAL_ONLY_ARCH_NAME = "TPEnet_a"

DEMO_EVAL_ARCHITECTURE_TO_NAME = {
    DEMO_EVAL_ARCH_TPENET_A: DEMO_EVAL_LOCAL_ONLY_ARCH_NAME,
    DEMO_EVAL_ARCH_DLINKNET_34: MODEL_DLINKNET_34,
    DEMO_EVAL_ARCH_ERFNET: MODEL_ERFNET,
    DEMO_EVAL_ARCH_BISENET_V2: MODEL_BISENET_V2,
    DEMO_EVAL_ARCH_SEGFORMER: MODEL_SEGFORMER,
    DEMO_EVAL_ARCH_SEGHARDNET: MODEL_SEGHARDNET,
}
DEMO_EVAL_ARCHITECTURE_NAME_TO_CODE = {
    arch_name: architecture_code
    for architecture_code, arch_name in DEMO_EVAL_ARCHITECTURE_TO_NAME.items()
}

# Audited constructor mismatch summary:
# - local copied repo expects ``n_classes`` and usually ``n_channels_reg``
# - cleaned training ``ptsemseg.models.get_model()`` expects
#   ``n_classes_segmentation`` and does not forward ``n_channels_reg`` generically
# - therefore construction stays local for now, even for shared architecture names
DEMO_EVAL_ARCHITECTURES_SAFE_FOR_TRAINING_REGISTRY_CONSTRUCTION = frozenset()


@dataclass(frozen=True)
class DemoEvalModelCompatibility:
    """Static compatibility status for one demo/eval architecture."""

    architecture_code: int
    demo_eval_name: str
    shared_registry_name: str | None
    local_constructor: str
    shared_constructor: str | None
    safe_for_training_registry_construction: bool
    risk_level: str
    recommendation: str


_DEMO_EVAL_MODEL_COMPATIBILITY = {
    DEMO_EVAL_ARCH_TPENET_A: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_TPENET_A,
        demo_eval_name=DEMO_EVAL_LOCAL_ONLY_ARCH_NAME,
        shared_registry_name=None,
        local_constructor="TPEnet_a(n_classes, n_channels_reg)",
        shared_constructor=None,
        safe_for_training_registry_construction=False,
        risk_level="high",
        recommendation=(
            "Keep local. This project-specific architecture has no proven "
            "training-registry equivalent."
        ),
    ),
    DEMO_EVAL_ARCH_DLINKNET_34: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_DLINKNET_34,
        demo_eval_name=MODEL_DLINKNET_34,
        shared_registry_name=MODEL_DLINKNET_34,
        local_constructor="DinkNet34(n_classes, n_channels_reg)",
        shared_constructor="DinkNet34(n_classes_seg)",
        safe_for_training_registry_construction=False,
        risk_level="high",
        recommendation=(
            "Keep local until checkpoint keys, output tuple order, and "
            "regression-head behavior are compared."
        ),
    ),
    DEMO_EVAL_ARCH_ERFNET: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_ERFNET,
        demo_eval_name=MODEL_ERFNET,
        shared_registry_name=MODEL_ERFNET,
        local_constructor="ERFNet(n_classes, n_channels_reg)",
        shared_constructor="ERFNet(n_classes_seg, n_channels_reg)",
        safe_for_training_registry_construction=False,
        risk_level="high",
        recommendation=(
            "Keep local until decoder channel behavior, checkpoint keys, and "
            "output tuple structure are verified."
        ),
    ),
    DEMO_EVAL_ARCH_BISENET_V2: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_BISENET_V2,
        demo_eval_name=MODEL_BISENET_V2,
        shared_registry_name=MODEL_BISENET_V2,
        local_constructor="Bisenet_v2(n_classes, n_channels_reg)",
        shared_constructor="Bisenet_v2(n_classes_seg)",
        safe_for_training_registry_construction=False,
        risk_level="high",
        recommendation=(
            "Keep local until auxiliary-output handling and checkpoint "
            "compatibility are proven."
        ),
    ),
    DEMO_EVAL_ARCH_SEGFORMER: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_SEGFORMER,
        demo_eval_name=MODEL_SEGFORMER,
        shared_registry_name=MODEL_SEGFORMER,
        local_constructor="SegFormer(n_classes, n_channels_reg)",
        shared_constructor="SegFormer(n_classes_seg)",
        safe_for_training_registry_construction=False,
        risk_level="medium/high",
        recommendation=(
            "Keep local until output heads, tuple order, and checkpoint keys "
            "are compared for the configured demo/eval mode."
        ),
    ),
    DEMO_EVAL_ARCH_SEGHARDNET: DemoEvalModelCompatibility(
        architecture_code=DEMO_EVAL_ARCH_SEGHARDNET,
        demo_eval_name=MODEL_SEGHARDNET,
        shared_registry_name=MODEL_SEGHARDNET,
        local_constructor="SegHarDNet(n_classes, n_channels_reg)",
        shared_constructor="SegHarDNet(n_classes_seg)",
        safe_for_training_registry_construction=False,
        risk_level="high",
        recommendation=(
            "Default demo/eval mode now uses a shared-model compatibility "
            "wrapper that preserves fixed output size. Keep copied local "
            "implementation for unverified non-default modes."
        ),
    ),
}


def get_demo_eval_architecture_name(architecture_code: int) -> str:
    """Resolve a demo/eval integer architecture code to its model name.

    Shared architectures are validated against the cleaned training registry.
    The copied repo's local-only ``TPEnet_a`` remains intentionally local.
    """
    try:
        arch_name = DEMO_EVAL_ARCHITECTURE_TO_NAME[architecture_code]
    except KeyError as exc:
        raise KeyError(f"Unsupported demo/eval architecture code: {architecture_code}") from exc

    if arch_name != DEMO_EVAL_LOCAL_ONLY_ARCH_NAME:
        registered = get_registered_model_names()
        if arch_name not in registered:
            raise KeyError(
                f"Architecture '{arch_name}' is not registered in ptsemseg.models.registry. "
                f"Registered names: {registered}"
            )

    return arch_name


def get_demo_eval_architecture_code(architecture_name: str) -> int:
    """Resolve a demo/eval architecture name to its legacy integer code."""
    try:
        return DEMO_EVAL_ARCHITECTURE_NAME_TO_CODE[architecture_name]
    except KeyError as exc:
        supported = ", ".join(DEMO_EVAL_ARCHITECTURE_NAME_TO_CODE)
        raise KeyError(
            f"Unsupported demo/eval architecture name: {architecture_name}. "
            f"Supported names: {supported}"
        ) from exc


def validate_demo_eval_architecture_config(
    architecture_code: int,
    architecture_name: str | None = None,
) -> str:
    """Validate optional YAML architecture-name metadata against the code.

    The copied demo/eval runtime still uses the integer architecture code as
    the behavior-driving value. This helper lets tests and docs verify that
    human-readable YAML metadata stays aligned without changing runtime model
    construction.
    """
    expected_name = get_demo_eval_architecture_name(architecture_code)
    if architecture_name is not None and architecture_name != expected_name:
        raise ValueError(
            "Demo/eval architecture config mismatch: "
            f"code {architecture_code} resolves to '{expected_name}', "
            f"but architecture_name is '{architecture_name}'."
        )
    return expected_name


def get_demo_eval_model_compatibility(architecture_code: int) -> dict[str, Any]:
    """Return the audited model-construction status for one architecture."""
    get_demo_eval_architecture_name(architecture_code)
    return asdict(_DEMO_EVAL_MODEL_COMPATIBILITY[architecture_code])


def get_demo_eval_model_compatibility_matrix() -> tuple[dict[str, Any], ...]:
    """Return audited model-construction status for all demo/eval architectures."""
    return tuple(
        get_demo_eval_model_compatibility(architecture_code)
        for architecture_code in sorted(DEMO_EVAL_ARCHITECTURE_TO_NAME)
    )


def can_construct_demo_eval_model_from_training_registry(architecture_code: int) -> bool:
    """Return whether it is currently safe to construct this architecture through ptsemseg."""
    get_demo_eval_architecture_name(architecture_code)
    return architecture_code in DEMO_EVAL_ARCHITECTURES_SAFE_FOR_TRAINING_REGISTRY_CONSTRUCTION


def load_demo_eval_checkpoint(model, fname_weights_to_be_loaded: str, arch: str):
    """Load demo/eval checkpoints through the cleaned training helper.

    This keeps the copied demo/eval checkpoint-path selection local and
    intentionally preserves the old file-loading behavior:

    - uses raw ``torch.load(path)`` with no forced ``map_location``
    - expects ``checkpoint['model_state']`` to exist

    It only reuses the cleaned training helper for the state-dict alignment
    and ``strict=False`` load step.
    """
    ckpt: Any = torch.load(fname_weights_to_be_loaded)
    state_dict_weights0 = ckpt["model_state"]
    print('loaded weights-to-be-loaded form %s !' % fname_weights_to_be_loaded)
    return align_and_load_state_dict(model=model, state_dict_weights0=state_dict_weights0)
