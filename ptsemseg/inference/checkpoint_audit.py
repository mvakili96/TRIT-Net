"""Checkpoint compatibility diagnostics for demo/eval integration.

These helpers are intentionally read-only. They inspect checkpoint state-dict
keys and shapes without changing runtime checkpoint-loading behavior.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Mapping

import torch


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    state_dict_key: str = "model_state",
    map_location: str = "cpu",
) -> OrderedDict[str, Any]:
    """Load one checkpoint state dict for compatibility auditing.

    Demo/eval runtime loading still expects ``checkpoint["model_state"]`` and
    uses its legacy load path. This diagnostic helper also defaults to
    ``model_state`` so audit results match that expectation, but maps tensors to
    CPU to keep inspection safe on machines without GPUs.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint mapping at {checkpoint_path}, "
            f"got {type(checkpoint).__name__}."
        )

    if state_dict_key not in checkpoint:
        raise KeyError(
            f"No '{state_dict_key}' in checkpoint. Found keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint[state_dict_key]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected checkpoint['{state_dict_key}'] to be a mapping, "
            f"got {type(state_dict).__name__}."
        )

    return OrderedDict(state_dict)


def compare_state_dict_keys(
    model_state_dict: Mapping[str, Any],
    checkpoint_state_dict: Mapping[str, Any],
    examples_limit: int = 20,
) -> dict[str, Any]:
    """Compare model and checkpoint keys/shapes without loading parameters."""
    model_keys = set(model_state_dict)
    checkpoint_keys = set(checkpoint_state_dict)
    shared_keys = sorted(model_keys & checkpoint_keys)

    shape_mismatches = []
    matching_shape_count = 0
    for key in shared_keys:
        model_shape = _shape_tuple(model_state_dict[key])
        checkpoint_shape = _shape_tuple(checkpoint_state_dict[key])
        if model_shape == checkpoint_shape:
            matching_shape_count += 1
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "model_shape": model_shape,
                    "checkpoint_shape": checkpoint_shape,
                }
            )

    missing_in_checkpoint = sorted(model_keys - checkpoint_keys)
    unexpected_in_checkpoint = sorted(checkpoint_keys - model_keys)

    return {
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "shared_key_count": len(shared_keys),
        "matching_shape_count": matching_shape_count,
        "shape_mismatch_count": len(shape_mismatches),
        "missing_in_checkpoint_count": len(missing_in_checkpoint),
        "unexpected_in_checkpoint_count": len(unexpected_in_checkpoint),
        "shape_mismatch_examples": shape_mismatches[:examples_limit],
        "missing_in_checkpoint_examples": missing_in_checkpoint[:examples_limit],
        "unexpected_in_checkpoint_examples": unexpected_in_checkpoint[:examples_limit],
    }


def compare_model_to_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    state_dict_key: str = "model_state",
    examples_limit: int = 20,
) -> dict[str, Any]:
    """Compare one model's state dict against a checkpoint state dict."""
    checkpoint_state_dict = load_checkpoint_state_dict(
        checkpoint_path=checkpoint_path,
        state_dict_key=state_dict_key,
    )
    summary = compare_state_dict_keys(
        model_state_dict=model.state_dict(),
        checkpoint_state_dict=checkpoint_state_dict,
        examples_limit=examples_limit,
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["state_dict_key"] = state_dict_key
    return summary


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)
