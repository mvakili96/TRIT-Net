"""Model output-structure diagnostics for demo/eval integration.

These helpers are read-only. They are meant to compare model output tuple
structure before any copied demo/eval model is replaced by a shared model.
"""

from __future__ import annotations

from typing import Any

import torch


def summarize_model_outputs(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    """Run one no-grad forward pass and summarize output tensor structure."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    if was_training:
        model.train()

    flat_outputs = _as_output_sequence(outputs)
    return summarize_outputs(flat_outputs)


def summarize_outputs(outputs: Any) -> dict[str, Any]:
    """Summarize a tensor or tuple/list of tensors."""
    flat_outputs = _as_output_sequence(outputs)
    return {
        "output_count": len(flat_outputs),
        "outputs": [
            {
                "index": index,
                "type": type(output).__name__,
                "shape": _shape_tuple(output),
                "dtype": str(getattr(output, "dtype", None)),
            }
            for index, output in enumerate(flat_outputs)
        ],
    }


def compare_output_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare two output summaries for tuple length, shape, dtype, and type."""
    left_outputs = left_summary["outputs"]
    right_outputs = right_summary["outputs"]
    pair_count = min(len(left_outputs), len(right_outputs))

    pairwise = []
    for index in range(pair_count):
        left = left_outputs[index]
        right = right_outputs[index]
        pairwise.append(
            {
                "index": index,
                "type_match": left["type"] == right["type"],
                "shape_match": left["shape"] == right["shape"],
                "dtype_match": left["dtype"] == right["dtype"],
                "left": left,
                "right": right,
            }
        )

    return {
        "output_count_match": left_summary["output_count"] == right_summary["output_count"],
        "left_output_count": left_summary["output_count"],
        "right_output_count": right_summary["output_count"],
        "all_pair_types_match": all(item["type_match"] for item in pairwise),
        "all_pair_shapes_match": all(item["shape_match"] for item in pairwise),
        "all_pair_dtypes_match": all(item["dtype_match"] for item in pairwise),
        "pairwise": pairwise,
    }


def _as_output_sequence(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)
