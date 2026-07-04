"""Execution setup helpers for the legacy demo/eval runner."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, List

from ptsemseg.inference.path_extraction import PathExtraction_TPEnet


@dataclass
class DemoEvalExecutionContext:
    """Resolved input list and processor used by the demo/eval loop."""

    list_fnames_img: List[str]
    path_extractor: PathExtraction_TPEnet


def initialize_demo_eval_execution(
    args_oper: Any,
    args_alg: Any,
    num_seg_classes: int,
    num_channel_reg: int,
    seg_in_pp: bool,
    architecture: int,
) -> DemoEvalExecutionContext:
    """Preserve legacy input discovery and path extractor construction."""

    print("Process all image inside : {}".format(args_oper.dir_input))

    list_fnames_img = os.listdir(args_oper.dir_input)
    list_fnames_img.sort(key=lambda f: int(re.sub(r"\D", "", f)))

    path_extractor = PathExtraction_TPEnet(
        args_alg,
        num_seg_classes,
        num_channel_reg,
        seg_in_pp,
        architecture,
    )

    return DemoEvalExecutionContext(
        list_fnames_img=list_fnames_img,
        path_extractor=path_extractor,
    )


__all__ = [
    "DemoEvalExecutionContext",
    "initialize_demo_eval_execution",
]
