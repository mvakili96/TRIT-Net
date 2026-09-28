"""Runtime initialization for the legacy demo/eval entry point.

The functions here intentionally preserve the startup behavior from the copied
demo/eval runner: runtime defaults are resolved first, the demo preset and GT
pickle are loaded before the main loop, and both legacy argparse parsers still
parse the process command line.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Any, Dict, List

from ptsemseg.evaluation import MyHelper_GT
from ptsemseg.inference.demo_eval_args import define_args_algorithm
from ptsemseg.inference.demo_eval_args import define_args_operation
from ptsemseg.inference.demo_eval_args import set_value_for_args_algorithm
from ptsemseg.inference.runtime_defaults import get_demo_preset
from ptsemseg.inference.runtime_defaults import get_demo_runtime_settings
from ptsemseg.inference.runtime_defaults import get_metrics_output_dir
from ptsemseg.inference.runtime_defaults import get_output_subdirs


@dataclass
class DemoEvalRuntimeContext:
    """Resolved runtime state needed by the demo/eval processing loop."""

    runtime_settings: Dict[str, Any]
    title_testrun_this: str
    fname_pathlabel_gt_in: str
    format_fname_img_in: str
    format_fname_img_out: str
    w_img: int
    dx_valid_a: int
    dx_valid_b: int
    metrics_output_dir: str
    output_subdirs: Dict[str, str]
    demo_preset: Dict[str, Any]
    obj_helper_GT: MyHelper_GT
    list_pathlabel_gt_in: List[Any]
    totnum_steps: int
    architecture: int
    num_seg_classes: int
    num_channel_reg: int
    seg_in_pp: bool
    flag_miou: bool
    flag_save_img: bool
    flag_save_data: bool
    flag_single_multiple_path_evaluation: bool
    data_in_use: int
    args_oper: Any
    args_alg: Any


def initialize_demo_eval_runtime() -> DemoEvalRuntimeContext:
    """Resolve demo/eval settings exactly as the legacy runner did."""

    runtime_settings = get_demo_runtime_settings()

    title_testrun_this = runtime_settings["title_testrun_this"]

    w_img = 960
    metrics_output_dir = get_metrics_output_dir()
    output_subdirs = get_output_subdirs()

    demo_preset = runtime_settings.get("demo_preset", get_demo_preset(title_testrun_this))
    fname_pathlabel_gt_in = demo_preset["fname_pathlabel_gt_in"]
    format_fname_img_in = demo_preset["format_fname_img_in"]
    format_fname_img_out = demo_preset["format_fname_img_out"]
    dx_valid_a = demo_preset["dx_valid_a"]
    dx_valid_b = demo_preset["dx_valid_b"]

    obj_helper_GT = MyHelper_GT(title_testrun_this, w_img, dx_valid_a, dx_valid_b)

    with open(fname_pathlabel_gt_in, "rb") as fh:
        list_pathlabel_gt_in = pickle.load(fh)

    totnum_steps = len(list_pathlabel_gt_in)

    architecture = runtime_settings["architecture"]
    num_seg_classes = runtime_settings["num_seg_classes"]
    num_channel_reg = runtime_settings["num_channel_reg"]
    seg_in_pp = runtime_settings["seg_in_pp"]
    flag_miou = runtime_settings["flag_miou"]
    flag_save_img = runtime_settings["flag_save_img"]
    flag_save_data = runtime_settings["flag_save_data"]
    flag_single_multiple_path_evaluation = runtime_settings["flag_single_multiple_path_evaluation"]
    data_in_use = runtime_settings["data_in_use"]

    parser_oper = define_args_operation(data_in_use, architecture)
    parser_alg = define_args_algorithm()

    args_oper = parser_oper.parse_args()
    args_alg = parser_alg.parse_args()
    args_alg = set_value_for_args_algorithm(args_alg)

    return DemoEvalRuntimeContext(
        runtime_settings=runtime_settings,
        title_testrun_this=title_testrun_this,
        fname_pathlabel_gt_in=fname_pathlabel_gt_in,
        format_fname_img_in=format_fname_img_in,
        format_fname_img_out=format_fname_img_out,
        w_img=w_img,
        dx_valid_a=dx_valid_a,
        dx_valid_b=dx_valid_b,
        metrics_output_dir=metrics_output_dir,
        output_subdirs=output_subdirs,
        demo_preset=demo_preset,
        obj_helper_GT=obj_helper_GT,
        list_pathlabel_gt_in=list_pathlabel_gt_in,
        totnum_steps=totnum_steps,
        architecture=architecture,
        num_seg_classes=num_seg_classes,
        num_channel_reg=num_channel_reg,
        seg_in_pp=seg_in_pp,
        flag_miou=flag_miou,
        flag_save_img=flag_save_img,
        flag_save_data=flag_save_data,
        flag_single_multiple_path_evaluation=flag_single_multiple_path_evaluation,
        data_in_use=data_in_use,
        args_oper=args_oper,
        args_alg=args_alg,
    )


__all__ = [
    "DemoEvalRuntimeContext",
    "initialize_demo_eval_runtime",
]
