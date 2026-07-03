"""Compatibility wrapper for shared demo/eval argument helpers.

The active implementation lives in :mod:`ptsemseg.inference.demo_eval_args`.
This legacy module remains so old demo/eval imports keep working during the
deduplication transition.
"""

from ptsemseg.inference.demo_eval_args import define_args_algorithm
from ptsemseg.inference.demo_eval_args import define_args_operation
from ptsemseg.inference.demo_eval_args import set_value_for_args_algorithm
from ptsemseg.inference.demo_eval_args import str2bool

__all__ = [
    "define_args_algorithm",
    "define_args_operation",
    "set_value_for_args_algorithm",
    "str2bool",
]
