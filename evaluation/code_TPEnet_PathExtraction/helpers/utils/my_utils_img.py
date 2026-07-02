"""Compatibility wrapper for the shared demo/eval path-extraction helper.

The implementation lives in :mod:`ptsemseg.evaluation.path_extraction`.
Keep this module temporarily so any older local imports continue to resolve.
"""

from ptsemseg.evaluation.path_extraction import MyUtils_Image

__all__ = ["MyUtils_Image"]
