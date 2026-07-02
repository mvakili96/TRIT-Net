"""Compatibility wrapper for the shared demo/eval path extractor.

The implementation lives in :mod:`ptsemseg.inference.path_extraction`.
This module keeps the legacy ``import PE_TPEnet`` path available during the
transition.
"""

from ptsemseg.inference.path_extraction import PathExtraction_TPEnet

__all__ = ["PathExtraction_TPEnet"]
