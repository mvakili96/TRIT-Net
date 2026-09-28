"""Public TRIT-Net demo/eval entry point.

The script preserves the legacy launch path while using shared inference code.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ptsemseg.inference.demo_eval_pipeline import run_demo_eval


if __name__ == "__main__":
    run_demo_eval()
