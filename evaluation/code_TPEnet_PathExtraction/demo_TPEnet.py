# 2020/8/11
# Jungwon Kang
"""Compatibility entry point for the copied TPEnet demo/eval script.

The runnable demo/eval program lives in ``demo_eval_runner.py``. This public
script stays intentionally thin while preserving the old launch command.
"""

from demo_eval_runner import run_demo_eval


if __name__ == "__main__":
    run_demo_eval()
