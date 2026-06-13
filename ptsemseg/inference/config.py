"""Shared config helpers for demo/eval runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ptsemseg.training.setup import load_config


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_EVAL_CONFIG_PATH = _REPO_ROOT / "configs" / "demo_eval.yml"


def get_repo_root() -> Path:
    return _REPO_ROOT


def get_default_demo_eval_config_path() -> str:
    return str(DEFAULT_DEMO_EVAL_CONFIG_PATH)


def load_demo_eval_config(config_path: str | None = None) -> Dict:
    resolved_path = config_path or get_default_demo_eval_config_path()
    return load_config(resolved_path)
