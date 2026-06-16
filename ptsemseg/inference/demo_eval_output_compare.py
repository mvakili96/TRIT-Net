"""One-image demo/eval output comparison helpers.

This module compares the copied local SegHarDNet implementation against the
shared demo/eval wrapper without changing the public demo/eval entry point.
It is intended for Stage G verification before deleting copied model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ptsemseg.inference.checkpoint_audit import load_checkpoint_state_dict
from ptsemseg.training.weights import align_and_load_state_dict


DEFAULT_OUTPUT_ROOT = Path("/tmp/trit_net_demo_eval_compare")


def compare_default_seghardnet_demo_outputs(
    image_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Compare old copied and new wrapped default SegHarDNet demo outputs.

    The comparison runs the real ``PathExtraction_TPEnet.process()`` pipeline
    twice: once with the copied local model class and once with the current
    factory path. Checkpoints are loaded with CPU-safe diagnostics only inside
    this harness so public demo/eval loading behavior remains unchanged.
    """
    repo_root = Path(__file__).resolve().parents[2]
    eval_root = repo_root / "evaluation" / "code_TPEnet_PathExtraction"
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    with _temporary_sys_path(eval_root), _temporary_cwd(eval_root):
        from helpers.models import get_model as current_eval_get_model
        from helpers.models.SegHarDNet import SegHarDNet as LocalSegHarDNet
        from helpers.utils import my_utils_net
        from helpers.utils.my_utils_img import MyUtils_Image
        from runtime_defaults import ARCH_SEGHARDNET
        from runtime_defaults import get_default_processing_size
        from runtime_defaults import get_demo_runtime_settings
        import PE_TPEnet
        import my_args_TPEnet

        settings = get_demo_runtime_settings()
        selected_image_path = _resolve_image_path(
            image_path=image_path,
            eval_root=eval_root,
            default_input_dir=settings["dir_input"],
        )
        checkpoint_path = _resolve_eval_path(settings["file_weight_override"], eval_root)

        args_alg = my_args_TPEnet.define_args_algorithm(
            settings["dataset_for_use"],
            settings["architecture"],
        ).parse_args([])
        args_alg = my_args_TPEnet.set_value_for_args_algorithm(
            settings["dataset_for_use"],
            args_alg,
        )

        image_utils = MyUtils_Image()
        image_size = get_default_processing_size(ARCH_SEGHARDNET)
        img_raw = image_utils.read_img_raw_jpg_from_file(str(selected_image_path), image_size)

        def local_get_model(model_dict, n_classes, n_channels_reg, version=None):
            return LocalSegHarDNet(n_classes=n_classes, n_channels_reg=n_channels_reg)

        def cpu_safe_checkpoint_loader(model, fname_weights_to_be_loaded, arch):
            state_dict = load_checkpoint_state_dict(
                _resolve_eval_path(fname_weights_to_be_loaded, eval_root)
            )
            return align_and_load_state_dict(model=model, state_dict_weights0=state_dict)

        local_extractor = _build_extractor_with_patches(
            PE_TPEnet=PE_TPEnet,
            my_utils_net=my_utils_net,
            args_alg=args_alg,
            num_seg_classes=settings["num_seg_classes"],
            num_channel_reg=settings["num_channel_reg"],
            seg_in_pp=settings["seg_in_pp"],
            architecture=settings["architecture"],
            get_model_fn=local_get_model,
            load_checkpoint_fn=cpu_safe_checkpoint_loader,
        )
        wrapper_extractor = _build_extractor_with_patches(
            PE_TPEnet=PE_TPEnet,
            my_utils_net=my_utils_net,
            args_alg=args_alg,
            num_seg_classes=settings["num_seg_classes"],
            num_channel_reg=settings["num_channel_reg"],
            seg_in_pp=settings["seg_in_pp"],
            architecture=settings["architecture"],
            get_model_fn=current_eval_get_model,
            load_checkpoint_fn=cpu_safe_checkpoint_loader,
        )

        local_outputs = _run_one_image(local_extractor, img_raw)
        wrapper_outputs = _run_one_image(wrapper_extractor, img_raw)
        image_id = _extract_image_id(selected_image_path.name)
        output_root.mkdir(parents=True, exist_ok=True)

        comparisons = []
        for output_name, extension in (
            ("IMG", ".jpg"),
            ("SEG", ".bmp"),
            ("CEN", ".png"),
            ("AFM", ".png"),
        ):
            filename = f"resluting_image_{image_id}{extension}"
            local_array = local_outputs[output_name]
            wrapper_array = wrapper_outputs[output_name]
            local_path = output_root / "copied_eval" / output_name / filename
            wrapper_path = output_root / "shared_wrapper" / output_name / filename
            _write_image(local_path, local_array)
            _write_image(wrapper_path, wrapper_array)
            comparisons.append(
                {
                    "output": output_name,
                    "filename": filename,
                    "array": _compare_arrays(local_array, wrapper_array),
                    "file_sha256_match": _sha256(local_path) == _sha256(wrapper_path),
                    "copied_eval_path": str(local_path),
                    "shared_wrapper_path": str(wrapper_path),
                }
            )

        return {
            "image_path": str(selected_image_path),
            "checkpoint_path": str(checkpoint_path),
            "output_root": str(output_root),
            "factory_model_type": type(current_eval_get_model({"arch": "seghardnet"}, 4, 1)).__name__,
            "all_arrays_exact": all(item["array"]["exact"] for item in comparisons),
            "all_file_hashes_match": all(item["file_sha256_match"] for item in comparisons),
            "comparisons": comparisons,
        }


def _build_extractor_with_patches(
    *,
    PE_TPEnet,
    my_utils_net,
    args_alg,
    num_seg_classes,
    num_channel_reg,
    seg_in_pp,
    architecture,
    get_model_fn,
    load_checkpoint_fn,
):
    original_get_model = PE_TPEnet.get_model
    original_load_checkpoint = my_utils_net.load_demo_eval_checkpoint
    PE_TPEnet.get_model = get_model_fn
    my_utils_net.load_demo_eval_checkpoint = load_checkpoint_fn
    try:
        return PE_TPEnet.PathExtraction_TPEnet(
            args_alg,
            num_seg_classes,
            num_channel_reg,
            seg_in_pp,
            architecture,
        )
    finally:
        PE_TPEnet.get_model = original_get_model
        my_utils_net.load_demo_eval_checkpoint = original_load_checkpoint


def _run_one_image(extractor, img_raw: np.ndarray) -> dict[str, np.ndarray]:
    (
        list_res_paths,
        _dict_res_time,
        _dict_res_imgs,
        _img_res_center_combined,
        img_res_seg,
        _model_seg_output,
        _model_cen_output,
        img_res_centerness,
        img_res_AFM_direct,
    ) = extractor.process(np.copy(img_raw))
    final_img = extractor.show_final_path_on_ori_v0(list_res_paths, np.copy(img_raw))
    return {
        "IMG": final_img,
        "SEG": img_res_seg,
        "CEN": img_res_centerness,
        "AFM": img_res_AFM_direct,
    }


def _resolve_image_path(
    *,
    image_path: str | Path | None,
    eval_root: Path,
    default_input_dir: str,
) -> Path:
    if image_path is not None:
        return _resolve_eval_path(image_path, eval_root)

    input_dir = _resolve_eval_path(default_input_dir, eval_root)
    image_files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    return sorted(image_files, key=lambda path: _natural_sort_key(path.name))[0]


def _resolve_eval_path(path: str | Path, eval_root: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    candidate = eval_root / resolved
    if candidate.exists():
        return candidate
    return resolved


def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "exact": left is right,
            "left_shape": None if left is None else tuple(left.shape),
            "right_shape": None if right is None else tuple(right.shape),
            "max_abs_diff": None,
        }

    same_shape = left.shape == right.shape
    if same_shape:
        max_abs_diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    else:
        max_abs_diff = None
    return {
        "exact": bool(same_shape and np.array_equal(left, right)),
        "left_shape": tuple(int(dim) for dim in left.shape),
        "right_shape": tuple(int(dim) for dim in right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "max_abs_diff": max_abs_diff,
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        raise ValueError(f"Cannot write missing image for {path}")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_image_id(filename: str) -> int:
    numbers = re.findall(r"\d+", filename)
    if not numbers:
        raise ValueError(f"No numeric image id found in {filename}")
    return int(numbers[-1])


def _natural_sort_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


@contextmanager
def _temporary_sys_path(path: Path):
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(path_str)


@contextmanager
def _temporary_cwd(path: Path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Compare default SegHarDNet demo/eval outputs.")
    parser.add_argument("--image", default=None, help="Optional image path to compare.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for copied-eval and shared-wrapper output images.",
    )
    args = parser.parse_args()
    result = compare_default_seghardnet_demo_outputs(
        image_path=args.image,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
