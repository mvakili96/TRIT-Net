# TRIT-Net

TRIT-Net is a compact framework focused on rail-track scene understanding. It provides training utilities, data loaders, and multi-head models (segmentation, centerline heatmaps, and optional AFM head) derived from a lightweight ptsemseg-style codebase.

## Highlights

- Multi-head models (segmentation + centerline [+ AFM])
- Config-driven training via YAML files in `configs/`
- Custom schedulers and optimizer factories
- Reproducible training loop in `train_py/train_my_rpnet_c.py`
- Current tracked training path uses `Triplet_Loader` with a RailSem-style aligned-file dataset layout

## Repository layout

- `train_py/` — training scripts (main trainer: `train_my_rpnet_c.py`)
- `configs/` — YAML configs (e.g., `trit_net.yml`)
- `ptsemseg/` — models, loaders, schedulers, optimizers, losses
  - `ptsemseg/models/` — model implementations (multi-head decoders)
  - `ptsemseg/loader/` — dataset loaders and helpers
  - `ptsemseg/schedulers/` — custom LR schedulers and factory
  - `ptsemseg/optimizers/` — optimizer factory
- `helpers_my/` — compatibility wrappers for legacy utility imports
- `data/` — local dataset root (not included; expected layout below)
- `runs/` — local log/TensorBoard output directory

## Entry point and runtime artifacts

The main training entry point is:

```bash
python train_py/train_my_rpnet_c.py
```

By default, the trainer reads `configs/trit_net.yml`, creates a
`tensorboardX.SummaryWriter` directly on `./runs`, and writes the timestamped
training log file inside that same log directory. Runtime directories such as
`data/`, `runs/`, `train_py/uint8/`, `.ruff_cache/`, and `__pycache__/` are
local artifacts, not source code. They may contain datasets, event files,
generated visualizations, copied configs, or tool caches and should not be
cleaned up by refactors unless that is requested explicitly.

Checkpoint files are different: they are currently saved as
`Mybest_<iter>.pkl` in the current working directory, not inside `runs/`.
If you launch training from the repository root using the recommended command,
that means checkpoint files are written into the repository root itself.

Current cleanup work is intended to be behavior-preserving: avoid changing
training logic, model definitions, dataset behavior, losses, metrics,
checkpoint format, log format, or output format unless a later stage explicitly
calls for it.

## Data layout (current expected layout under `data/<DatasetName>/`)

- `jpgs/` — RGB images
- `C_image/` — centerline or contour images
- `AFM/` — AFM maps (only for 4-class setups)
- `Seg3/` or `Seg4/` — segmentation label folders depending on `num_seg_classes`

The current `Triplet_Loader` expects these directories to contain aligned
filenames in sorted order. Training uses the first `train_split` sorted items
from each directory, so the dataset layout is effectively a RailSem-style
paired-file layout rather than a generic multi-dataset abstraction.

At the moment, the tracked training path is documented for this layout only.
The README does not claim a separate actively maintained training path for
RailDB or RailSet.

Loader contract: loaders return a dict with keys used by the trainer, e.g.:

- `img_raw_fl_n` — raw image tensor
- `gt_img_label_seg` — segmentation label map
- `gt_labelmap_centerline` — centerline ground truth
- `gt_AFM` — AFM map (when `num_seg_classes: 4`)

## Configuration

Configs live in `configs/*.yml`. The main config used by the trainer is `configs/trit_net.yml`.
Important sections:

- `model.arch` — model architecture string (must match supported models)
- `network_image_sizes` — per-architecture input sizes (height `h`, width `w`)
- `data.train_split` — highest sorted file index used for training; for example, `6000` uses indexes `0..5999`
- `training.num_seg_classes` — must be `3` or `4` (trainer enforces this)
- `training.checkpoint_interval` — training-only checkpoint save interval
- `training.optimizer` — optimizer selection and hyperparams
- `training.lr_schedule` — scheduler selection and params; when `max_iter` is omitted, the trainer uses `training.train_iters`
- `weight_init_t` — per-architecture initialization-checkpoint paths used before training starts; the trainer looks up `weight_init_t[model.arch]` and loads that checkpoint unless the value is `-1`

The current default config also contains machine-local assumptions:

- `data.root` points to `./data/RailSem19/`
- `weight_init_t.*` points to local checkpoint files under `./runs/...`
- `model.arch` selects the active architecture to train

`weight_init_t` expects a checkpoint compatible with the selected architecture.
The current loader accepts checkpoints containing either `model_state` or
`state_dict`. If the selected path is wrong for your machine, or you want to
train from scratch, edit that entry before launching training.
This is one of the first settings a new user should expect to change on a new
machine.

## Quick setup (recommended)

Create a conda environment and install dependencies. The repo now separates:

- `requirements.txt` — core runtime packages for training and model import
- `requirements-dev.txt` — core runtime packages plus formatter/linter tools
- `requirements-full.txt` — the broader pinned local environment snapshot that was previously stored in `requirements.txt`

Recommended setup:

```bash
conda create -n trit python=3.10 -y
conda activate trit

# install torch/torchvision for your CPU/CUDA setup if needed
# see: https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Optional:

```bash
# for local formatting/linting
pip install -r requirements-dev.txt

# for recreating the broader historical environment snapshot
pip install -r requirements-full.txt
```

## Run training (example)

Use the default config or pass a config path with `-c/--config`.

Before the first real run on a new machine, check these four items:

1. Install the core runtime dependencies from `requirements.txt`, plus a compatible `torch` and `torchvision` build for your CPU/CUDA setup.
2. Confirm that `data.root` in `configs/trit_net.yml` points at a dataset directory laid out as `jpgs/`, `C_image/`, `AFM/`, and `Seg3` or `Seg4` with aligned sorted filenames.
3. Confirm that `weight_init_t[model.arch]` points to a checkpoint that exists on your machine, or change that value to `-1` if you want to skip initialization from a checkpoint.
4. Decide where you want checkpoint files to land: by default they are written into the current working directory when you run the trainer.

Simple run example (from repo root):

```bash
conda activate trit
python train_py/train_my_rpnet_c.py
```

For a short dry-run, use a temporary config with a small `train_iters`, `checkpoint_interval`, and `batch_size`, then pass it with `-c`.

If you are unsure whether your setup is correct, start with:

```bash
python train_py/train_my_rpnet_c.py --help
```

Then try a one-iteration smoke run with a temporary config rather than a full training run.

## Logs and checkpoints

- TensorBoard event files are written directly under the selected logdir, which defaults to `./runs`
- The trainer also copies the config file into the selected logdir on a best-effort basis
- The Python logger writes `run_<timestamp>.log` into the selected logdir
- Model checkpoints are saved as `Mybest_<iter>.pkl` in the current working directory
- Checkpoints are written when `(iteration + 1) % training.checkpoint_interval == 0` and again at the final training iteration
- If you want checkpoints somewhere else, that is not currently a config option; it would require changing the checkpoint helper or the process working directory

View logs:

```bash
tensorboard --logdir runs
```

## Adding a new model or dataset

- New model: add a Python file under `ptsemseg/models/` and ensure it follows the multi-head output contract. Register or import it where the project selects models.
- New dataset: follow the pattern in `ptsemseg/loader/` — return the same dict keys and respect the `size_img_rsz`/`size_out` contract. Note that the currently documented and verified training path is the RailSem-style aligned-file layout described above.
- Add a new config in `configs/` and set `fname_config` in the trainer or extend the trainer CLI.

## Linting & formatting

This repo contains a `pyproject.toml` with Black and Ruff settings. Run formatters locally before committing:

```bash
pip install ruff black
ruff check . && black .
```

## Contributing

- Open an issue or PR describing the change.
- Keep changes behavior-preserving if you are refactoring.
- Run linters/formatters on changed files.

## Notes

- The training setup helper currently centralizes these runtime defaults: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, `CUDA_LAUNCH_BLOCKING=1`, and `TF_ENABLE_ONEDNN_OPTS=0`. These are code-level defaults, not config-file settings.
- The project was bootstrapped from a compact ptsemseg-derived pipeline tailored for rail segmentation tasks.
