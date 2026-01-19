# UltraShape Data Scripts

This folder contains two data-prep scripts:

- `download_polycam_sneakers.py` downloads Polycam sneaker models to `polycam-downloads/`.
- `prepare_dataset.py` converts downloads into GLB meshes and train/val/test splits.

## Environment (uv)

We manage the virtual environment with uv at `UltraShape-1.0/.venv`.

```bash
cd UltraShape-1.0
uv sync
```

Run scripts with uv:

```bash
uv run data/download_polycam_sneakers.py
uv run data/prepare_dataset.py --dry-run
```

Add Python deps to `pyproject.toml` with uv:

```bash
uv add requests
```

Legacy pip requirements still exist for reference:

- `data/requirements_polycam_download.txt`
- `data/requirements_prepare_dataset.txt`

## Script: download_polycam_sneakers.py

Downloads 3D sneaker models from the Polycam "Sneakers & Shoes" page and saves
assets under `data/polycam-downloads/` (one folder per capture). By default it
saves glTF assets (`.gltf`, `.bin`, textures), which are easy to convert to GLB.

Usage:

```bash
uv run data/download_polycam_sneakers.py --help
```

Common options:

- `--limit 20` download only the first 20 models.
- `--prefer-glb` fetch real GLB assets when available.
- `--include-unsavable` include captures marked as unsavable.
- `--request-retries 5` retry network requests more times.
- `--output-dir polycam-downloads` override output directory.

Examples:

```bash
uv run data/download_polycam_sneakers.py --limit 50 --prefer-glb
uv run data/download_polycam_sneakers.py --filters "tags:sneaker" --hits-per-page 25
```

## Script: prepare_dataset.py

Converts downloaded assets into GLB meshes, writes train/val/test splits, and
stores outputs under `data/dataset/`:

- `data/dataset/meshes/*.glb`
- `data/dataset/train.json`, `val.json`, `test.json`
- `data/dataset/manifest.json`, `mesh_paths.json`, `render.json`

Usage:

```bash
uv run data/prepare_dataset.py --help
```

Common options:

- `--gltfpack /path/to/gltfpack` override gltfpack path.
- `--blender /path/to/blender` override blender path.
- `--polycam-root data/polycam-downloads` input root for Polycam.
- `--artec-root data/artec3d-downloads` input root for Artec3D.
- `--dataset-root data/dataset` output root for the dataset.
- `--split 0.8,0.1,0.1` train/val/test split.
- `--overwrite` overwrite existing outputs.
- `--dry-run` print planned conversions only.

Examples:

```bash
uv run data/prepare_dataset.py --dry-run
uv run data/prepare_dataset.py --gltfpack /home/zhoujianyu/tools/gltfpack --blender /usr/bin/blender
```

## External tools

`prepare_dataset.py` uses external tools for conversion:

- `gltfpack` (preferred for `.gltf -> .glb`), configure via `--gltfpack` or `GLTFPACK`.
- `blender` (required for `.obj/.fbx/.dae/.ply/.stl`), configure via `--blender` or `BLENDER`.

Blender's glTF exporter requires `numpy` in Blender's own Python environment.
On Ubuntu, this is easiest via apt:

```bash
sudo apt install python3-numpy
blender -b --python-expr "import numpy as np; print(np.__version__)"
```

If `collada_import` is missing, `.dae` conversions may fail on some Blender builds.
