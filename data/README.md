# Polycam Sneakers Downloader

This script downloads 3D sneaker models from the Polycam "Sneakers & Shoes"
page and saves the assets under the local `polycam-downloads/` directory.

By default it saves glTF assets (`.gltf`, `.bin`, textures) which are easy to
convert into GLB.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 download_polycam_sneakers.py
```

Downloaded models will be stored under `polycam-downloads/` (one folder per capture).

## Options

```bash
python3 download_polycam_sneakers.py --help
```

Common flags:

- `--limit 20` to download only the first 20 models.
- `--prefer-glb` to download real GLB assets when available.
- `--include-unsavable` to include captures marked as unsavable.

## Convert to GLB

If you want GLB files, you can convert a downloaded `.gltf` like:

```bash
gltfpack -i polycam-downloads/<folder>/raw.gltf -o polycam-downloads/<folder>/model.glb
```

Any glTF-capable tool (Blender, gltf-transform, etc.) also works.
