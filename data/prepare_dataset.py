#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".dae"}
POLYCAM_PREFERRED = ("model.glb", "raw.glb", "model.gltf", "raw.gltf")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z._-]+", "_", value)
    value = value.strip("._-")
    return value or "asset"


def parse_split(value: str):
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 3 or any(p < 0 for p in parts):
        raise ValueError("split must be three non-negative numbers, e.g. 0.8,0.1,0.1")
    total = sum(parts)
    if total <= 0:
        raise ValueError("split total must be > 0")
    return [p / total for p in parts]


def find_polycam_models(root: Path):
    models = []
    if not root.exists():
        return models
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        candidates = []
        for name in POLYCAM_PREFERRED:
            path = entry / name
            if path.exists():
                candidates.append(path)
        if not candidates:
            candidates = sorted([p for p in entry.iterdir()
                                 if p.is_file() and p.suffix.lower() in {".glb", ".gltf"}])
        if not candidates:
            continue
        candidates.sort(key=lambda p: (p.suffix.lower() != ".glb", p.name))
        models.append(candidates[0])
    return models


def find_generic_models(root: Path):
    models = []
    if not root.exists():
        return models
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            models.append(path)
    return models


def make_uid(prefix: str, rel_path: Path) -> str:
    stem = slugify(str(rel_path.with_suffix("")))
    digest = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{stem}_{digest}"


def is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def normalize_executable(path: Path, name: str) -> Path:
    if path.is_dir():
        return path / name
    return path


def resolve_tool(name: str, override=None, env_var=None, extra_paths=None):
    if override:
        candidate = normalize_executable(Path(override).expanduser(), name)
        if is_executable(candidate):
            return str(candidate)
        raise SystemExit(f"{name} not found or not executable at {candidate}")

    env_value = os.environ.get(env_var, "").strip() if env_var else ""
    if env_value:
        candidate = normalize_executable(Path(env_value).expanduser(), name)
        if is_executable(candidate):
            return str(candidate)
        print(f"Warning: {env_var} set but not executable at {candidate}", file=sys.stderr)

    found = shutil.which(name)
    if found:
        return found

    for extra in extra_paths or []:
        candidate = normalize_executable(extra, name)
        if is_executable(candidate):
            return str(candidate)
    return None


def detect_tools(repo_root: Path, args):
    return {
        "gltfpack": resolve_tool(
            "gltfpack",
            args.gltfpack,
            "GLTFPACK",
            extra_paths=[
                repo_root / "tools" / "gltfpack",
                Path.home() / "tools" / "gltfpack",
            ],
        ),
        "blender": resolve_tool(
            "blender",
            args.blender,
            "BLENDER",
            extra_paths=[Path("/usr/local/bin/blender")],
        ),
    }


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_gltfpack(src: Path, dst: Path, gltfpack: str):
    ensure_parent(dst)
    cmd = [gltfpack, "-i", src.name, "-o", str(dst)]
    subprocess.run(cmd, cwd=src.parent, check=True)


BLENDER_SCRIPT = r"""
import bpy
import os
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) != 2:
    raise SystemExit("Usage: blender -b --python script.py -- <src> <dst>")

src, dst = argv
ext = os.path.splitext(src)[1].lower()

bpy.ops.wm.read_factory_settings(use_empty=True)

if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=src)
elif ext == ".obj":
    try:
        bpy.ops.import_scene.obj(filepath=src)
    except Exception:
        bpy.ops.wm.obj_import(filepath=src)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext == ".ply":
    bpy.ops.import_mesh.ply(filepath=src)
elif ext == ".stl":
    bpy.ops.import_mesh.stl(filepath=src)
elif ext == ".dae":
    bpy.ops.wm.collada_import(filepath=src)
else:
    raise SystemExit(f"Unsupported extension: {ext}")

bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', export_apply=True)
"""


def run_blender(src: Path, dst: Path, blender: str):
    ensure_parent(dst)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(BLENDER_SCRIPT)
        script_path = handle.name
    try:
        cmd = [blender, "-b", "--python", script_path, "--", str(src), str(dst)]
        subprocess.run(cmd, check=True)
    finally:
        Path(script_path).unlink(missing_ok=True)


def convert_to_glb(src: Path, dst: Path, tools, overwrite: bool):
    if dst.exists() and not overwrite:
        return "skipped"
    ext = src.suffix.lower()
    if ext == ".glb":
        ensure_parent(dst)
        shutil.copy2(src, dst)
        return "copied"
    if ext == ".gltf":
        if tools["gltfpack"]:
            run_gltfpack(src, dst, tools["gltfpack"])
            return "gltfpack"
        if tools["blender"]:
            run_blender(src, dst, tools["blender"])
            return "blender"
        raise RuntimeError("gltfpack/blender not available for .gltf conversion")
    if tools["blender"]:
        run_blender(src, dst, tools["blender"])
        return "blender"
    raise RuntimeError(f"blender not available for {ext} conversion")


def write_json(path: Path, data, overwrite: bool):
    if path.exists() and not overwrite:
        return False
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare UltraShape dataset from downloads.")
    parser.add_argument("--polycam-root", default="data/polycam-downloads",
                        help="Polycam downloads directory (relative to repo root).")
    parser.add_argument("--artec-root", default="data/artec3d-downloads",
                        help="Artec3D downloads directory (relative to repo root).")
    parser.add_argument("--dataset-root", default="data/dataset",
                        help="Dataset output directory (relative to repo root).")
    parser.add_argument("--gltfpack", help="Path to gltfpack binary (overrides PATH).")
    parser.add_argument("--blender", help="Path to blender binary (overrides PATH).")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="List sources without converting.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    polycam_root = (repo_root / args.polycam_root).resolve()
    artec_root = (repo_root / args.artec_root).resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    meshes_dir = dataset_root / "meshes"
    sample_dir = dataset_root / "sample"
    render_root = dataset_root / "render"

    split = parse_split(args.split)
    tools = detect_tools(repo_root, args)

    polycam_models = find_polycam_models(polycam_root)
    artec_models = find_generic_models(artec_root)

    if not polycam_root.exists():
        print(f"Polycam root missing: {polycam_root}")
    if not artec_root.exists():
        print(f"Artec root missing: {artec_root}")
    print(f"Found {len(polycam_models)} Polycam models and {len(artec_models)} Artec models.")
    print(f"Tools: gltfpack={tools['gltfpack'] or 'not found'}, blender={tools['blender'] or 'not found'}")

    sources = []
    for path in polycam_models:
        rel = path.relative_to(polycam_root)
        sources.append(("polycam", path, rel))
    for path in artec_models:
        rel = path.relative_to(artec_root)
        sources.append(("artec", path, rel))

    if not sources:
        print("No source models found.")
        return 1

    sources = list(dict.fromkeys(sources))
    sources.sort(key=lambda item: str(item[1]))

    if args.dry_run:
        for prefix, path, rel in sources:
            uid = make_uid(prefix, rel)
            print(f"{prefix}: {path} -> {uid}.glb")
        return 0

    meshes_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    processed = []
    failures = []

    for prefix, src, rel in sources:
        uid = make_uid(prefix, rel)
        dst = meshes_dir / f"{uid}.glb"
        try:
            action = convert_to_glb(src, dst, tools, overwrite=args.overwrite)
            processed.append(uid)
            manifest.append({
                "uid": uid,
                "source": str(src.relative_to(repo_root)),
                "output": str(dst.relative_to(repo_root)),
                "action": action,
            })
        except Exception as exc:
            failures.append((src, str(exc)))
            manifest.append({
                "uid": uid,
                "source": str(src.relative_to(repo_root)),
                "output": str(dst.relative_to(repo_root)),
                "action": "failed",
                "error": str(exc),
            })

    mesh_paths = [str((meshes_dir / f"{uid}.glb").relative_to(repo_root)) for uid in processed]
    render_json = {uid: str((render_root / uid / "rgba").relative_to(repo_root)) for uid in processed}

    rng = random.Random(args.seed)
    shuffled = processed[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_n = int(total * split[0])
    val_n = int(total * split[1])
    test_n = total - train_n - val_n

    train_ids = shuffled[:train_n]
    val_ids = shuffled[train_n:train_n + val_n]
    test_ids = shuffled[train_n + val_n:]

    write_json(dataset_root / "mesh_paths.json", mesh_paths, overwrite=args.overwrite)
    write_json(dataset_root / "render.json", render_json, overwrite=args.overwrite)
    write_json(dataset_root / "train.json", train_ids, overwrite=args.overwrite)
    write_json(dataset_root / "val.json", val_ids, overwrite=args.overwrite)
    write_json(dataset_root / "test.json", test_ids, overwrite=args.overwrite)
    write_json(dataset_root / "manifest.json", manifest, overwrite=args.overwrite)

    if failures:
        print("\nSome files failed to convert:")
        for src, err in failures:
            print(f"- {src}: {err}")
        print("\nInstall gltfpack or Blender to convert non-GLB assets.")

    print(f"\nProcessed {len(processed)} assets. Failed: {len(failures)}.")
    print(f"Output: {dataset_root.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
