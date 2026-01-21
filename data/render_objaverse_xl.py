#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Set, Tuple


def resolve_root_dir(root_dir: Optional[str]) -> str:
    if root_dir:
        return os.path.abspath(root_dir)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str, root_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def normalize_exts(mesh_exts: str) -> Tuple[str, ...]:
    exts: List[str] = []
    for ext in mesh_exts.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext)
    return tuple(exts)


def collect_mesh_paths(mesh_dir: str, mesh_exts: Tuple[str, ...], recursive: bool) -> List[str]:
    mesh_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(mesh_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in mesh_exts:
                    mesh_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(mesh_dir):
            path = os.path.join(mesh_dir, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in mesh_exts:
                mesh_paths.append(path)
    return sorted(mesh_paths)


def resolve_mesh_paths(
    mesh_json: Optional[str],
    mesh_dir: Optional[str],
    mesh_exts: Tuple[str, ...],
    recursive: bool,
    root_dir: str,
) -> List[str]:
    if mesh_json:
        mesh_json = resolve_path(mesh_json, root_dir)
        with open(mesh_json, "r", encoding="utf-8") as f:
            mesh_paths = json.load(f)
        if not isinstance(mesh_paths, list):
            raise ValueError(f"mesh_json must be a list of paths: {mesh_json}")
    elif mesh_dir:
        mesh_dir = resolve_path(mesh_dir, root_dir)
        if not os.path.isdir(mesh_dir):
            raise FileNotFoundError(f"mesh_dir not found: {mesh_dir}")
        mesh_paths = collect_mesh_paths(mesh_dir, mesh_exts, recursive=recursive)
    else:
        raise ValueError("Either mesh_json or mesh_dir must be provided")

    if not mesh_paths:
        raise ValueError("No mesh files found for rendering")

    resolved_paths: List[str] = []
    for path in mesh_paths:
        if not isinstance(path, str):
            raise ValueError("mesh paths must be strings")
        resolved_paths.append(resolve_path(path, root_dir))
    return resolved_paths


def resolve_rendering_dir(root_dir: str, rendering_dir: Optional[str]) -> str:
    if rendering_dir:
        render_dir = resolve_path(rendering_dir, root_dir)
    else:
        workspace_root = os.path.abspath(os.path.join(root_dir, ".."))
        render_dir = os.path.join(workspace_root, "objaverse-xl", "scripts", "rendering")
    blender_script = os.path.join(render_dir, "blender_script.py")
    if not os.path.isfile(blender_script):
        raise FileNotFoundError(f"blender_script.py not found under {render_dir}")
    return render_dir


def resolve_blender(blender_path: Optional[str]) -> str:
    if blender_path:
        blender = blender_path
        if os.path.isabs(blender) and not os.path.isfile(blender):
            raise FileNotFoundError(f"blender not found: {blender}")
        if not os.path.isabs(blender):
            resolved = shutil.which(blender)
            if resolved:
                blender = resolved
    else:
        blender = shutil.which("blender") or "blender"
    return blender


def mesh_uid(mesh_path: str) -> str:
    return os.path.splitext(os.path.basename(mesh_path))[0]


def has_renders(render_dir: str, num_renders: int) -> bool:
    if not os.path.isdir(render_dir):
        return False
    png_count = len([name for name in os.listdir(render_dir) if name.lower().endswith(".png")])
    return png_count >= num_renders


def render_mesh(
    blender_path: str,
    rendering_dir: str,
    mesh_path: str,
    output_dir: str,
    num_renders: int,
    engine: str,
    only_northern_hemisphere: bool,
    timeout: int,
    display: Optional[str],
    dry_run: bool,
) -> None:
    blender_script = os.path.join(rendering_dir, "blender_script.py")
    cmd = [
        blender_path,
        "--background",
        "--python",
        blender_script,
        "--",
        "--object_path",
        mesh_path,
        "--output_dir",
        output_dir,
        "--num_renders",
        str(num_renders),
        "--engine",
        engine,
    ]
    if only_northern_hemisphere:
        cmd.append("--only_northern_hemisphere")

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return

    env = os.environ.copy()
    if display:
        env["DISPLAY"] = display

    subprocess.run(cmd, cwd=rendering_dir, check=True, timeout=timeout, env=env)


def write_json(path: str, data: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def path_for_json(render_dir: str, root_dir: str) -> str:
    try:
        common = os.path.commonpath([render_dir, root_dir])
    except ValueError:
        common = ""
    if common and common == root_dir:
        return os.path.relpath(render_dir, root_dir)
    return render_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Render dataset meshes via objaverse-xl Blender scripts.")
    parser.add_argument("--mesh_json", type=str, default=None, help="Path to mesh_paths.json (overrides mesh_dir).")
    parser.add_argument("--mesh_dir", type=str, default="data/dataset/meshes", help="Directory containing mesh files.")
    parser.add_argument("--mesh_exts", type=str, default=".glb,.gltf,.obj,.ply,.stl,.fbx,.dae,.usd,.usda,.usdz,.abc,.blend",
                        help="Comma-separated mesh extensions.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan mesh_dir.")
    parser.add_argument("--root_dir", type=str, default=None, help="Base directory for relative paths.")
    parser.add_argument("--output_dir", type=str, default="data/dataset/render", help="Directory to save renders.")
    parser.add_argument("--render_json", type=str, default="data/dataset/render.json", help="Path to render.json.")
    parser.add_argument("--rendering_dir", type=str, default=None, help="Path to objaverse-xl/scripts/rendering.")
    parser.add_argument("--blender", type=str, default=None, help="Path to blender binary (overrides PATH).")
    parser.add_argument("--num_renders", type=int, default=16, help="Number of renders per mesh.")
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--only_northern_hemisphere", action="store_true", help="Render only the northern hemisphere.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout (seconds) per render.")
    parser.add_argument("--display", type=str, default=None, help="Override DISPLAY for headless rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Render even if outputs already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of meshes.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N meshes.")
    args = parser.parse_args()

    root_dir = resolve_root_dir(args.root_dir)
    mesh_exts = normalize_exts(args.mesh_exts)
    mesh_paths = resolve_mesh_paths(
        mesh_json=args.mesh_json,
        mesh_dir=args.mesh_dir,
        mesh_exts=mesh_exts,
        recursive=args.recursive,
        root_dir=root_dir,
    )

    if args.offset:
        mesh_paths = mesh_paths[args.offset:]
    if args.limit is not None:
        mesh_paths = mesh_paths[:args.limit]

    if not mesh_paths:
        print("No mesh files selected after applying offset/limit.")
        return 1

    rendering_dir = resolve_rendering_dir(root_dir, args.rendering_dir)
    blender_path = resolve_blender(args.blender)
    output_root = resolve_path(args.output_dir, root_dir)
    render_json_path = resolve_path(args.render_json, root_dir)

    os.makedirs(output_root, exist_ok=True)

    uid_to_render: Dict[str, str] = {}
    failures: List[str] = []
    failed_uids: Set[str] = set()
    duplicate_uids: Set[str] = set()

    for mesh_path in mesh_paths:
        uid = mesh_uid(mesh_path)
        if uid in uid_to_render:
            failures.append(f"{uid}: duplicate uid")
            failed_uids.add(uid)
            duplicate_uids.add(uid)
            continue
        render_dir = os.path.join(output_root, uid, "rgba")
        uid_to_render[uid] = render_dir

    for mesh_path in mesh_paths:
        uid = mesh_uid(mesh_path)
        if uid in duplicate_uids:
            continue
        render_dir = uid_to_render.get(uid)
        if not render_dir:
            continue

        if not os.path.isfile(mesh_path):
            failures.append(f"{uid}: mesh not found at {mesh_path}")
            failed_uids.add(uid)
            continue

        if not args.overwrite and has_renders(render_dir, args.num_renders):
            print(f"[Skip] {uid}: renders already exist")
            continue

        try:
            render_mesh(
                blender_path=blender_path,
                rendering_dir=rendering_dir,
                mesh_path=mesh_path,
                output_dir=render_dir,
                num_renders=args.num_renders,
                engine=args.engine,
                only_northern_hemisphere=args.only_northern_hemisphere,
                timeout=args.timeout,
                display=args.display,
                dry_run=args.dry_run,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{uid}: render timeout")
            failed_uids.add(uid)
            continue
        except subprocess.CalledProcessError as exc:
            failures.append(f"{uid}: render failed ({exc.returncode})")
            failed_uids.add(uid)
            continue

        if not has_renders(render_dir, args.num_renders) and not args.dry_run:
            failures.append(f"{uid}: missing renders after blender run")
            failed_uids.add(uid)

    render_json: Dict[str, str] = {}
    for uid, render_dir in uid_to_render.items():
        if uid in failed_uids:
            continue
        render_json[uid] = path_for_json(render_dir, root_dir)

    write_json(render_json_path, render_json)
    print(f"[Info] Wrote render.json with {len(render_json)} entries to {render_json_path}")

    if failures:
        print("[Error] Rendering failures:")
        for entry in failures:
            print(f"  - {entry}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
