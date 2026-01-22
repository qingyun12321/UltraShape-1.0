#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        if key.startswith("_forward_module."):
            key = key[len("_forward_module."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "module" in ckpt and isinstance(ckpt["module"], dict):
            state = ckpt["module"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    return _strip_prefixes(state)


def _strip_common_prefix(state: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    for prefix in prefixes:
        if state and all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def _extract_sub_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise ValueError(f"No keys found with prefix '{prefix}'")
    return sub


def _resolve_latest_ckpt_dir(root: str) -> str:
    root = os.path.expanduser(root)
    if os.path.isfile(root):
        return root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint path not found: {root}")

    # If this is a deepspeed ckpt dir (ckpt-step=*.ckpt), grab model states.
    ds_candidates = glob.glob(os.path.join(root, "checkpoint", "*_model_states.pt"))
    if ds_candidates:
        return sorted(ds_candidates, key=os.path.getmtime)[-1]

    # Otherwise, try to find latest ckpt-step directory or .ckpt file.
    step_dirs = sorted(glob.glob(os.path.join(root, "ckpt-step=*.ckpt")), key=os.path.getmtime)
    if step_dirs:
        return _resolve_latest_ckpt_dir(step_dirs[-1])

    ckpt_files = sorted(glob.glob(os.path.join(root, "*.ckpt")), key=os.path.getmtime)
    if ckpt_files:
        return ckpt_files[-1]

    raise FileNotFoundError(f"No checkpoint files found under: {root}")


def build_infer_ckpt(dit_ckpt_root: str, vae_ckpt_root: Optional[str], output_path: str) -> None:
    dit_ckpt_path = _resolve_latest_ckpt_dir(dit_ckpt_root)
    dit_state = _load_state_dict(dit_ckpt_path)

    vae_state = _extract_sub_state(dit_state, "first_stage_model.")
    dit_model_state = _extract_sub_state(dit_state, "model.")
    cond_state = _extract_sub_state(dit_state, "cond_stage_model.")

    if vae_ckpt_root:
        vae_ckpt_path = _resolve_latest_ckpt_dir(vae_ckpt_root)
        vae_state = _load_state_dict(vae_ckpt_path)
        vae_state = _strip_common_prefix(vae_state, ("vae_model.", "model.", "first_stage_model."))

    payload = {
        "vae": vae_state,
        "dit": dit_model_state,
        "conditioner": cond_state,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)

    print(f"[build_infer_ckpt] dit_ckpt={dit_ckpt_path}")
    if vae_ckpt_root:
        print(f"[build_infer_ckpt] vae_ckpt={vae_ckpt_root}")
    print(f"[build_infer_ckpt] output={output_path}")
    print(f"[build_infer_ckpt] keys: vae={len(vae_state)}, dit={len(dit_model_state)}, conditioner={len(cond_state)}")


def _resolve_repo_path(path: Optional[str], repo_root: Path) -> Optional[Path]:
    if path is None:
        return None
    path = Path(os.path.expanduser(path))
    if path.is_absolute():
        return path
    # Allow explicit cwd-relative usage with ./ or ../
    if str(path).startswith("."):
        return (Path.cwd() / path).resolve()
    return (repo_root / path).resolve()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build inference .pt from training checkpoints")
    parser.add_argument(
        "--dit-ckpt",
        default="outputs/dit_ultrashape/exp1_token8192/ckpt",
        help="DiT checkpoint dir or file (deepspeed ckpt dir is supported)",
    )
    parser.add_argument(
        "--vae-ckpt",
        default=None,
        help="Optional VAE checkpoint dir or file (overrides VAE weights from DiT ckpt)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/ultrashape_v1.pt",
        help="Output .pt path",
    )
    args = parser.parse_args()

    dit_ckpt = _resolve_repo_path(args.dit_ckpt, repo_root)
    vae_ckpt = _resolve_repo_path(args.vae_ckpt, repo_root)
    output_path = _resolve_repo_path(args.output, repo_root)
    if output_path is None:
        raise ValueError("Output path cannot be empty")
    if output_path.is_dir():
        output_path = output_path / "ultrashape_v1.pt"

    build_infer_ckpt(str(dit_ckpt), str(vae_ckpt) if vae_ckpt else None, str(output_path))


if __name__ == "__main__":
    main()
