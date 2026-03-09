#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert ultrashape_v1_pretrained.pt into training-loadable ckpt files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "ultrashape_v1_pretrained.pt",
        help="Path to ultrashape_v1_pretrained.pt",
    )
    parser.add_argument(
        "--out-vae",
        type=Path,
        default=Path(__file__).resolve().parent / "ultrashape_v1_vae.ckpt",
        help="Output VAE ckpt path",
    )
    parser.add_argument(
        "--out-dit-cond",
        type=Path,
        default=Path(__file__).resolve().parent / "ultrashape_v1_dit_cond.ckpt",
        help="Output DiT+Conditioner ckpt path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input}")

    ckpt = torch.load(args.input, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    for key in ("vae", "dit", "conditioner"):
        if key not in ckpt:
            raise KeyError(f"Missing key '{key}' in checkpoint")

    # 1) VAE ckpt (for instantiate_vae_from_config_local)
    vae_state = ckpt["vae"]
    torch.save({"state_dict": vae_state}, args.out_vae)

    # 2) DiT + Conditioner ckpt (for Diffuser.init_from_ckpt)
    dit_state = {f"model.{k}": v for k, v in ckpt["dit"].items()}
    cond_state = {f"cond_stage_model.{k}": v for k, v in ckpt["conditioner"].items()}
    merged = {}
    merged.update(dit_state)
    merged.update(cond_state)
    torch.save({"state_dict": merged}, args.out_dit_cond)

    print("Saved:")
    print(f"  VAE ckpt: {args.out_vae} ({len(vae_state)} tensors)")
    print(f"  DiT+Cond ckpt: {args.out_dit_cond} ({len(merged)} tensors)")


if __name__ == "__main__":
    main()
