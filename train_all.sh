#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

NODE_RANK="${1:-0}"
MASTER_IP="${2:-127.0.0.1}"
shift $(( $# > 1 ? 2 : $# ))
EXTRA_ARGS=("$@")

VAE_CKPT_DIR="${VAE_CKPT_DIR:-outputs/vae_ultrashape/exp1_token8192/ckpt}"
DIT_CONFIG="${DIT_CONFIG:-configs/train_dit_refine.yaml}"

echo "[train_all] Start VAE training..."
bash train.sh vae "$NODE_RANK" "$MASTER_IP" "${EXTRA_ARGS[@]}"

if [ ! -d "$VAE_CKPT_DIR" ]; then
  echo "[train_all] Missing VAE ckpt dir: $VAE_CKPT_DIR" >&2
  exit 1
fi

mapfile -t CKPTS < <(ls -t "$VAE_CKPT_DIR"/ckpt-step=*.ckpt 2>/dev/null || true)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "[train_all] No VAE checkpoints found in $VAE_CKPT_DIR" >&2
  exit 1
fi

LATEST_CKPT="${CKPTS[0]}"
LATEST_REL="$(python3 - <<PY
import os
root = r\"$ROOT_DIR\"
ckpt = r\"$LATEST_CKPT\"
print(os.path.relpath(ckpt, root))
PY
)"

python3 - <<PY
import io

path = r\"$DIT_CONFIG\"
new_ckpt = r\"$LATEST_REL\"

with open(path, \"r\", encoding=\"utf-8\") as f:
    lines = f.read().splitlines()

updated = False
for i, line in enumerate(lines):
    if \"from_pretrained:\" in line:
        indent = line.split(\"from_pretrained:\")[0]
        lines[i] = f\"{indent}from_pretrained: {new_ckpt}\"
        updated = True
        break

if not updated:
    raise SystemExit(f\"from_pretrained not found in {path}\")

with open(path, \"w\", encoding=\"utf-8\") as f:
    f.write(\"\\n\".join(lines) + \"\\n\")
PY

echo "[train_all] Updated $DIT_CONFIG -> from_pretrained: $LATEST_REL"

echo "[train_all] Start DiT training..."
bash train.sh dit "$NODE_RANK" "$MASTER_IP" "${EXTRA_ARGS[@]}"
