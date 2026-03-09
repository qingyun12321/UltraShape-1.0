#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

CONDA_ENV="/mnt/data/miniconda3/envs/ultrashape"
PYTHON_BIN="$CONDA_ENV/bin/python"

if [ -x "$PYTHON_BIN" ]; then
  export CONDA_PREFIX="$CONDA_ENV"
  export PATH="$CONDA_ENV/bin:$PATH"
  if [ -d "$CONDA_ENV/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  PYTHON="$PYTHON_BIN"
else
  PYTHON="python"
fi

: "${ULTRASHAPE_DISABLE_FLASH_ATTN:=1}"
export ULTRASHAPE_DISABLE_FLASH_ATTN

select_idle_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -nr \
      | head -n1 \
      | awk -F',' '{gsub(/ /,"",$1); print $1}'
  else
    echo ""
  fi
}

: "${ULTRASHAPE_CUDA_VISIBLE_DEVICES:=auto}"
if [ "$ULTRASHAPE_CUDA_VISIBLE_DEVICES" = "auto" ]; then
  AUTO_GPU="$(select_idle_gpu)"
  if [ -n "$AUTO_GPU" ]; then
    ULTRASHAPE_CUDA_VISIBLE_DEVICES="$AUTO_GPU"
  else
    ULTRASHAPE_CUDA_VISIBLE_DEVICES="0"
  fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$ULTRASHAPE_CUDA_VISIBLE_DEVICES"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
: "${ULTRASHAPE_OCTREE_RES:=512}"
: "${ULTRASHAPE_STEPS:=50}"
: "${ULTRASHAPE_CHUNK_SIZE:=2048}"
: "${ULTRASHAPE_NUM_LATENTS:=}"
: "${ULTRASHAPE_SCALE:=0.99}"
: "${ULTRASHAPE_SEED:=42}"
: "${ULTRASHAPE_REMOVE_BG:=0}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[run] ULTRASHAPE_OCTREE_RES=${ULTRASHAPE_OCTREE_RES}"
echo "[run] ULTRASHAPE_STEPS=${ULTRASHAPE_STEPS}"
echo "[run] ULTRASHAPE_CHUNK_SIZE=${ULTRASHAPE_CHUNK_SIZE}"
if [ -n "$ULTRASHAPE_NUM_LATENTS" ]; then
  echo "[run] ULTRASHAPE_NUM_LATENTS=${ULTRASHAPE_NUM_LATENTS}"
fi

# Allow overriding inputs with args or env vars while keeping defaults.
IMAGE_PATH="${1:-${ULTRASHAPE_IMAGE:-inputs/image/demo.png}}"
MESH_PATH="${2:-${ULTRASHAPE_MESH:-inputs/coarse_mesh/res.glb}}"
CONFIG_PATH="${ULTRASHAPE_CONFIG:-configs/infer_dit_refine.yaml}"
CKPT_PATH="${ULTRASHAPE_CKPT:-checkpoints/ultrashape_v1.pt}"
OUTPUT_DIR="${ULTRASHAPE_OUTPUT_DIR:-}"

# sampling
# "$PYTHON" scripts/sampling.py \
#     --mesh_json data/mesh_paths.json \
#     --output_dir data/sample

# inference refine_dit
set -- scripts/infer_dit_refine.py \
    --ckpt "$CKPT_PATH" \
    --image "$IMAGE_PATH" \
    --mesh "$MESH_PATH" \
    --config "$CONFIG_PATH" \
    --octree_res "$ULTRASHAPE_OCTREE_RES" \
    --steps "$ULTRASHAPE_STEPS" \
    --chunk_size "$ULTRASHAPE_CHUNK_SIZE" \
    --scale "$ULTRASHAPE_SCALE" \
    --seed "$ULTRASHAPE_SEED"

if [ -n "$OUTPUT_DIR" ]; then
  set -- "$@" --output_dir "$OUTPUT_DIR"
fi

if [ -n "$ULTRASHAPE_NUM_LATENTS" ]; then
  set -- "$@" --num_latents "$ULTRASHAPE_NUM_LATENTS"
fi

if [ "$ULTRASHAPE_REMOVE_BG" = "1" ]; then
  set -- "$@" --remove_bg
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$@"
