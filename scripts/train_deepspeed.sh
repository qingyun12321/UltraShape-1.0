
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0

detect_iface() {
    local iface=""
    iface="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -n1)"
    if [[ -z "$iface" ]]; then
        iface="$(ip -o -4 addr show up 2>/dev/null | awk -F': ' '/state UP/ {print $2; exit}')"
    fi
    if [[ -z "$iface" ]]; then
        iface="eth0"
    fi
    echo "$iface"
}

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME="$(detect_iface)"
fi

if [[ -d /sys/class/infiniband ]] && [[ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
    export NCCL_IB_DISABLE=0
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
    export NCCL_IB_SL=${NCCL_IB_SL:-3}
    export NCCL_IB_CUDA_SUPPORT=${NCCL_IB_CUDA_SUPPORT:-1}
    export NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD:-16384}
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
    export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
    export NCCL_IB_TC=${NCCL_IB_TC:-160}
    export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
else
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
    export NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD:-16384}
fi

export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
# export NCCL_DEBUG=INFO

node_num=$1
node_rank=$2
num_gpu_per_node=$3
master_ip=$4
config=$5
output_dir=$6
extra_args=("${@:7}")

echo node_num $node_num
echo node_rank $node_rank
echo master_ip $master_ip
echo config $config
echo output_dir $output_dir

export MASTER_ADDR=${MASTER_ADDR:-$master_ip}
export MASTER_PORT=${MASTER_PORT:-12348}
echo "[train_deepspeed] MASTER_ADDR=${MASTER_ADDR}"
echo "[train_deepspeed] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

# Prefer venv python when available.
python_exec="python3"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    python_exec="${UV_PROJECT_ENVIRONMENT}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    python_exec="${VIRTUAL_ENV}/bin/python"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -x "${script_dir}/../.venv/bin/python" ]]; then
        python_exec="${script_dir}/../.venv/bin/python"
    fi
fi
echo "[train_deepspeed] PYTHON_EXEC=${python_exec}"

if test -d "$output_dir"; then
    cp $config $output_dir
else
    mkdir -p "$output_dir"
    cp $config $output_dir
fi

NODE_RANK=$node_rank \
HF_HUB_OFFLINE=0 \
$python_exec main.py \
    --num_nodes $node_num \
    --num_gpus $num_gpu_per_node \
    --config $config \
    --output_dir $output_dir \
    --deepspeed \
    "${extra_args[@]}"
