export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export num_gpu_per_node=${NUM_GPU_PER_NODE:-8}

export node_num=${NODE_NUM:-1}
model_type=${1:-dit}
node_rank=${2:-0}
master_ip=${3:-${MASTER_IP:-127.0.0.1}}
extra_args=("${@:4}")

case "$model_type" in
    dit)
        config=configs/train_dit_refine.yaml
        output_dir=outputs/dit_ultrashape/exp1_token8192
        update_every=4
        ;;
    vae)
        config=configs/train_vae_refine.yaml
        output_dir=outputs/vae_ultrashape/exp1_token8192
        update_every=1
        ;;
    *)
        echo "Usage: $0 [dit|vae] [node_rank] [master_ip] [extra args...]"
        exit 1
        ;;
esac

bash scripts/train_deepspeed.sh \
    $node_num \
    $node_rank \
    $num_gpu_per_node \
    $master_ip \
    $config \
    $output_dir \
    --fast \
    --update_every $update_every \
    "${extra_args[@]}"
