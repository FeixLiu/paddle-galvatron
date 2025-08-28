launch="${interpreter} -u -m paddle.distributed.launch"
launch="${launch} --master $master:$port --nnodes $nnodes --rank $rank --gpus 0,1,2,3,4,5,6,7"

export INTERPRETER=${interpreter}
export LAUNCHER=${launch}
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
export NCCL_IB_DISABLE=0

PROFILE_HARDWARE_ARGS=(
    --num_nodes $nnodes
    --num_gpus_per_node 8
    --backend 'paddle'
    --max_pp_deg 8
    --max_tp_deg 8
)

${interpreter} profile_hardware.py \
    "${PROFILE_HARDWARE_ARGS[@]}"