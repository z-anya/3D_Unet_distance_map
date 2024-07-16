#!/bin/bash
#source  /home/dalhxwlyjsuo/username/guest_lizg/etc/profile.d/conda.sh
#source /share/ccsuite/ENV/setenvanacond3.sh
#source activate 3d_unet


export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export PYTHONUNBUFFERED=1
# 重要配置，运行的节点角色。
NODE_RANK="${1}"
# 运行的节点数量。
NODES="${2}"
# 每个节点运行的进程数量。通常根据申请资源而定，例如申请的资源有 8 个GPU，则写 8 个进程，每个进程使用 1 个GPU
NPROC_PER_NODE=7

# 主节点的地址和端口。
MASTER_ADDR="${3}"
MASTER_PORT="29501"

# 计算batch_size
#OUTPUT_LOG=train_rank"${NODE_RANK}"_"${BATCH_JOB_ID}".log
OUTPUT_LOG=test_train_rank"${NODE_RANK}".log
gpus=$((NODES * NPROC_PER_NODE))
micro_batch_size_per_gpu=1
gradient_accumulation_steps=1
train_batch_size=$((gpus * micro_batch_size_per_gpu * gradient_accumulation_steps))
echo $micro_batch_size_per_gpu
echo $train_batch_size
torchrun \
     --nnodes="${NODES}" \
     --node_rank="${NODE_RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
     --max_restarts=3 \
     /home/dalhxwlyjsuo/guest_lizg/unet/train_UnitedNet.py \
     --world_size=$gpus \
     --micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
     --gradient_accumulation_steps=$gradient_accumulation_steps \
     --batch_size=$train_batch_size  >> "${OUTPUT_LOG}" 2>&1
