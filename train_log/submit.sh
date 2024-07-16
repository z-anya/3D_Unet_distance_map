#!/bin/bash
#DSUB -n lzg_test
#DSUB -A root.dalhxwlyjsuo
#DSUB -l hws_gnode
#DSUB --job_type cosched
#DSUB -R 'cpu=48;gpu=8;mem=360000'
#DSUB -N 1
#DSUB -oo log.out
#DSUB -eo error.out
source /share/ccsuite/ENV/setenvanacond3.sh
source activate lzg_torch_edit

#export PYTHONUNBUFFERED=1
#python 运行程序


# 创建状态文件，用于控制采集的进程
STATE_FILE="state_${BATCH_JOB_ID}"
/usr/bin/touch ${STATE_FILE}
# 后台循环采集，每间隔 1s 采集一次 GPU 数据。
# 采集的数据将输出到本地 gpu_作业 ID.log 文件中
function gpus_collection(){
while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
/usr/bin/sleep 1
#/usr/bin/nvidia-smi >> "gpu${BATCH_JOB_ID}.log"
/usr/bin/nvidia-smi >> "gpu.log"
done
}
gpus_collection &
gpus=8
micro_batch_size_per_gpu=2
gradient_accumulation_steps=4
batch_size=$((gpus * micro_batch_size_per_gpu * gradient_accumulation_steps))
deepspeed --num_gpus=$gpus \
          /home/dalhxwlyjsuo/guest_lizg/frame_is_word/lvm/load_example.py \
          --world_size=$gpus \
         --micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
         --gradient_accumulation_steps=$gradient_accumulation_steps \
         --batch_size=$batch_size \
         --epochs=10 \
         --warmup_steps=5 \
         --lr=1e-5 \
         --numworkers=48 \
         --ds_stage=1 \
         --placeholder=1 \

# 关闭 GPU 采集进程
echo "over" >> "${STATE_FILE}"
