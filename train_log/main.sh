#!/bin/bash
### 指定作业的运行名称
#DSUB -n multi_node_test
#DSUB -A root.dalhxwlyjsuo
### 默认配置，一般不需要修改
#DSUB -l hws_gnode

### 指定单个节点的资源配置
#DSUB -R 'cpu=30;gpu=7;mem=32000'

### 指定副本数量，可以理解为计算节点数量。
#DSUB -N 3
#DSUB -eo %J_error.out
#DSUB -oo %J_log.out

### 重要配置！节点数量。跟 -N 指定的副本数量相同。
NODE_NUM=3
### 重要配置！作业脚本名称。
SCRIPT="multi_node.sh"
HOSTFILE="hostfile_${BATCH_JOB_ID}"
STATE_FILE="state_${BATCH_JOB_ID}"
/usr/bin/touch ${STATE_FILE}

function init_node(){
    _LOCAL_HOST_=`hostname`
    echo "${_LOCAL_HOST_}" >> "${HOSTFILE}"
	if [[ $? != 0 ]]; then
		echo "Error: shell cmd error..."
		return 1
	fi

	while [[ `cat "${HOSTFILE}" | wc -l` < "${NODE_NUM}" ]]; do
		/usr/bin/sleep 1
		echo "[${_LOCAL_HOST_}] Wait for other tasks to initialize... "
	done
	return 0
}

function gpus_collection(){
	ROLE="${1}"
	while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
		/usr/bin/sleep 1
		/usr/bin/nvidia-smi >> "gpu_${ROLE}_${BATCH_JOB_ID}.log"
	done
}
function start_rank(){
	_LOCAL_HOST_=`hostname`
	PRE_ROLE='0'
	CURRENT_ROLE='0'

	if [[ -f "$HOSTFILE" ]]; then
		for line in `/usr/bin/cat $HOSTFILE`
		do
			let k=k+1
			host[$k]=$line
			if [[ "${_LOCAL_HOST_}" == "${line}" ]]; then
				local current_rank_id=`expr $k - 1`
				local pre_rank_id=`expr $k - 2`
				if [[ "${current_rank_id}" == "0" ]]; then
					pre_rank_id=${current_rank_id}
				fi
				PRE_ROLE="${pre_rank_id}"
				CURRENT_ROLE="${current_rank_id}"
			fi
		done
	else
		echo "Error: ${HOSTFILE} not found ..."
		return 1
	fi

	if [[ "${CURRENT_ROLE}" != "0" ]]; then
		while [[ `cat "${STATE_FILE}" | grep "${PRE_ROLE}" | grep "true" | wc -l` == "0" ]]; do
			/usr/bin/sleep 1
			echo "[rank${CURRENT_ROLE}] Wait for rank${PRE_ROLE} to initialize..."
		done
	fi

	echo "${_LOCAL_HOST_} ${host[1]} ${CURRENT_ROLE} true"
	echo "${_LOCAL_HOST_} ${host[1]} ${CURRENT_ROLE} true" >> "${STATE_FILE}"

	# gpus_collection 为GPU数据采集进程。GPU的利用率、显存将输出到 gpu_rank1_作业id.log 的日志中
	gpus_collection "rank${CURRENT_ROLE}" &
	# 在此启动作业脚本
	NODES="${#host[@]}"
	/bin/bash "${SCRIPT}" "${CURRENT_ROLE}" "${NODES}" "${host[1]}"

	echo "${_LOCAL_HOST_} ${host[1]} ${CURRENT_ROLE} over"
	echo "${_LOCAL_HOST_} ${host[1]} ${CURRENT_ROLE} over" >> "${STATE_FILE}"
	return 0
}

function main(){
	init_node
	if [[ $? != 0 ]]; then
		return 1
	fi
	start_rank
}

main



