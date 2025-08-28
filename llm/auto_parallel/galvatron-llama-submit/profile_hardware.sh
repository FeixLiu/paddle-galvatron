nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_WORKERS_IP_PORT_LIST
unset PADDLE_TRAINERS
unset PADDLE_NUM_GRADIENT_SERVERS

START_RANK=0
END_RANK=8

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi
export rank=$(($rank-$START_RANK))
export nnodes=$(($END_RANK-$START_RANK))
master_ip=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
export master=$master_ip
export port=36677

export interpreter="<path to your own python>"

rm -rf log
rm -rf output

bash scripts/profile_hardware.sh
