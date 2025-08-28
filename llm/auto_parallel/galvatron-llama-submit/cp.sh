#!/bin/bash

hostfile=${TRAIN_WORKSPACE}/hostfile
hostlist=$(cat $hostfile | awk '{print $1}' | xargs)
last_hostname=$(tail -n 1 $hostfile | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  if [ `hostname -i` == $host ]; then
    continue
  fi
  if [ $last_hostname == $host ]; then
    sleep 2
    scp $1 ${host}:${PWD}
  else
    nohup scp $1 ${host}:${PWD} &
  fi
done