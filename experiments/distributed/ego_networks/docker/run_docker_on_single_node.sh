#!/bin/bash

set -x

WORKSPACE=/home/chaoyanghe/FedGraphNN

sudo chmod 777 /var/run/docker.sock

echo "stop previous docker run..."
docker container kill $(docker ps -q)


echo "start new docker run"
nvidia-docker run -i -v $WORKSPACE:$WORKSPACE --shm-size=60g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env AWS_BATCH_JOB_NODE_INDEX=$index \
--env AWS_BATCH_JOB_NUM_NODES=$node_num_for_training \
--env AWS_BATCH_JOB_MAIN_NODE_INDEX=0 \
--env AWS_BATCH_JOB_ID=string \
--env AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$master_ip \
--env  \
M5_BATCH_BOOTSTRAP=$WORKSPACE/experiments/distributed/recommender_system/docker/bootstrap.sh \
--env \
M5_BATCH_ENTRY_SCRIPT=$WORKSPACE/experiments/distributed/recommender_system/docker/entry_script.sh \
-u fedgraphnn --net=host \
fedml/fedgraphnn:3.0