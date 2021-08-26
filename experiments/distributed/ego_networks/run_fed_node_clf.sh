#!/usr/bin/env bash

# sh run_fed_node_clf.sh 10 10 1 1 gcn hetero 2.0 20 1 32 0.0015 32 3 0.3 0.0001 cora
CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
PARTITION_ALPHA=$7 # 2.0
ROUND=$8 # 20
EPOCH=$9 # 1
BATCH_SIZE=$10 # 32
LR=${11} # 0.0015
HIDDEN_DIM=${12} # 32
N_LAYERS=${13} # 3
DR=${14} # 0.3
WD=${15} # 0.3
DATASET=${16}


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 fed_node_clf.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --hidden_size $HIDDEN_DIM \
  --dropout $DR \
  --partition_method $DISTRIBUTION  \
  --partition_alpha $PARTITION_ALPHA \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --n_layers $N_LAYERS \
  --lr $LR \
  --wd $WD
