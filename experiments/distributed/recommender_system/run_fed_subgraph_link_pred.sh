#!/usr/bin/env bash


# sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 10 20 1 0.01 64 5 0.1 ciao

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
PARTITION_ALPHA=$7  # 0.1
ROUND=$8 # 1
EPOCH=$9 # 20
BATCH_SIZE=$10 # 1
LR=${11} # 0.01
HIDDEN_DIM=${12} # 64
N_LAYERS=${13} # 5
DR=${14} # 0.1
DATASET=${15} # ciao


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 fed_subgraph_link_pred.py \
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
  --n_layers $N_LAYERS \
  --batch_size $BATCH_SIZE \
  --lr $LR