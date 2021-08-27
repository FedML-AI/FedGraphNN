#!/usr/bin/env bash
set -x

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

USER=fedgraphnn
HOME=/home/$USER
sudo chmod -R 777 ${HOME}/.config
sudo chown -R ${USER}:${USER} ${HOME}

wandb login "ee0b5f53d949c84cee7decbe7a629e63fb2f8408"
wandb on

WORKSPACE=/fsx/hchaoyan/home/FedGraphNN
sudo chown -R ${USER}:${USER} ${WORKSPACE}


cd $WORKSPACE/experiments/distributed/recommender_system

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb on

sh run_fed_subgraph_link_pred.sh 1 1 1 8 gat uniform 0.1 100 1 1 0.001 64 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 gat uniform 0.1 100 1 1 0.01 64 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 gat uniform 0.1 100 1 1 0.01 32 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 gat uniform 0.1 100 1 1 0.01 128 5 0.1 epinion
#
#
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 sage uniform 0.1 100 1 1 0.001 64 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 sage uniform 0.1 100 1 1 0.01 64 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 sage uniform 0.1 100 1 1 0.01 32 5 0.1 epinion
#sh run_fed_subgraph_link_pred.sh 1 1 1 8 sage uniform 0.1 100 1 1 0.01 128 5 0.1 epinion

# sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 cora
