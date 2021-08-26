#!/usr/bin/env bash
set -x

WORKSPACE=/home/$USER/FedGraphNN
cd $WORKSPACE/experiments/distributed/ego_networks

sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 1 20 1 0.01 64 5 0.1 ciao
