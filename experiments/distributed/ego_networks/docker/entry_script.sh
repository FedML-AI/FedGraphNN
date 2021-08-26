#!/usr/bin/env bash
set -x

WORKSPACE=/home/chaoyanghe/FedGraphNN
cd $WORKSPACE/experiments/distributed/ego_networks

sh run_fed_node_clf.sh 10 10 1 8 sage hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 PubMed
