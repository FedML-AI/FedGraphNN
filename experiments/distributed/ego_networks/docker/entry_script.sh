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

# WORKSPACE=/Users/chaoyanghe/sourcecode/FedGraphNN

cd $WORKSPACE/data_preprocessing/ego_networks

#mkdir $WORKSPACE/data/ego-networks/
#mkdir $WORKSPACE/data/ego-networks/cora
#mkdir $WORKSPACE/data/ego-networks/citeseer
#mkdir $WORKSPACE/data/ego-networks/DBLP
#mkdir $WORKSPACE/data/ego-networks/PubMed
#mkdir $WORKSPACE/data/ego-networks/CS
#mkdir $WORKSPACE/data/ego-networks/Physics

# FL client number = 10, ego number = 1000
#python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data cora --ego_number 1000 --hop_number 2
#python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data citeseer --ego_number 1000 --hop_number 2
#python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data DBLP --ego_number 1000 --hop_number 2
#python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data PubMed --ego_number 1000 --hop_number 2

cd $WORKSPACE/data/ego_networks/cora
wget --no-check-certificate --no-proxy https://fedmol.s3.us-west-1.amazonaws.com/datasets/ego-networks/cora/egonetworks.pkl
cd $WORKSPACE/data/ego_networks/cora

md5sum ./../../data/ego-networks/cora/egonetworks.pkl
md5sum ./../../data/ego-networks/citeseer/egonetworks.pkl
md5sum ./../../data/ego-networks/DBLP/egonetworks.pkl
md5sum ./../../data/ego-networks/PubMed/egonetworks.pkl


cd $WORKSPACE/experiments/distributed/ego_networks

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb on
nohup python3 sweep_ego_sage.py --starting_run_id 0 > sweeping.log 2>&1 &

tail -f sweeping.log

# sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 cora
