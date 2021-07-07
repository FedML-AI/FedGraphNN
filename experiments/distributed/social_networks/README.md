## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login wandb_id
```

## Data Distribution
#### Classification
SIDER: client number = 4; LDA alpha = 0.2

CLINTOX: client number = 4; LDA alpha = 0.5 (doing)
```
python sweep_cls.py --starting_run_id 0 --model_name graphsage --dataset clintox --gpu 4 --alpha 0.5
```

BBBP: client number = 4; LDA alpha = 2

BACE: client number = 4; LDA alpha = 0.5 (doing)
```
python sweep_cls.py --starting_run_id 0 --model_name graphsage --dataset bace --gpu 4 --alpha 0.5
```


TOX21 (large-scale): client number = 8; LDA alpha = 3
```
python sweep_cls.py --starting_run_id 0 --model_name graphsage --dataset tox21 --gpu 8 --alpha 3
```

PBCA (large-scale): client number = 8; LDA alpha = 3
```
python sweep_cls.py --starting_run_id 0 --model_name graphsage --dataset pbca --gpu 8 --alpha 3
```

#### Regression
ESOL: client number = 4; LDA alpha = 2
```
python sweep_reg.py --starting_run_id 0 --model_name graphsage --dataset esol --gpu 4 --alpha 2
```

FREESOLV: client number = 4; LDA alpha = 0.5
```
python sweep_reg.py --starting_run_id 0 --model_name graphsage --dataset freesolv --gpu 4 --alpha 0.5
```

LIPO: client number = 8; LDA alpha = 2
```
python sweep_reg.py --starting_run_id 0 --model_name graphsage --dataset clintox --gpu 8 --alpha 2
```


QM9 (large-scale): client number = 8; LDA alpha = 3 
```
python sweep_reg.py --starting_run_id 0 --model_name graphsage --dataset qm9 --gpu 8 --alpha 3
```

HERG (large-scale): client number = 8; LDA alpha = 3
```
python sweep_reg.py --starting_run_id 0 --model_name graphsage --dataset herg --gpu 8 --alpha 3
```

## Run Experiments

### Molecule Property Classification experiments
```
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 graphsage hetero 0.2 20 1 1 0.0015 64 32 0.3 64 64 sider "./../../../data/sider/" 0

sh run_fedavg_distributed_pytorch.sh 8 8 1 8 graphsage hetero 0.2 20 1 1 0.0015 64 32 0.3 64 64 tox21 "./../../../data/tox21/" 0

## run on background
nohup sh run_fedavg_distributed_pytorch.sh 4 4 1 4 graphsage hetero 0.2 20 1 1 0.0015 64 32 0.3 64 64 sider "./../../../data/sider/" 0 > ./fedavg-graphsage.log 2>&1 &

## run sweep
python sweep_cls.py --starting_run_id 0
```


### Molecule Property Regression experiments
```
sh run_fedavg_distributed_reg.sh 4 4 1 4 graphsage hetero 0.5 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0

##run on background
nohup sh run_fedavg_distributed_reg.sh 4 4 1 4 graphsage hetero 0.5 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0 > ./fedavg-graphsage.log 2>&1 &
```

### Arguments
This is an ordered list of arguments used in distributed/federated experiments. 
```
CLIENT_NUM=$1 -> Number of clients in dist/fed setting
WORKER_NUM=$2 -> Number of workers
SERVER_NUM=$3 -> Number of servers
GPU_NUM_PER_SERVER=$4 -> GPU number per server
MODEL=$5 -> Model name
DISTRIBUTION=$6 -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
ROUND=$7 -> Number of Distiributed/Federated Learning Rounds
EPOCH=$8 -> Number of epochs to train clients' local models
BATCH_SIZE=$9 -> Batch size 
LR=${10}  -> learning rate
SAGE_DIM=${11} -> Dimenionality of GraphSAGE embedding
NODE_DIM=${12} -> Dimensionality of node embeddings
SAGE_DR=${13} -> Dropout rate applied between GraphSAGE Layers
READ_DIM=${14} -> Dimensioanlity of readout embedding
GRAPH_DIM=${15} -> Dimensionality of graph embedding
DATASET=${16} -> Dataset name (Please check data folder to see all available datasets)
DATA_DIR=${17} -> Dataset directory
CI=${18}
```

## Experimental Results
GraphSage + FedAvg + SIDER: 0.67 (Test/ROC-AUC)

```
graph_embedding_dim 64
hidden_size 64
node_embedding_dim 16
dropout 0.6
lr 0.0015
batch_size 1
comm_round 20
epochs 1
```

GraphSage + FedAvg + BBBP: 0.8935 (Test/ROC-AUC)

```
graph_embedding_dim 64
hidden_size 64
node_embedding_dim 64
dropout 0.6
lr 0.015
batch_size 1
comm_round 50
epochs 1
```
