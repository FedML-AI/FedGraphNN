## Data Preparation

## Experiments 
To use datasets,first run  ```bash download_and_unzip.sh```  located under each dataset folder in  ```data/moleculenet```

### Distributed/Federated Molecule Property Classification experiments
```
sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0 > ./fedavg-graphsage.log 2>&1 &
```

### Distributed/Federated Molecule Property Regression experiments
```
sh run_fedavg_distributed_reg.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0

##run on background
nohup sh run_fedavg_distributed_reg.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0 > ./fedavg-graphsage.log 2>&1 &
```

#### Arguments for Distributed/Federated Training
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
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
