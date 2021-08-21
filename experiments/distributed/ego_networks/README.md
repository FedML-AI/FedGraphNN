# Node-Level Tasks (Ego Networks)


## Data Preparation

For each dataset, ego-networks needs to be sampled first.  
```
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data citeseer --type_network citation --ego_number 1000 --hop_number 2
```
#### Arguments for Data Preparation code
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
--path -> the path for loading dataset

--data -> the name of dataset ()

--type_network -> 'the type of dataset': ["citation", "coauthor", "social"]

--ego_number --> 'the number of egos sampled'

--hop_number --> 'the number of hops'
```


## Experiments 

### Distributed/Federated Node Classification experiments
```
sh run_fed_node_clf.sh 4 4 1 1 gcn hetero 0.5 20 1 32 0.0015 32 3 0.3 CS
```

### Distributed/Federated Link Prediction experiments
```
sh run_fed_link_pred.sh 4 4 1 1 gcn hetero 0.5 20 1 32 0.0015 32 3 0.3 CS

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
