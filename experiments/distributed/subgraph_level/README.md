# Node-Level Tasks (Ego Networks)


## Data Preparation


For each dataset, ego-networks needs to be sampled first.  
```
WORKSPACE=/home/$USER/FedGraphNN
# WORKSPACE=/Users/chaoyanghe/sourcecode/FedGraphNN
cd $WORKSPACE/data_preprocessing/ego_networks

mkdir $WORKSPACE/data/ego-networks/
mkdir $WORKSPACE/data/ego-networks/cora
mkdir $WORKSPACE/data/ego-networks/citeseer
mkdir $WORKSPACE/data/ego-networks/DBLP
mkdir $WORKSPACE/data/ego-networks/PubMed
mkdir $WORKSPACE/data/ego-networks/CS
mkdir $WORKSPACE/data/ego-networks/Physics

python sampleEgonetworks.py --path ./../../data/ego-networks/ --data cora --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data citeseer --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data DBLP --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data PubMed --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data CS --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data Physics --ego_number 1000 --hop_number 2
```

#### Arguments for Data Preparation code
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
--path -> the path for loading dataset

--data -> the name of dataset: "CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"

--type_network -> 'the type of dataset': citation", "coauthor"

--ego_number --> 'the number of egos sampled'

--hop_number --> 'the number of hops'
```

#### Datasets to Preprocess

citation networks (# nodes): e.g. DBLP (17716), Cora (2708), CiteSeer (3327), PubMed (19717)

collaboration networks (# nodes): e.g. CS (18333), Physics (34493)
 
 social networks (# ego-networks): e.g. COLLAB, IMDB, DEEZER_EGO_NETS (9629), TWITCH_EGOS (127094)


## Experiments 

### Distributed/Federated Node Classification experiments
```
WORKSPACE=/home/$USER/FedGraphNN
cd $WORKSPACE/experiments/distributed/ego_networks
sh run_fed_node_clf.sh 4 4 1 1 gcn hetero 0.5 20 1 32 0.0015 32 3 0.3 CS
```

### Distributed/Federated Link Prediction experiments
```
WORKSPACE=/home/$USER/FedGraphNN
cd $WORKSPACE/experiments/distributed/ego_networks
sh run_fed_link_pred.sh 4 4 1 1 gcn hetero 0.5 20 1 32 0.0015 32 3 0.3 CS
```

#### Arguments for Distributed/Federated Training
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
CLIENT_NUM -> Number of clients in dist/fed setting
WORKER_NUM-> Number of workers
SERVER_NUM -> Number of servers
GPU_NUM_PER_SERVER -> GPU number per server
MODEL -> Model name
DISTRIBUTION -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
PARTITION_ALPHA -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
ROUND -> Number of Distiributed/Federated Learning Rounds
EPOCH -> Number of epochs to train clients' local models
BATCH_SIZE -> Batch size 
LR  -> learning rate
DIM -> Dimenionality of GraphSAGE embedding
N_LAYER -> Dimensionality of node embeddings
SAGE_DR -> Dropout rate applied between GraphSAGE Layers
DATASET -> Dataset name (Please check data folder to see all available datasets)
FED_OPT -> Name for federated optimizer to use
```
