# FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks
A Research-oriented Federated Learning Library and Benchmark Platform for Graph Neural Networks. 


Datasets: http://moleculenet.ai/

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fedgraphnn python=3.7
conda activate fedgraphnn
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fedmolecule
conda install -c anaconda mpi4py grpcio
conda install scikit-learn numpy h5py setproctitle networkx
pip install -r requirements.txt 
cd FedML; git submodule init; git submodule update; cd ../;
pip install -r FedML/requirements.txt
```

## Data Preparation


## Experiments 


### Centralized Molecule Property Classification experiments
```
python experiments/centralized/sage_moleculenet/molecule_classification_multilabel.py
```

### Centralized Molecule Property Regression experiments
```
python experiments/centralized/sage_moleculenet/molecule_regression_multivariate.py
```

#### Arguments for Centralized Training
This is a list of arguments used in centralized experiments. 
```
--dataset --> Dataset used for training
--data_dir' --> Data directory
--partition_method -> how to partition the dataset
--sage_hidden_size' -->Size of GraphSAGE hidden layer
--node_embedding_dim --> Dimensionality of the vector space the atoms will be embedded in
--sage_dropout --> Dropout used between GraphSAGE layers
--readout_hidden_dim --> Size of the readout hidden layer
--graph_embedding_dim --> Dimensionality of the vector space the molecule will be embedded in
--client_optimizer -> Optimizer function(Adam or SGD)
--lr --> learning rate (default: 0.0015)
--wd --> Weight decay(default=0.001)
--epochs -->Number of epochs
--frequency_of_the_test --> How frequently to run eval
--device -->gpu device for training
```

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

## Code Structure of FedGraphNN
<!-- Note: The code of FedGraphNN only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

- `FedML`: A soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.

- `data`: Provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms. FedGraphNN supports more advanced datasets and models for federated training of graph neural networks.

- `data_preprocessing`: Domain-specific PyTorch Data loaders for centralized and distributed training. 

- `model`: GNN models.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
1. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
3. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.

- `experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/DDP_demo`.


# Update FedML Submodule
```
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
```


## Citation
Please cite our FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedML".

 
