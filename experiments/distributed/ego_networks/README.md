# Node-Level Tasks (Ego Networks)


## Data Preparation


For each dataset, ego-networks needs to be sampled first.  

citation networks: e.g. "cora", "citeseer", "DBLP", "PubMed"
```
WORKSPACE=/fsx/hchaoyan/home/FedGraphNN
# WORKSPACE=/Users/chaoyanghe/sourcecode/FedGraphNN
cd $WORKSPACE/data_preprocessing/ego_networks

mkdir $WORKSPACE/data/ego-networks/
mkdir $WORKSPACE/data/ego-networks/cora
mkdir $WORKSPACE/data/ego-networks/citeseer
mkdir $WORKSPACE/data/ego-networks/DBLP
mkdir $WORKSPACE/data/ego-networks/PubMed

# FL client number = 10, ego number = 1000
python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data cora --ego_number 1000 --hop_number 2 &&
python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data citeseer --ego_number 1000 --hop_number 2 &&
python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data DBLP --ego_number 1000 --hop_number 2 && 
python3 sampleEgonetworks.py --path ./../../data/ego-networks/ --data PubMed --ego_number 1000 --hop_number 2

# FL client number = 10, ego number = 10
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data CS --ego_number 10 --hop_number 3
python sampleEgonetworks.py --path ./../../data/ego-networks/ --data Physics --ego_number 10 --hop_number 3
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

## Experiments 

### Distributed/Federated Node Classification experiments
```
WORKSPACE=/home/$USER/FedGraphNN
cd $WORKSPACE/experiments/distributed/ego_networks

# for citation network (cora, citeseer, pubmed, dblp), we allow each FL client to have heterogenous number of ego networks.
# gcn model
sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 cora
sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 citeseer
sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 DBLP
sh run_fed_node_clf.sh 10 10 1 8 gcn hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 PubMed

# sgc model
sh run_fed_node_clf.sh 10 10 1 8 sgc hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 cora
sh run_fed_node_clf.sh 10 10 1 8 sgc hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 citeseer
sh run_fed_node_clf.sh 10 10 1 8 sgc hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 DBLP
sh run_fed_node_clf.sh 10 10 1 8 sgc hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 PubMed

# sage
sh run_fed_node_clf.sh 10 10 1 8 sage hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 cora
sh run_fed_node_clf.sh 10 10 1 8 sage hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 citeseer
sh run_fed_node_clf.sh 10 10 1 8 sage hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 DBLP
sh run_fed_node_clf.sh 10 10 1 8 sage hetero 2.0 20 1 1 0.0015 1 3 0.3 0.0001 PubMed

```


### Sweeping
```change wandb id to your own
export LC_ALL=C.UTF-8 &&
export LANG=C.UTF-8 &&
WORKSPACE=/fsx/hchaoyan/home/FedGraphNN &&
cd $WORKSPACE/experiments/distributed/ego_networks/ &&
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408 &&
wandb on
nohup python3 sweep_ego_sage_DBLP_0.1.py --starting_run_id 0 > sweeping.log 2>&1 & (running 0)
nohup python3 sweep_ego_sage_DBLP_10.0.py --starting_run_id 0 > sweeping.log 2>&1 & (running 1)

nohup python3 sweep_ego_sage_PubMed_0.1.py --starting_run_id 0 > sweeping.log 2>&1 & (running 2)
nohup python3 sweep_ego_sage_PubMed_10.0.py --starting_run_id 0 > sweeping.log 2>&1 & (running 3)

nohup python3 sweep_ego_sage_cora_0.1.py --starting_run_id 0 > sweeping.log 2>&1 & (running 4)
nohup python3 sweep_ego_sage_cora_10.0.py --starting_run_id 0 > sweeping.log 2>&1 & (running 5)

nohup python3 sweep_ego_sage_citeseer_0.1.py --starting_run_id 0 > sweeping.log 2>&1 & (running 6)
nohup python3 sweep_ego_sage_citeseer_10.0.py --starting_run_id 0 > sweeping.log 2>&1 & (running 7)
```