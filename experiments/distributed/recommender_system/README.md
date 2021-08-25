

## Run Experiments
```
# optimal hyper-parameters are obtained by the sweeping scripts (sweep_rs.py)

# dataset: ciao; model: gcn
sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: ciao; model: gat
sh run_fed_subgraph_link_pred.sh 28 28 1 8 gat uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: ciao; model: sage
sh run_fed_subgraph_link_pred.sh 28 28 1 8 sage uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: epinions; model: gcn
sh run_fed_subgraph_link_pred.sh 27 27 1 8 gcn uniform 0.1 2 2 1 0.01 64 5 0.1 epinions

# dataset: epinions; model: gat
sh run_fed_subgraph_link_pred.sh 27 27 1 8 gat uniform 0.1 2 2 1 0.01 64 5 0.1 epinions

# dataset: epinions; model: sage
sh run_fed_subgraph_link_pred.sh 27 27 1 8 sage uniform 0.1 2 2 1 0.01 64 5 0.1 epinions
```

## Sweep Hyper-parameters
```
wandb on
python3 sweep_rs.py --starting_run_id 0
```


# Run Experiments on Docker

### Run on a single node (GPU server with single/multiple GPUs)


### Run on multiple nodes