

## Run Experiments (Client Number = 28/27, each client has only one category)
```
# optimal hyper-parameters are obtained by the sweeping scripts (sweep_rs.py)

# dataset: ciao; model: gcn; client_num = 28
sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: ciao; model: gat; client_num = 28
sh run_fed_subgraph_link_pred.sh 28 28 1 8 gat uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: ciao; model: sage; client_num = 28
sh run_fed_subgraph_link_pred.sh 28 28 1 8 sage uniform 0.1 2 2 1 0.01 64 5 0.1 ciao

# dataset: epinions; model: gcn; client_num = 27
sh run_fed_subgraph_link_pred.sh 27 27 1 8 gcn uniform 0.1 2 2 1 0.01 64 5 0.1 epinions

# dataset: epinions; model: gat; client_num = 27
sh run_fed_subgraph_link_pred.sh 27 27 1 8 gat uniform 0.1 2 2 1 0.01 64 5 0.1 epinions

# dataset: epinions; model: sage; client_num = 27
sh run_fed_subgraph_link_pred.sh 27 27 1 8 sage uniform 0.1 2 2 1 0.01 64 5 0.1 epinions
```

## Run Experiments (Client Number = 8, each client has multiple categories)
```
# optimal hyper-parameters are obtained by the sweeping scripts (sweep_rs.py)

# dataset: ciao; model: gcn; client_num = 8
sh run_fed_subgraph_link_pred.sh 8 8 1 8 gcn uniform 0.1 100 5 1 0.01 32 5 0.1 ciao

# dataset: ciao; model: gat; client_num = 8
sh run_fed_subgraph_link_pred.sh 8 8 1 8 gat uniform 0.1 100 2 1 0.001 32 5 0.1 ciao

# dataset: ciao; model: sage; client_num = 8
sh run_fed_subgraph_link_pred.sh 8 8 1 8 sage uniform 0.1 50 5 1 0.01 32 5 0.1 ciao

# dataset: epinions; model: gcn; client_num = 8
sh run_fed_subgraph_link_pred.sh 8 8 1 8 gcn uniform 0.1 100 2 1 0.01 64 5 0.1 epinions

# dataset: epinions; model: gat; client_num = 8
sh run_fed_subgraph_link_pred.sh 8 8 1 8 gat uniform 0.1 50 1 1 0.001 64 5 0.1 epinions

# dataset: epinions; model: sage; client_num = 8（***)
sh run_fed_subgraph_link_pred.sh 8 8 1 8 sage uniform 0.1 2 2 1 0.01 64 5 0.1 epinions
```

## Sweep Hyper-parameters
```
# change id to your own
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb on
python3 sweep_rs.py --starting_run_id 0
```


# Run Experiments on Docker

### Run on a single node (GPU server with single/multiple GPUs)


### Run on multiple nodes