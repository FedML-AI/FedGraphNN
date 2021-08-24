

## Run Experiments
```
# optimal hyper-parameters are obtained by the sweeping scripts (sweep_rs.py)
sh run_fed_subgraph_link_pred.sh 28 28 1 8 gcn uniform 0.1 1 20 1 0.01 64 5 0.1 ciao
```

## Sweep Hyper-parameters
```
python3 sweep_rs.py --starting_run_id 0
```


# Run Experiments on Docker

### Run on a single node (GPU server with single/multiple GPUs)


### Run on multiple nodes