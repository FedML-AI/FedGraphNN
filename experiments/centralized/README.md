# Centralized Training


## Run Experiments

## Experiments
#### Centralized Molecule Property Classification experiments
```
python experiments/centralized/moleculenet/molecule_classification_multilabel.py
```

#### Centralized Molecule Property Regression experiments
```
python experiments/centralized/moleculenet/molecule_regression_multivariate.py
```

### Arguments
This is a list of arguments used in centralized experiments. 
```
--dataset --> Dataset used for training
--data_dir --> Data directory
=--normalize_features --> Whether or not to symmetrically normalize feat matrices
--normalize_adjacency --> Whether or not to symmetrically normalize adj matrices
--sparse_adjacency --> Whether or not the adj matrix is to be processed as a sparse matrix
--model --> Model name. Currently supports SAGE, GAT and GCN
--hidden_size -->Size of GNN hidden layer
--node_embedding_dim --> Dimensionality of the vector space the atoms will be embedded in
--alpha --> Alpha value for LeakyRelu used in GAT
--num_heads --> Number of attention heads used in GAT
--dropout --> Dropout used between GraphSAGE layers
--readout_hidden_dim --> Size of the readout hidden layer
--graph_embedding_dim --> Dimensionality of the vector space the molecule will be embedded in
--client_optimizer -> Optimizer function(Adam or SGD)
--lr --> learning rate (default: 0.0015)
--wd --> Weight decay(default=0.001)
--epochs -->Number of epochs
--frequency_of_the_test --> How frequently to run eval
--device --> gpu device for training
--metric --> Metric to be used to evaluate classification models

```
