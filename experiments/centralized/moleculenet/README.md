## Data Preparation

## Experiments 


### Centralized Molecule Property Classification experiments
```
python experiments/centralized/moleculenet/molecule_classification_multilabel.py
```

### Centralized Molecule Property Regression experiments
```
python experiments/centralized/moleculenet/molecule_regression_multivariate.py
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

