# FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks
A Research-oriented Federated Learning Library and Benchmark Platform for Graph Neural Networks. 
Accepted to ICLR-DPML and MLSys21 - GNNSys'21 workshops. 

Datasets: http://moleculenet.ai/

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fedgraphnn python=3.8.3
conda activate fedgraphnn
sh install.sh
```

## Data Preparation

1. Graph - level 
      a. MoleculeNet [] []
      b. Social Networks [] []
2. Sub-graph Level
      a. Knowledge Graphs [] []
      b. Recommendation Systems [] []
3. Node-level
      a. Coauthor Networks [] []
      b. Citation Networks [] []

## Experiments 

1. Graph - level 
      a. MoleculeNet [centralized] [federated]
      b. Social Networks [] [federated]
2. Sub-graph Level
      a. Knowledge Graphs [] [federated]
      b. Recommendation Systems [] [federated]
3. Node-level
      a. Coauthor Networks [] [federated]
      b. Citation Networks [] [federated]

## Code Structure of FedGraphNN
<!-- Note: The code of FedGraphNN only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

- `FedML`: A soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.

- `data`: Provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms. FedGraphNN supports more advanced datasets and models for federated training of graph neural networks. Currently, we have molecular machine learning datasets. 

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
```
@misc{he2021fedgraphnn,
      title={FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks}, 
      author={Chaoyang He and Keshav Balasubramanian and Emir Ceyani and Yu Rong and Peilin Zhao and Junzhou Huang and Murali Annavaram and Salman Avestimehr},
      year={2021},
      eprint={2104.07145},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

 
