import argparse
from molecule_classification_multilabel import train_model
import pickle

import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # PipeTransformer related
    parser.add_argument("--starting_run_id", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="sider", help="Dataset used for training"
    )

    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")

    parser.add_argument(
        "--normalize_features",
        type=bool,
        default=False,
        help="Whether or not to symmetrically normalize feat matrices",
    )

    parser.add_argument(
        "--normalize_adjacency",
        type=bool,
        default=False,
        help="Whether or not to symmetrically normalize adj matrices",
    )

    parser.add_argument(
        "--sparse_adjacency",
        type=bool,
        default=False,
        help="Whether or not the adj matrix is to be processed as a sparse matrix",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="graphsage",
        help="Model name. Currently supports SAGE, GAT and GCN.",
    )

    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Size of GNN hidden layer"
    )

    parser.add_argument(
        "--node_embedding_dim",
        type=int,
        default=32,
        help="Dimensionality of the vector space the atoms will be embedded in",
    )

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha value for LeakyRelu used in GAT"
    )

    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of attention heads used in GAT"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used between GraphSAGE layers",
    )

    parser.add_argument(
        "--readout_hidden_dim",
        type=int,
        default=64,
        help="Size of the readout hidden layer",
    )

    parser.add_argument(
        "--graph_embedding_dim",
        type=int,
        default=64,
        help="Dimensionality of the vector space the molecule will be embedded in",
    )

    parser.add_argument(
        "--client_optimizer",
        type=str,
        default="adam",
        metavar="O",
        help="SGD with momentum; adam",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0015,
        metavar="LR",
        help="learning rate (default: 0.0015)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", metavar="WD", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=5,
        help="How frequently to run eval",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        metavar="DV",
        help="gpu device for training",
    )

    return parser.parse_args()


parser = argparse.ArgumentParser()
args = add_args(parser)

epoch_hop = [20, 25, 30]
lr_hpo = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
node_dim_hpo = [16, 32, 64]
hidden_dim_hpo = [16, 32, 64]
read_dim_hpo = [16, 32, 64]
graph_dim_hpo = [16, 32, 64]
dropout_hpo = [0.5]

max_test_score = 0
optimal_hyperparam_dict = {
    "epochs": -1,
    "lr": -1,
    "node_embedding_dim": -1,
    "hidden_size": -1,
    "readout_hidden_dim": -1,
    "graph_embedding_dim": -1,
    "dropout": -1,
}

print("Starting Sweep for {}".format(args.dataset))

for epoch in epoch_hop:
    for lr in lr_hpo:
        for node_dim in node_dim_hpo:
            for hidden_dim in hidden_dim_hpo:
                for read_dim in read_dim_hpo:
                    for graph_dim in graph_dim_hpo:
                        for dr in dropout_hpo:
                            args.epoch = epoch
                            args.lr = lr
                            args.node_embedding_dim = node_dim
                            args.hidden_size = hidden_dim
                            args.readout_hidden_dim = read_dim
                            args.graph_embedding_dim = graph_dim
                            args.dropout = dr

                            test_score = train_model(args)

                            if test_score[0] > max_test_score:
                                print("New max text score = {}".format(test_score[0]))
                                print("Saving newly found optimal hps")

                                max_test_score = test_score[0]
                                optimal_hyperparam_dict["epochs"] = epoch
                                optimal_hyperparam_dict["lr"] = lr
                                optimal_hyperparam_dict["node_embedding_dim"] = node_dim
                                optimal_hyperparam_dict["hidden_size"] = hidden_dim
                                optimal_hyperparam_dict["readout_hidden_dim"] = read_dim
                                optimal_hyperparam_dict[
                                    "graph_embedding_dim"
                                ] = graph_dim
                                optimal_hyperparam_dict["dropout"] = dr


print("Best test score = {}".format(max_test_score))
with open(args.dataset + "_best_hps.pkl", "wb") as handle:
    pickle.dump(optimal_hyperparam_dict, handle)
