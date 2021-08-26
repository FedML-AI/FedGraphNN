import argparse
import os
import random
import sys
import torch
import numpy as np

from ptflops import get_model_complexity_info

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from data_preprocessing.molecule.data_loader import get_dataloader, get_data
from model.sage_readout import *
from model.gat_readout import *
from model.gcn_readout import *
from scipy.sparse import rand


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument("--dataset", type=str, default="sider", help="Dataset")

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
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0015,
        metavar="LR",
        help="learning rate (default: 0.0015)",
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

    args = parser.parse_args()

    return args


datasets = [
    "freesolv",
    "qm9",
    "herg",
    "esol",
    "lipo",
    "sider",
    "tox21",
    "clintox",
    "bace",
    "bbbp",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    compact = args.model == "graphsage"

    if args.dataset not in datasets:
        raise Exception("no such dataset!")
    dataset_path = args.data_dir + "/" + args.dataset

    train_data, val_data, test_data = get_dataloader(
        dataset_path,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )
    adj_mats, feature_matrices, labels = get_data(dataset_path)

    avg_node = [m.shape[0] for m in feature_matrices]
    avg_node = sum(avg_node) / len(avg_node)
    print("Average number of nodes =", str(avg_node))
    avg_edge = [np.sum(m) for m in adj_mats]
    avg_edge = sum(avg_edge) / len(avg_edge)
    print("Average number of edges =", str(avg_edge))
    feat_dim = feature_matrices[0].shape[1]
    num_cats = labels[0].shape[0]
    print("feat_dim = %d" % feat_dim)
    del feature_matrices
    del labels
    del adj_mats
    xx = rand(
        int(avg_node),
        int(avg_node),
        density=avg_edge / (avg_node * avg_node),
        format="csr",
    )
    xx.data[:] = 1

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.device == "cuda:0") else "cpu"
    )

    if args.model == "graphsage":
        model = SageMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
        )
        model.to(device)
        f = [
            torch.randn(int(avg_node)),
            torch.randn((int(avg_node), int(feat_dim / 4))),
            torch.randn((int(avg_node) * 2, int(feat_dim / 4))),
        ]
        f = [
            level.to(device=device, dtype=torch.long, non_blocking=True) for level in f
        ]
        prepare_input = lambda res: {
            "feature_matrix": torch.randn(int(avg_node), feat_dim).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "forest": f,
        }

    elif args.model == "gat":
        model = GatMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.alpha,
            args.num_heads,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
        )
        prepare_input = lambda res: {
            "adj_matrix": torch.from_numpy(xx.toarray()).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "feature_matrix": torch.randn((int(avg_node), feat_dim)).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
        }
        model.to(device)
        flops, params = get_model_complexity_info(
            model,
            input_res=(1, 1, 1),
            input_constructor=prepare_input,
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True,
        )
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
        print("{:<30}  {:<8}".format("Computational complexity: ", flops))
    elif args.model == "gcn":
        model = GcnMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
            sparse_adj=args.sparse_adjacency,
        )
        model.to(device)
        prepare_input = lambda res: {
            "adj_matrix": torch.from_numpy(xx.toarray()).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "feature_matrix": torch.randn((int(avg_node), feat_dim)).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
        }
    else:
        raise Exception("No such model")
    # Since molecule data size is dynamic, we compute average FLOPs

    macs, params = get_model_complexity_info(
        model,
        input_res=(1, 1, 1),
        input_constructor=prepare_input,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
    )
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    print("{:<30}  {:<8}".format("Computational complexity: ", 2 * macs))
