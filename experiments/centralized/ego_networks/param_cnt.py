import argparse
import os
import random
import sys
import torch
import numpy as np
from torch_geometric.data import Data

from ptflops import get_model_complexity_info

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from data_preprocessing.ego_networks.data_loader import  get_data
from model.ego_networks.gat import GATNodeCLF
from model.ego_networks.gcn import GCNNodeCLF
from model.ego_networks.sgc import SGCNodeCLF
from model.ego_networks.sage import SAGENodeCLF

from scipy.sparse import rand


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")

    parser.add_argument("--data_dir", type=str, default="./../../../data/ego-networks", help="Data directory")


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
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used between GraphSAGE layers",
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        metavar="LR",
        help="learning rate (default: 0.0015)",
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
    "cora",
    "citeseer",
    "DBLP",
    "PubMed",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)


    if args.dataset not in datasets:
        raise Exception("no such dataset!")
    dataset_path = args.data_dir 

    subgraphs, num_graphs, num_features, num_labels = get_data(dataset_path, args.dataset)
    avg_node = [m.x.size(0) for m in subgraphs]
    avg_node = sum(avg_node) / len(avg_node)
    print("Average number of nodes =", str(avg_node))
    avg_edge = [m.edge_index.size(1) for m in subgraphs]
    avg_edge = sum(avg_edge) / len(avg_edge)
    print("Average number of edges =", str(avg_edge))
    feat_dim = num_features
    num_cats = num_labels
    print("feat_dim = %d" % feat_dim)
    del subgraphs
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

    if args.model == "gcn":
        model = GCNNodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
        prepare_input = lambda res: {"data": Data( x= torch.randn(int(avg_node), feat_dim).to(
                device=device, dtype=torch.float32, non_blocking=True
            ) , edge_index = torch.randn(2, int(avg_edge)).to(
                device=device, dtype=torch.long, non_blocking=True
            ))} 
        
    elif args.model == "sgc":
        model = SGCNodeCLF(in_dim=feat_dim, num_classes=num_cats, K=args.n_layers)
        prepare_input = lambda res: {"inp": Data( x= torch.randn(int(avg_node), feat_dim).to(
                device=device, dtype=torch.float32, non_blocking=True
            ) , edge_index = torch.randn(2, int(avg_edge)).to(
                device=device, dtype=torch.long, non_blocking=True
            ))} 
       
    elif args.model == "sage":
        model = SAGENodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
        prepare_input = lambda res: {"data": Data( x= torch.randn(int(avg_node), feat_dim).to(
                device=device, dtype=torch.float32, non_blocking=True
            ) , edge_index = torch.randn(2, int(avg_edge)).to(
                device=device, dtype=torch.long, non_blocking=True
            ))} 
       
    elif args.model == "gat":
        model = GATNodeCLF(
            in_channels = feat_dim,
            out_channels = num_cats, 
            dropout=args.dropout,
        )
        prepare_input = lambda res: {"inp": Data( x= torch.randn(int(avg_node), feat_dim).to(
                device=device, dtype=torch.float32, non_blocking=True
            ) , edge_index = torch.randn(2, int(avg_edge)).to(
                device=device, dtype=torch.long, non_blocking=True
            ))} 
       
    else:
        raise Exception("No such model")

    model.to(device)
 

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
